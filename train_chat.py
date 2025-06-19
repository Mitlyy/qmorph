#!/usr/bin/env python3
import glob
import logging
import os
import pickle

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from src.model.lemma_decoder import LemmaDecoder
from src.model.morph_decoder import MorphDecoder
from src.model.quantum_embedding import QuantumEmbedding
from src.model.transformer_core import TransformerCore
from src.preprocessing.enchance_vocabs import load_or_enhance_vocabs
from src.preprocessing.lemmatizer import Lemmatizer
from src.preprocessing.tokenizer import Tokenizer

logging.basicConfig(level=logging.INFO)


def partial_load_state(module: nn.Module, old_state: dict):
    msd = module.state_dict()
    for name, old_param in old_state.items():
        if name not in msd:
            continue
        new_param = msd[name]
        if old_param.shape == new_param.shape:
            new_param.copy_(old_param)
        elif (
            old_param.ndim == new_param.ndim
            and old_param.shape[1:] == new_param.shape[1:]
            and old_param.shape[0] <= new_param.shape[0]
        ):
            new_param[: old_param.shape[0]].copy_(old_param)
    module.load_state_dict(msd)


class ChatDataset(Dataset):
    def __init__(
        self,
        chat_file: str,
        lemma_vocab_file: str,
        morph_vocab_file: str,
        form_mapping_file: str,
        max_length: int,
        cache_dir: str,
    ):
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, os.path.basename(chat_file) + ".pkl")

        if os.path.exists(cache_path):
            logging.info(f"Loading cached dataset from {cache_path}")
            with open(cache_path, "rb") as f:
                self.data = pickle.load(f)
            return

        logging.info("Building dataset from scratch…")
        lemma_vocab, morph_vocab, form_mapping = load_or_enhance_vocabs(
            lemma_vocab_file, morph_vocab_file, form_mapping_file, chat_file, cache_dir
        )
        tokenizer = Tokenizer(lowercase=True)
        lemmatizer = Lemmatizer()

        pad_id = lemma_vocab["<pad>"]
        unk_id = lemma_vocab.get("<unk>", pad_id)

        self.data = []
        raw = open(chat_file, "r", encoding="utf-8").read().strip()
        dialogs = [d for d in raw.split("\n\n") if d.strip()]

        for dialog in dialogs:
            for line in dialog.splitlines():
                if "-" not in line:
                    continue
                _, utter = line.split("-", 1)
                tokens = tokenizer.tokenize(utter)
                lem_morph = lemmatizer.lemmatize(tokens)
                lemma_ids = [lemma_vocab.get(l, unk_id) for l, _ in lem_morph]
                morph_ids = [
                    morph_vocab.get(m, morph_vocab.get("<unk>", pad_id))
                    for _, m in lem_morph
                ]
                L = len(lemma_ids)
                if L > max_length:
                    lemma_ids = lemma_ids[:max_length]
                    morph_ids = morph_ids[:max_length]
                    L = max_length
                else:
                    lemma_ids += [pad_id] * (max_length - L)
                    morph_ids += [pad_id] * (max_length - L)

                self.data.append(
                    (
                        torch.tensor(lemma_ids, dtype=torch.long),
                        torch.tensor(morph_ids, dtype=torch.long),
                        L,
                    )
                )

        with open(cache_path, "wb") as f:
            pickle.dump(self.data, f)
        logging.info(f"Cached dataset to {cache_path} (size={len(self.data)})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ChatModel(nn.Module):
    def __init__(self, embedding, transformer, lemma_decoder, morph_decoder):
        super().__init__()
        self.embedding = embedding
        self.transformer = transformer
        self.lemma_decoder = lemma_decoder
        self.morph_decoder = morph_decoder

    def forward(self, lemma_ids):
        mask = lemma_ids.eq(self.embedding.pad_idx)
        psi, _ = self.embedding(lemma_ids)
        contextual = self.transformer(psi, mask)
        return self.lemma_decoder(contextual), self.morph_decoder(contextual)


def main():
    cfg = yaml.safe_load(open("config/config_chat.yaml", encoding="utf-8"))
    torch.manual_seed(cfg.get("seed", 42))
    device = torch.device(
        "cuda"
        if cfg["training"]["device"] == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    # Датасеты
    train_ds = ChatDataset(
        cfg["data"]["train_file"],
        cfg["vocab"]["lemma_vocab"],
        cfg["vocab"]["morph_vocab"],
        cfg["vocab"]["form_mapping"],
        cfg["training"]["max_length"],
        cfg["data"]["cache_dir"],
    )
    valid_ds = ChatDataset(
        cfg["data"]["test_file"],
        cfg["vocab"]["lemma_vocab"],
        cfg["vocab"]["morph_vocab"],
        cfg["vocab"]["form_mapping"],
        cfg["training"]["max_length"],
        cfg["data"]["cache_dir"],
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg["training"]["batch_size"], shuffle=False
    )

    lemma_vocab, morph_vocab, _ = load_or_enhance_vocabs(
        cfg["vocab"]["lemma_vocab"],
        cfg["vocab"]["morph_vocab"],
        cfg["vocab"]["form_mapping"],
        cfg["data"]["train_file"],
        cfg["data"]["cache_dir"],
    )
    vocab_size = max(lemma_vocab.values()) + 1
    morph_vocab_size = max(morph_vocab.values()) + 1

    emb_c = cfg["model"]["quantum_embedding"]
    embedding = QuantumEmbedding(
        vocab_size, emb_c["embed_dim"], emb_c["n_senses"], emb_c["pad_idx"]
    ).to(device)
    trf_c = cfg["model"]["transformer"]
    transformer = TransformerCore(
        embed_dim=trf_c["embed_dim"],
        num_heads=trf_c["num_heads"],
        ff_dim=trf_c["ff_dim"],
        num_layers=trf_c["num_layers"],
        dropout=trf_c["dropout"],
        max_seq_len=trf_c["max_seq_len"],
        is_autoregressive=True,
    ).to(device)
    ld_c = cfg["model"]["lemma_decoder"]
    lemma_decoder = LemmaDecoder(
        ld_c["embed_dim"], vocab_size, ld_c["pad_idx"], ld_c["dropout"]
    ).to(device)
    md_c = cfg["model"]["morph_decoder"]
    morph_decoder = MorphDecoder(
        md_c["embed_dim"],
        morph_vocab_size,
        md_c["form_mapping"],
        md_c["dropout"],
        md_c["pad_idx"],
    ).to(device)

    model = ChatModel(embedding, transformer, lemma_decoder, morph_decoder).to(device)

    for p in model.embedding.parameters():
        p.requires_grad = False
    logging.info("Frozen embedding; Transformer + decoders are trainable")

    ckpt_dir = cfg["checkpoint"]["dir"]
    base_pref = cfg["checkpoint"]["prefix"]
    chat_pref = cfg["checkpoint"]["chat_prefix"]

    chat_ckpts = sorted(glob.glob(f"{ckpt_dir}/{chat_pref}_*.pt"))
    if chat_ckpts:
        last = chat_ckpts[-1]
        logging.info(f"Resuming fine-tune from {last}")
        ck = torch.load(last, map_location=device)
        partial_load_state(model.embedding, ck["embedding_state"])
        model.transformer.load_state_dict(ck["transformer_state"], strict=False)
        partial_load_state(model.lemma_decoder, ck["lemma_decoder_state"])
        partial_load_state(model.morph_decoder, ck["morph_decoder_state"])
        optimizer_state = ck["optimizer_state"]
        scheduler_state = ck["scheduler_state"]
        start_epoch = ck["epoch"] + 1
    else:
        base_ckpts = sorted(glob.glob(f"{ckpt_dir}/{base_pref}_*.pt"))
        if base_ckpts:
            last = base_ckpts[-1]
            logging.info(f"Loading base model from {last}")
            ck = torch.load(last, map_location=device)
            partial_load_state(model.embedding, ck["embedding_state"])
            model.transformer.load_state_dict(ck["transformer_state"], strict=False)
            partial_load_state(model.lemma_decoder, ck["lemma_decoder_state"])
            partial_load_state(model.morph_decoder, ck["morph_decoder_state"])
        else:
            logging.info("No checkpoint found, training from scratch")
        start_epoch = 1

    opt = cfg["optimizer"]["params"]
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(opt["lr"]),
        betas=tuple(opt["betas"]),
        eps=float(opt["eps"]),
        weight_decay=float(opt["weight_decay"]),
    )
    sched = cfg["scheduler"]["params"]
    scheduler = LambdaLR(
        optimizer,
        lambda step: (
            step / sched["warmup_steps"]
            if step < sched["warmup_steps"]
            else max(
                0.0,
                (sched["total_steps"] - step)
                / (sched["total_steps"] - sched["warmup_steps"]),
            )
        ),
    )
    if "optimizer_state" in locals():
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)

    lemma_loss_fn = nn.CrossEntropyLoss(ignore_index=emb_c["pad_idx"])
    morph_loss_fn = nn.CrossEntropyLoss(ignore_index=md_c["pad_idx"])

    for epoch in range(start_epoch, cfg["training"]["num_epochs"] + 1):
        model.train()
        for i, (lemma_ids, morph_ids, lengths) in enumerate(train_loader, 1):
            lemma_ids = lemma_ids.to(device)
            morph_ids = morph_ids.to(device)

            optimizer.zero_grad()
            logits_l, logits_m = model(lemma_ids)

            B, S, V = logits_l.size()
            l_logits = logits_l[:, :-1].reshape(-1, V)
            l_targs = lemma_ids[:, 1:].reshape(-1)
            loss_l = lemma_loss_fn(l_logits, l_targs)

            _, _, M = logits_m.size()
            m_logits = logits_m.view(-1, M)
            m_targs = morph_ids.view(-1)
            loss_m = morph_loss_fn(m_logits, m_targs)

            loss = loss_l + loss_m
            loss.backward()
            optimizer.step()
            scheduler.step()

            if i % cfg["training"]["log_interval"] == 0:
                logging.info(
                    f"[{epoch}] {i}/{len(train_loader)}  loss={loss.item():.4f}"
                )

        if epoch % cfg["training"]["save_every"] == 0:
            path = f"{ckpt_dir}/{chat_pref}_{epoch:02d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "embedding_state": model.embedding.state_dict(),
                    "transformer_state": model.transformer.state_dict(),
                    "lemma_decoder_state": model.lemma_decoder.state_dict(),
                    "morph_decoder_state": model.morph_decoder.state_dict(),
                },
                path,
            )
            logging.info(f"Saved chat checkpoint: {path}")

    logging.info("Fine-tuning complete")


if __name__ == "__main__":
    main()
