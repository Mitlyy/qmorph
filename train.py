# train.py

import glob
import json
import os
import sys

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from src.data_loader import get_dataloader
from src.model.lemma_decoder import LemmaDecoder
from src.model.morph_decoder import MorphDecoder
from src.model.quantum_embedding import QuantumEmbedding
from src.model.transformer_core import TransformerCore


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_vocab_sizes(lemma_vocab_path: str, morph_vocab_path: str) -> (int, int):
    with open(lemma_vocab_path, "r", encoding="utf-8") as f:
        lemma_vocab = json.load(f)
    with open(morph_vocab_path, "r", encoding="utf-8") as f:
        morph_vocab = json.load(f)
    vocab_size = max(lemma_vocab.values(), default=0) + 1
    morph_vocab_size = max(morph_vocab.values(), default=0) + 1
    return vocab_size, morph_vocab_size


def main():
    sys.path.append(os.path.dirname(__file__))
    config = load_config("config/config_books.yaml")

    use_cuda = config["training"]["device"] == "cuda" and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(config.get("seed", 42))

    train_loader = get_dataloader(
        data_file=config["data"]["train_file"],
        lemma_vocab_file=config["vocab"]["lemma_vocab"],
        morph_vocab_file=config["vocab"]["morph_vocab"],
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        max_length=config["training"]["max_length"],
        num_workers=0,
    )

    vocab_size, morph_vocab_size = build_vocab_sizes(
        config["vocab"]["lemma_vocab"], config["vocab"]["morph_vocab"]
    )

    emb_cfg = config["model"]["quantum_embedding"]
    embedding = QuantumEmbedding(
        vocab_size=vocab_size,
        embed_dim=emb_cfg["embed_dim"],
        n_senses=emb_cfg["n_senses"],
        pad_idx=emb_cfg["pad_idx"],
    ).to(device)

    trf_cfg = config["model"]["transformer"]
    transformer = TransformerCore(
        embed_dim=trf_cfg["embed_dim"],
        num_heads=trf_cfg["num_heads"],
        ff_dim=trf_cfg["ff_dim"],
        num_layers=trf_cfg["num_layers"],
        dropout=trf_cfg["dropout"],
        max_seq_len=trf_cfg["max_seq_len"],
        is_autoregressive=True,
    ).to(device)

    ld_cfg = config["model"]["lemma_decoder"]
    lemma_decoder = LemmaDecoder(
        embed_dim=ld_cfg["embed_dim"],
        vocab_size=vocab_size,
        pad_idx=ld_cfg["pad_idx"],
        dropout=ld_cfg["dropout"],
    ).to(device)

    dec_cfg = config["model"]["morph_decoder"]
    morph_decoder = MorphDecoder(
        embed_dim=dec_cfg["embed_dim"],
        morph_vocab_size=morph_vocab_size,
        form_mapping_file=dec_cfg["form_mapping"],
        dropout=dec_cfg["dropout"],
        pad_idx=dec_cfg["pad_idx"],
    ).to(device)

    model_params = (
        list(embedding.parameters())
        + list(transformer.parameters())
        + list(lemma_decoder.parameters())
        + list(morph_decoder.parameters())
    )
    optim_cfg = config["optimizer"]
    opt_params = optim_cfg["params"].copy()
    opt_params["lr"] = float(opt_params["lr"])
    opt_params["eps"] = float(opt_params["eps"])
    opt_params["weight_decay"] = float(opt_params["weight_decay"])
    opt_params["betas"] = tuple(opt_params["betas"])
    optimizer = AdamW(model_params, **opt_params)

    sched_cfg = config["scheduler"]["params"]
    warmup_steps = sched_cfg["warmup_steps"]
    total_steps = sched_cfg["total_steps"]
    scheduler = LambdaLR(
        optimizer,
        lambda step: (
            step / warmup_steps
            if step < warmup_steps
            else max(0.0, (total_steps - step) / (total_steps - warmup_steps))
        ),
    )

    pad_idx = dec_cfg["pad_idx"]
    morph_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    lemma_criterion = nn.CrossEntropyLoss(ignore_index=ld_cfg["pad_idx"])

    ckpt_dir = config["checkpoint"]["dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    prefix = config["checkpoint"]["prefix"]
    keep_last = config["checkpoint"]["keep_last"]

    log_interval = config["training"]["log_interval"]
    save_every = config["training"]["save_every"]
    global_step = 0

    num_epochs = config["training"]["num_epochs"]
    start_epoch = 1
    latest_ckpt = None
    all_ckpts = sorted(glob.glob(os.path.join(ckpt_dir, f"{prefix}_epoch_*.pt")))
    if all_ckpts:
        latest_ckpt = all_ckpts[-1]
        print(f"Resuming from checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)

        embedding.load_state_dict(checkpoint["embedding_state"])
        transformer.load_state_dict(checkpoint["transformer_state"])
        lemma_decoder.load_state_dict(checkpoint["lemma_decoder_state"])
        morph_decoder.load_state_dict(checkpoint["morph_decoder_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
    else:
        print("No checkpoint found. Starting from scratch.")
        global_step = 0

    for epoch in range(start_epoch, num_epochs + 1):
        embedding.train()
        transformer.train()
        lemma_decoder.train()
        morph_decoder.train()

        for batch_idx, (lemma_ids, morph_ids, lengths) in enumerate(train_loader, 1):
            lemma_ids = lemma_ids.to(device)
            morph_ids = morph_ids.to(device)
            padding_mask = lemma_ids.eq(emb_cfg["pad_idx"])

            optimizer.zero_grad()

            psi, _ = embedding(lemma_ids)
            contextual = transformer(psi, padding_mask)

            lemma_logits = lemma_decoder(contextual)  # [B, S, V]
            pred_logits = lemma_logits[:, :-1, :].reshape(-1, vocab_size)
            targets = lemma_ids[:, 1:].reshape(-1)
            lemma_loss = lemma_criterion(pred_logits, targets)

            morph_logits = morph_decoder(contextual)
            B, S, M = morph_logits.shape
            morph_loss = morph_criterion(
                morph_logits.view(B * S, M), morph_ids.view(B * S)
            )

            loss = lemma_loss + morph_loss
            loss.backward()

            if config["training"]["grad_clip"]:
                nn.utils.clip_grad_norm_(model_params, config["training"]["grad_clip"])

            optimizer.step()
            scheduler.step()
            global_step += 1

            if batch_idx % log_interval == 0:
                print(
                    f"[Epoch {epoch}/{num_epochs}] "
                    f"[Batch {batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"(lemma: {lemma_loss.item():.4f}, morph: {morph_loss.item():.4f})"
                )

        if epoch % save_every == 0:
            ckpt_path = os.path.join(ckpt_dir, f"{prefix}_epoch_{epoch:02d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "embedding_state": embedding.state_dict(),
                    "transformer_state": transformer.state_dict(),
                    "lemma_decoder_state": lemma_decoder.state_dict(),
                    "morph_decoder_state": morph_decoder.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                },
                ckpt_path,
            )

            all_ckpts = sorted(
                glob.glob(os.path.join(ckpt_dir, f"{prefix}_epoch_*.pt"))
            )
            if len(all_ckpts) > keep_last:
                for old in all_ckpts[:-keep_last]:
                    os.remove(old)

            print(f"Saved checkpoint: {ckpt_path}")
            torch.cuda.empty_cache()
    print("Training complete.")


if __name__ == "__main__":
    main()
