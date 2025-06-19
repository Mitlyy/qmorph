#!/usr/bin/env python3
import glob
import json
import os
import re

import torch
import yaml
from torch.nn.functional import softmax

from src.model.lemma_decoder import LemmaDecoder
from src.model.morph_decoder import MorphDecoder
from src.model.quantum_embedding import QuantumEmbedding
from src.model.transformer_core import TransformerCore
from src.preprocessing.tokenizer import Tokenizer


def extract_after_bot_regex(text: str) -> str:
    match = re.search(r"(?<=<bot>)(.*)", text)
    return match.group(1).strip() if match else ""


def load_config(path="config/config_chat.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_ckpt(ckpt_dir: str, prefix: str):
    pattern = os.path.join(ckpt_dir, f"{prefix}_*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    rx = re.compile(rf"{re.escape(prefix)}_(\d+)\.pt$")
    candidates = []
    for p in files:
        m = rx.search(os.path.basename(p))
        if m:
            candidates.append((int(m.group(1)), p))
    if candidates:
        return max(candidates, key=lambda x: x[0])[1]
    return max(files, key=os.path.getmtime)


def build_model(cfg, device):
    with open(cfg["vocab"]["lemma_vocab"], encoding="utf-8") as f:
        lemma_vocab = json.load(f)
    with open(cfg["vocab"]["morph_vocab"], encoding="utf-8") as f:
        morph_vocab = json.load(f)
    with open(cfg["vocab"]["form_mapping"], encoding="utf-8") as f:
        form_mapping = json.load(f)

    inv_lemma = {v: k for k, v in lemma_vocab.items()}

    emb_c = cfg["model"]["quantum_embedding"]
    embedding = QuantumEmbedding(
        vocab_size=len(lemma_vocab),
        embed_dim=emb_c["embed_dim"],
        n_senses=emb_c["n_senses"],
        pad_idx=emb_c["pad_idx"],
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
        embed_dim=ld_c["embed_dim"],
        vocab_size=len(lemma_vocab),
        pad_idx=ld_c["pad_idx"],
        dropout=ld_c["dropout"],
    ).to(device)

    md_c = cfg["model"]["morph_decoder"]
    morph_decoder = MorphDecoder(
        embed_dim=md_c["embed_dim"],
        morph_vocab_size=len(morph_vocab),
        form_mapping_file=cfg["vocab"]["form_mapping"],
        dropout=md_c["dropout"],
        pad_idx=md_c["pad_idx"],
    ).to(device)

    return (
        embedding,
        transformer,
        lemma_decoder,
        morph_decoder,
        lemma_vocab,
        morph_vocab,
        inv_lemma,
    )


def load_weights(cfg, device, embedding, transformer, lemma_decoder, morph_decoder):
    ckpt_dir = cfg["checkpoint"]["dir"]
    chat_pref = cfg["checkpoint"].get("chat_prefix")
    base_pref = cfg["checkpoint"]["prefix"]

    path = None
    if chat_pref:
        path = find_latest_ckpt(ckpt_dir, chat_pref)
    if not path:
        path = find_latest_ckpt(ckpt_dir, base_pref)
    if not path:
        raise RuntimeError(f"Ни одного чекпоинта не найдено в {ckpt_dir}")

    print(f"Loading checkpoint: {path}")
    ck = torch.load(path, map_location=device)
    embedding.load_state_dict(ck["embedding_state"])
    transformer.load_state_dict(ck["transformer_state"])
    lemma_decoder.load_state_dict(ck["lemma_decoder_state"])
    morph_decoder.load_state_dict(ck["morph_decoder_state"])


@torch.no_grad()
def generate_lemmas(
    seq: torch.LongTensor,
    embedding: QuantumEmbedding,
    transformer: TransformerCore,
    lemma_decoder: LemmaDecoder,
    inf_cfg: dict,
    device,
):
    """
    Генерация лемм до первого стоп-токена.
    """
    max_len = inf_cfg["max_gen_length"]
    temperature = inf_cfg.get("temperature", 1.0)
    top_k = inf_cfg.get("top_k", 50)
    decoder_type = inf_cfg.get("decoder_type", "greedy")

    pad_id = inf_cfg["pad_idx"]
    eos_id = inf_cfg.get("eos_id")
    user_id = inf_cfg.get("user_id")
    bot_id = inf_cfg.get("bot_id")

    output_ids = []
    for _ in range(max_len):
        mask = seq.eq(pad_id)
        psi, _ = embedding(seq)
        context = transformer(psi, mask)
        logits = lemma_decoder(context)[0, -1]  # [V]
        probs = softmax(logits / temperature, dim=-1)

        if decoder_type == "top_k":
            vals, idxs = torch.topk(probs, top_k)
            choice = torch.multinomial(vals, 1).item()
            next_id = idxs[choice].item()
        else:
            next_id = int(probs.argmax())

        # стоп-критерий
        if next_id in (pad_id, eos_id, user_id, bot_id):
            break

        output_ids.append(next_id)
        seq = torch.cat([seq, torch.tensor([[next_id]], device=device)], dim=1)

    return output_ids


def main():
    cfg = load_config()
    device = torch.device(
        "cuda"
        if cfg["training"]["device"] == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    (
        embedding,
        transformer,
        lemma_decoder,
        morph_decoder,
        lemma_vocab,
        morph_vocab,
        inv_lemma,
    ) = build_model(cfg, device)

    load_weights(cfg, device, embedding, transformer, lemma_decoder, morph_decoder)
    embedding.eval()
    transformer.eval()
    lemma_decoder.eval()
    morph_decoder.eval()

    tokenizer = Tokenizer(lowercase=True)
    pad_id = cfg["model"]["quantum_embedding"]["pad_idx"]
    eos_token = cfg["inference"].get("eos_token", "eos")
    user_token = cfg["inference"].get("user_token", "user")
    bot_token = cfg["inference"].get("bot_token", "bot")
    eos_id = lemma_vocab.get(eos_token)
    user_id = lemma_vocab.get(user_token)
    bot_id = lemma_vocab.get(bot_token)

    inf_cfg = cfg["inference"].copy()
    inf_cfg.update(
        {"pad_idx": pad_id, "eos_id": eos_id, "user_id": user_id, "bot_id": bot_id}
    )

    print("=== Chat mode (type 'exit' or Ctrl-C) ===")
    while True:
        text = input("User: ").strip()
        if not text or text.lower() in ("exit", "quit"):
            break

        tokens = tokenizer.tokenize(text)
        start_ids = [
            lemma_vocab.get(t, lemma_vocab.get("<unk>", pad_id)) for t in tokens
        ]
        print(start_ids)
        seq = torch.tensor([start_ids + [user_id]], dtype=torch.long, device=device)

        new_ids = generate_lemmas(
            seq, embedding, transformer, lemma_decoder, inf_cfg, device
        )
        full_ids = torch.tensor([start_ids + new_ids], dtype=torch.long, device=device)
        print(full_ids)
        mask = full_ids.eq(pad_id)
        psi, _ = embedding(full_ids)
        context = transformer(psi, mask)

        morph_logits = morph_decoder(context)
        forms = morph_decoder.decode_forms(full_ids, morph_logits)[0]

        out_tokens = []
        pad_id = cfg["model"]["quantum_embedding"]["pad_idx"]
        eos_id = lemma_vocab.get(cfg["inference"].get("eos_token", "eos"))
        user_id = lemma_vocab.get(cfg["inference"].get("user_token", "user"))
        bot_id = lemma_vocab.get(cfg["inference"].get("bot_token", "bot"))

        for lid, form in zip(full_ids[0].tolist(), forms):
            if form is None or form == "" or form == "<pad>":
                break
            if lid in (pad_id, eos_id, user_id, bot_id):
                break

            raw = form.strip("<>")
            if raw.startswith("lemma_"):
                idx = int(raw.split("_", 1)[1])
                w = inv_lemma.get(idx, "<unk>")
            else:
                w = form

            out_tokens.append(w)

        print("Bot:", " ".join(out_tokens))
        print("Bot:", extract_after_bot_regex(" ".join(out_tokens)))

    print("Bye")


if __name__ == "__main__":
    main()
