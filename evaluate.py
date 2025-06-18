# evaluate.py

import glob
import json
import math
import os
import sys

import torch
import torch.nn as nn
import yaml

sys.path.append(os.path.dirname(__file__))

from src.data_loader import get_dataloader
from src.model.lemma_decoder import LemmaDecoder
from src.model.morph_decoder import MorphDecoder
from src.model.quantum_embedding import QuantumEmbedding
from src.model.transformer_core import TransformerCore
from src.utils import build_vocab_sizes, find_latest_checkpoint, load_config, set_seed


def evaluate(
    embedding: nn.Module,
    transformer: nn.Module,
    lemma_decoder: LemmaDecoder,
    morph_decoder: MorphDecoder,
    dataloader: torch.utils.data.DataLoader,
    lemma_criterion: nn.CrossEntropyLoss,
    morph_criterion: nn.CrossEntropyLoss,
    pad_idx: int,
    device: torch.device,
) -> None:
    """
    Прогон по тестовому датасету, вычисление:
      - среднего лосса и перплексии для next-lemma prediction,
      - точности предсказания лемм,
      - среднего лосса и точности для морфологического декодера.
    """
    embedding.eval()
    transformer.eval()
    lemma_decoder.eval()
    morph_decoder.eval()

    total_lemma_loss = 0.0
    total_morph_loss = 0.0
    total_lemma_tokens = 0
    total_morph_tokens = 0
    correct_lemmas = 0
    correct_morphs = 0

    with torch.no_grad():
        for lemma_ids, morph_ids, lengths in dataloader:
            lemma_ids = lemma_ids.to(device)
            morph_ids = morph_ids.to(device)
            padding_mask = lemma_ids.eq(pad_idx)

            psi, _ = embedding(lemma_ids)
            contextual = transformer(psi, padding_mask)

            lemma_logits = lemma_decoder(contextual)
            B, S, V = lemma_logits.shape

            pred_logits = lemma_logits[:, :-1, :].contiguous().view(-1, V)
            lemma_targets = lemma_ids[:, 1:].contiguous().view(-1)

            loss_lemma = lemma_criterion(pred_logits, lemma_targets)
            mask_lem = lemma_targets != pad_idx
            n_lem_tokens = mask_lem.sum().item()

            total_lemma_loss += loss_lemma.item() * n_lem_tokens
            total_lemma_tokens += n_lem_tokens

            preds = pred_logits.argmax(dim=-1)
            correct_lemmas += (
                (preds == lemma_targets).masked_select(mask_lem).sum().item()
            )

            morph_logits = morph_decoder(contextual)
            _, _, M = morph_logits.shape

            morph_logits_flat = morph_logits.view(-1, M)
            morph_targets = morph_ids.view(-1)

            loss_morph = morph_criterion(morph_logits_flat, morph_targets)
            mask_morph = morph_targets != pad_idx
            n_morph_tokens = mask_morph.sum().item()

            total_morph_loss += loss_morph.item() * n_morph_tokens
            total_morph_tokens += n_morph_tokens

            preds_morph = morph_logits_flat.argmax(dim=-1)
            correct_morphs += (
                (preds_morph == morph_targets).masked_select(mask_morph).sum().item()
            )

    avg_lemma_loss = total_lemma_loss / total_lemma_tokens
    avg_morph_loss = total_morph_loss / total_morph_tokens
    perplexity = math.exp(avg_lemma_loss)
    lemma_accuracy = correct_lemmas / total_lemma_tokens * 100
    morph_accuracy = correct_morphs / total_morph_tokens * 100

    print("Evaluation results:")
    print(f"  Next-lemma loss      : {avg_lemma_loss:.4f}")
    print(f"  Next-lemma perplexity: {perplexity:.2f}")
    print(
        f"  Next-lemma accuracy  : {lemma_accuracy:.2f}% ({correct_lemmas}/{total_lemma_tokens})"
    )
    print(f"  Morph loss           : {avg_morph_loss:.4f}")
    print(
        f"  Morph accuracy       : {morph_accuracy:.2f}% ({correct_morphs}/{total_morph_tokens})"
    )


def main():
    config = load_config("config/config_books.yaml")
    set_seed(config.get("seed", 42))

    device = torch.device(
        "cuda"
        if config["training"]["device"] == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    test_loader = get_dataloader(
        data_file=config["data"]["test_file"],
        lemma_vocab_file=config["vocab"]["lemma_vocab"],
        morph_vocab_file=config["vocab"]["morph_vocab"],
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        max_length=config["training"]["max_length"],
        num_workers=0,
    )

    lemma_vocab_path = config["vocab"]["lemma_vocab"]
    morph_vocab_path = config["vocab"]["morph_vocab"]
    vocab_size, morph_vocab_size = build_vocab_sizes(lemma_vocab_path, morph_vocab_path)

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

    ckpt_path = find_latest_checkpoint(
        config["checkpoint"]["dir"], config["checkpoint"]["prefix"]
    )
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    embedding.load_state_dict(ckpt["embedding_state"])
    transformer.load_state_dict(ckpt["transformer_state"])
    lemma_decoder.load_state_dict(ckpt["lemma_decoder_state"])
    morph_decoder.load_state_dict(ckpt["morph_decoder_state"])

    pad_idx = emb_cfg["pad_idx"]
    lemma_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    morph_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    evaluate(
        embedding,
        transformer,
        lemma_decoder,
        morph_decoder,
        test_loader,
        lemma_criterion,
        morph_criterion,
        pad_idx,
        device,
    )


if __name__ == "__main__":
    main()
