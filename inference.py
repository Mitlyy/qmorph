# inference.py

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import torch

sys.path.append(os.path.dirname(__file__))

from src.model.lemma_decoder import LemmaDecoder
from src.model.morph_decoder import MorphDecoder
from src.model.quantum_embedding import QuantumEmbedding
from src.model.transformer_core import TransformerCore
from src.preprocessing.lemmatizer import Lemmatizer
from src.preprocessing.tokenizer import Tokenizer
from src.utils import build_vocab_sizes, find_latest_checkpoint, load_config, set_seed


def preprocess(
    text: str,
    tokenizer: Tokenizer,
    lemmatizer: Lemmatizer,
    lemma_vocab: dict,
    max_length: int,
):
    """
    Токенизация, лемматизация и преобразование в ID-леммы.
    """
    tokens = tokenizer.tokenize(text)[:max_length]
    lemma_ids = []
    for tok in tokens:
        lemma, _ = lemmatizer.lemmatize_token(tok)
        lemma_ids.append(lemma_vocab.get(lemma, lemma_vocab.get("<unk>", 1)))
    return tokens, torch.tensor([lemma_ids], dtype=torch.long)


def generate_lemmas(
    start_ids: torch.LongTensor,
    embedding: QuantumEmbedding,
    transformer: TransformerCore,
    lemma_decoder: LemmaDecoder,
    cfg: dict,
    device: torch.device,
) -> List[int]:
    """
    Генерация продолжения лемм автодекодером (greedy или beam).
    """
    max_len = cfg["max_gen_length"]
    decoder_type = cfg.get("decoder_type", "greedy")
    if decoder_type == "beam":
        beam_size = cfg.get("beam_size", 5)
        return lemma_decoder.beam_search_decode(
            embedding,
            transformer,
            start_ids.to(device),
            max_length=max_len,
            beam_size=beam_size,
            device=device,
        )
    else:
        return lemma_decoder.greedy_decode(
            embedding,
            transformer,
            start_ids.to(device),
            max_length=max_len,
            device=device,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Inference (generation) для quantum_nlp_project"
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Путь к config.yaml"
    )
    parser.add_argument("--sentence", type=str, help="Начало предложения для генерации")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Путь к файлу чекпоинта. Если не задан, берётся последний в папке checkpoint.dir",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.get("seed", 42))

    device = torch.device(
        "cuda"
        if config["training"]["device"] == "cuda" and torch.cuda.is_available()
        else "cpu"
    )

    lemma_vocab_path = config["vocab"]["lemma_vocab"]
    morph_vocab_path = config["vocab"]["morph_vocab"]
    lemma_vocab = json.load(open(lemma_vocab_path, "r", encoding="utf-8"))
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

    ckpt_path = args.checkpoint or find_latest_checkpoint(
        config["checkpoint"]["dir"], config["checkpoint"]["prefix"]
    )
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    embedding.load_state_dict(ckpt["embedding_state"])
    transformer.load_state_dict(ckpt["transformer_state"])
    lemma_decoder.load_state_dict(ckpt["lemma_decoder_state"])
    morph_decoder.load_state_dict(ckpt["morph_decoder_state"])
    embedding.eval()
    transformer.eval()
    lemma_decoder.eval()
    morph_decoder.eval()

    tokenizer = Tokenizer()
    lemmatizer = Lemmatizer()
    max_len = config["training"]["max_length"]

    if not args.sentence:
        print("Укажите начало предложения через --sentence")
        return

    tokens, start_ids = preprocess(
        args.sentence, tokenizer, lemmatizer, lemma_vocab, max_len
    )
    generated_ids = generate_lemmas(
        start_ids, embedding, transformer, lemma_decoder, config["inference"], device
    )

    full_ids = torch.tensor([generated_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        psi, _ = embedding(full_ids)
        contextual = transformer(psi, full_ids.eq(emb_cfg["pad_idx"]))
        morph_logits = morph_decoder(contextual)
        forms = morph_decoder.decode_forms(full_ids, morph_logits)[0]

    # Вывод
    print(f">>> Input tokens       : {tokens}")
    print(f">>> Generated lemma IDs: {generated_ids}")
    print(f">>> Output sentence    : {' '.join([w for w in forms if w])}")


if __name__ == "__main__":
    main()
