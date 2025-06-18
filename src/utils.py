# src/utils.py

import glob
import json
import os
import random

import numpy as np
import torch
import yaml


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


def find_latest_checkpoint(ckpt_dir: str, prefix: str) -> str:
    pattern = os.path.join(ckpt_dir, f"{prefix}_epoch_*.pt")
    ckpts = sorted(glob.glob(pattern))
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoints found in {ckpt_dir} with prefix {prefix}"
        )
    return ckpts[-1]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
