#!/usr/bin/env python3

import json
import os
from typing import Dict, List, Tuple

import torch

from .lemmatizer import Lemmatizer
from .tokenizer import Tokenizer


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_chat_lines(chat_path: str) -> List[str]:
    """
    Считывает файл диалогов, отбрасывает пустые строки и
    префиксы '<user> -' / '<bot> -', возвращает список чистых строк.
    """
    out = []
    with open(chat_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for tag in ("<user>", "<bot>"):
                if line.startswith(tag):
                    line = line[len(tag) :].lstrip(" -")
                    break
            out.append(line)
    return out


def _enhance_and_cache(
    chat_file: str,
    lemma_vocab: Dict[str, int],
    morph_vocab: Dict[str, int],
    form_mapping: Dict[str, Dict[str, str]],
    output_dir: str,
) -> None:
    """
    Разбирает чат, дополняет словари (предварительно в них уже есть <pad>/<unk>) и
    сохраняет их + кэш токенов в output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = Tokenizer(lowercase=True)
    lemmatizer = Lemmatizer()

    next_lemma_id = max(lemma_vocab.values(), default=-1) + 1
    next_morph_id = max(morph_vocab.values(), default=-1) + 1

    cache = []

    for line in parse_chat_lines(chat_file):
        tokens = tokenizer.tokenize(line)
        lemma_ids, morph_ids = [], []

        for tok in tokens:
            # если это маркер, оставляем его «как есть»
            if tok in ("<user>", "<bot>"):
                lemma, morph = tok, tok
            else:
                lemma, morph = lemmatizer.lemmatize([tok])[0]

            if lemma not in lemma_vocab:
                lemma_vocab[lemma] = next_lemma_id
                form_mapping[str(next_lemma_id)] = {}
                next_lemma_id += 1
            lid = lemma_vocab[lemma]

            if morph not in morph_vocab:
                morph_vocab[morph] = next_morph_id
                next_morph_id += 1
            mid = morph_vocab[morph]

            form_mapping[str(lid)][str(mid)] = tok

            lemma_ids.append(lid)
            morph_ids.append(mid)

        cache.append({"lemma_ids": lemma_ids, "morph_ids": morph_ids})

    save_json(lemma_vocab, os.path.join(output_dir, "lemma_vocab.json"))
    save_json(morph_vocab, os.path.join(output_dir, "morph_vocab.json"))
    save_json(form_mapping, os.path.join(output_dir, "form_mapping.json"))
    torch.save(cache, os.path.join(output_dir, "chat_cache.pt"))


def load_or_enhance_vocabs(
    lemma_vocab_file: str,
    morph_vocab_file: str,
    form_mapping_file: str,
    chat_file: str,
    cache_dir: str = "vocabs/vocab_chat",
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, Dict[str, str]]]:
    """
    Если в cache_dir уже есть chat_cache.pt — просто подгружает
    обновлённые словари оттуда. Иначе —
    1) добавляет в исходные словари спец-токены <pad> и <unk>,
    2) докидывает туда новые токены из chat_file и кэширует их,
    3) возвращает итоговые словари.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_pt = os.path.join(cache_dir, "chat_cache.pt")

    lemma_vocab = load_json(lemma_vocab_file)
    morph_vocab = load_json(morph_vocab_file)
    form_mapping = load_json(form_mapping_file)

    pad_token = "<pad>"
    unk_token = "<unk>"
    pad_idx = 0

    max_lid = max(lemma_vocab.values(), default=-1)
    max_mid = max(morph_vocab.values(), default=-1)
    next_lid = max_lid + 1
    next_mid = max_mid + 1

    if pad_token not in lemma_vocab:
        lemma_vocab[pad_token] = pad_idx
        form_mapping[str(pad_idx)] = {}
    if pad_token not in morph_vocab:
        morph_vocab[pad_token] = pad_idx

    if unk_token not in lemma_vocab:
        lemma_vocab[unk_token] = next_lid
        form_mapping[str(next_lid)] = {}
        next_lid += 1
    if unk_token not in morph_vocab:
        morph_vocab[unk_token] = next_mid
        next_mid += 1

    if not os.path.exists(cache_pt):
        print("Расширяю словари и кэширую токены…")
        _enhance_and_cache(chat_file, lemma_vocab, morph_vocab, form_mapping, cache_dir)
    else:
        print(f"Загружаю кэш из {cache_pt}")

    lemma_vocab = load_json(os.path.join(cache_dir, "lemma_vocab.json"))
    morph_vocab = load_json(os.path.join(cache_dir, "morph_vocab.json"))
    form_mapping = load_json(os.path.join(cache_dir, "form_mapping.json"))

    return lemma_vocab, morph_vocab, form_mapping
