#!/usr/bin/env python3
import argparse
import glob
import json
import os
from collections import defaultdict

from lemmatizer import Lemmatizer
from tokenizer import Tokenizer


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def extend_vocabularies(
    chat_dir: str,
    lemma_vocab_path: str,
    morph_vocab_path: str,
    form_mapping_path: str,
):
    lemma_vocab = load_json(lemma_vocab_path)
    morph_vocab = load_json(morph_vocab_path)
    form_mapping = load_json(form_mapping_path)

    next_lemma_id = max(lemma_vocab.values(), default=0) + 1
    next_morph_id = max(morph_vocab.values(), default=0) + 1

    tok = Tokenizer()
    lem = Lemmatizer()

    chat_files = glob.glob(os.path.join(chat_dir, "*.txt"))
    if not chat_files:
        print(f"Не найдено файлов в {chat_dir}")
        return

    for path in chat_files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not (line.startswith("<user>") or line.startswith("<bot>")):
                    continue
                if " - " not in line:
                    continue
                _, text = line.split(" - ", 1)
                tokens = tok.tokenize(text)
                for token in tokens:
                    lemma, morph_tag = lem.lemmatize_token(token)

                    if lemma not in lemma_vocab:
                        lemma_vocab[lemma] = next_lemma_id
                        next_lemma_id += 1

                    if morph_tag not in morph_vocab:
                        morph_vocab[morph_tag] = next_morph_id
                        next_morph_id += 1

                    lid = str(lemma_vocab[lemma])
                    mid = str(morph_vocab[morph_tag])
                    if lid not in form_mapping:
                        form_mapping[lid] = {}
                    if mid not in form_mapping[lid]:
                        form_mapping[lid][mid] = token

    save_json(lemma_vocab, lemma_vocab_path)
    save_json(morph_vocab, morph_vocab_path)
    save_json(form_mapping, form_mapping_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Расширить существующие словари лемм и морф-тегов по данным чатов"
    )
    parser.add_argument(
        "--chat_dir",
        type=str,
        required=True,
        help="Папка с .txt-файлами чатов",
    )
    parser.add_argument(
        "--lemma_vocab",
        type=str,
        default="vocab/lemma_vocab.json",
        help="Путь к lemma_vocab.json",
    )
    parser.add_argument(
        "--morph_vocab",
        type=str,
        default="vocab/morph_vocab.json",
        help="Путь к morph_vocab.json",
    )
    parser.add_argument(
        "--form_mapping",
        type=str,
        default="vocab/form_mapping.json",
        help="Путь к form_mapping.json",
    )
    args = parser.parse_args()

    extend_vocabularies(
        chat_dir=args.chat_dir,
        lemma_vocab_path=args.lemma_vocab,
        morph_vocab_path=args.morph_vocab,
        form_mapping_path=args.form_mapping,
    )
