import json
import os
import sys
from collections import defaultdict
from typing import Dict

from lemmatizer import Lemmatizer
from tokenizer import Tokenizer


def build_vocab(train_file: str, output_dir: str) -> None:
    """
    Построение словарей лемм, морфологических тегов и отображения форм слова.
    :param train_file: путь к файлу с обучающими данными (каждая строка — предложение)
    :param output_dir: директория для сохранения JSON-файлов
    """
    tokenizer = Tokenizer()
    lemmatizer = Lemmatizer()

    lemma_set = set()
    morph_set = set()
    form_mapping = defaultdict(dict)

    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = tokenizer.tokenize(line)
            for token in tokens:
                lemma, morph_tag = lemmatizer.lemmatize_token(token)
                lemma_set.add(lemma)
                morph_set.add(morph_tag)
                if morph_tag not in form_mapping[lemma]:
                    form_mapping[lemma][morph_tag] = token

    lemma_vocab = {lemma: idx for idx, lemma in enumerate(sorted(lemma_set), start=1)}
    morph_vocab = {tag: idx for idx, tag in enumerate(sorted(morph_set), start=1)}

    form_mapping_id: Dict[int, Dict[int, str]] = {}
    for lemma, tag_dict in form_mapping.items():
        lemma_id = lemma_vocab[lemma]
        form_mapping_id[lemma_id] = {
            morph_vocab[tag]: form for tag, form in tag_dict.items()
        }

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "lemma_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(lemma_vocab, f, ensure_ascii=False, indent=4)
    with open(os.path.join(output_dir, "morph_vocab.json"), "w", encoding="utf-8") as f:
        json.dump(morph_vocab, f, ensure_ascii=False, indent=4)
    with open(
        os.path.join(output_dir, "form_mapping.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(form_mapping_id, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Построение словарей для quantum_nlp_project"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="../data/train.txt",
        help="Путь к data/train.txt",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../vocabs/vocab_books",
        help="Папка для JSON-словари",
    )
    args = parser.parse_args()
    build_vocab(args.train_file, args.output_dir)
