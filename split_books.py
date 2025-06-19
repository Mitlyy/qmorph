#!/usr/bin/env python3
"""
split_books.py

Скрипт для разделения текстовых файлов-книг на два корпуса:
  - data/train.txt
  - data/test.txt

Каждое предложение из входных файлов будет отделено по знакам препинания (., !, ?),
отфильтровано, перемешано и разделено в указанном соотношении.
"""

import argparse
import glob
import os
import random
import re
from typing import List


def split_into_sentences(text: str) -> List[str]:
    """
    Разбивает текст на предложения по точкам, восклицательным и вопросительным знакам.
    Обрезает пробелы и удаляет пустые строки.
    """
    parts = re.split(r"(?<=[\.!\?])\s+", text)
    sentences = [p.strip() for p in parts if p and not p.isspace()]
    return sentences


def collect_sentences(input_dir: str, pattern: str = "*.txt") -> List[str]:
    """
    Собирает все предложения из всех .txt-файлов в папке input_dir.
    """
    files = glob.glob(os.path.join(input_dir, pattern))
    all_sentences: List[str] = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        sents = split_into_sentences(text)
        all_sentences.extend(sents)
    return all_sentences


def write_corpus(sentences: List[str], output_path: str) -> None:
    """
    Записывает список предложений в файл, каждое предложение — на новой строке.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for sent in sentences:
            f.write(sent + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Разделяет книги на train/test и размещает в папке data/"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Папка с .txt-файлами-книгами"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data",
        help="Папка, куда будут помещены train.txt и test.txt",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Доля предложений в обучающем корпусе (по умолчанию 0.9)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Случайное зерно для воспроизводимости"
    )
    args = parser.parse_args()

    sentences = collect_sentences(args.input_dir)
    n_total = len(sentences)
    if n_total == 0:
        print(
            f"Не найдено предложений в {args.input_dir} — убедитесь, что там есть .txt-файлы."
        )
        return

    random.seed(args.seed)
    random.shuffle(sentences)

    split_idx = int(n_total * args.train_ratio)
    train_sents = sentences[:split_idx]
    test_sents = sentences[split_idx:]

    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.txt")
    test_path = os.path.join(args.output_dir, "test.txt")

    write_corpus(train_sents, train_path)
    write_corpus(test_sents, test_path)

    print(f"Всего предложений: {n_total}")
    print(f"Записано в {train_path}: {len(train_sents)}")
    print(f"Записано в {test_path}:  {len(test_sents)}")


if __name__ == "__main__":
    main()
