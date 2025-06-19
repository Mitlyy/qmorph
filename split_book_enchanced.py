#!/usr/bin/env python3

import argparse
import glob
import os
import random
import re
from typing import List


def split_into_paragraphs(text: str) -> List[str]:
    raw_paragraphs = re.split(r"\n\s*\n+", text)
    paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]
    return paragraphs


def collect_paragraphs(input_dir: str, pattern: str = "*.txt") -> List[str]:
    files = glob.glob(os.path.join(input_dir, pattern))
    all_paragraphs = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        paragraphs = split_into_paragraphs(text)
        all_paragraphs.extend(paragraphs)
    return all_paragraphs


def write_corpus(paragraphs: List[str], output_path: str) -> None:
    with open(output_path, "w", encoding="utf-8") as f:
        for para in paragraphs:
            f.write(para.strip() + "\n\n")


def main():
    parser = argparse.ArgumentParser(
        description="Разделяет книги на train/test по абзацам."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Папка с .txt-файлами"
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/data1", help="Папка вывода"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.9, help="Доля для train.txt"
    )
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно")

    args = parser.parse_args()

    paragraphs = collect_paragraphs(args.input_dir)
    total = len(paragraphs)
    if total == 0:
        print(f"Не найдено абзацев в {args.input_dir}. Проверьте наличие .txt-файлов.")
        return

    random.seed(args.seed)
    random.shuffle(paragraphs)

    split_idx = int(total * args.train_ratio)
    train_paragraphs = paragraphs[:split_idx]
    test_paragraphs = paragraphs[split_idx:]

    os.makedirs(args.output_dir, exist_ok=True)
    write_corpus(train_paragraphs, os.path.join(args.output_dir, "train.txt"))
    write_corpus(test_paragraphs, os.path.join(args.output_dir, "test.txt"))

    print(f"Всего абзацев: {total}")
    print(f"Записано в train.txt: {len(train_paragraphs)}")
    print(f"Записано в test.txt:  {len(test_paragraphs)}")


if __name__ == "__main__":
    main()
