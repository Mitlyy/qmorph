#!/usr/bin/env python3
# chat_split.py

import argparse
import os
import random


def load_dialogues(path: str):
    text = open(path, encoding="utf-8").read().strip()
    parts = [d.strip() for d in text.split("\n\n") if d.strip()]
    return parts


def write_list(lines, path):
    with open(path, "w", encoding="utf-8") as f:
        for dlg in lines:
            f.write(dlg.replace("\n", " ") + "<eos>\n\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("infile", help="файл с диалогами, разделёнными пустой строкой")
    p.add_argument(
        "--out_dir",
        default="data/data_chat_2",
        help="куда положить chat_train.txt/chat_test.txt",
    )
    p.add_argument("--ratio", type=float, default=0.9, help="доля для тренировки")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dialogs = load_dialogues(args.infile)
    random.seed(args.seed)
    random.shuffle(dialogs)
    split = int(len(dialogs) * args.ratio)
    train, test = dialogs[:split], dialogs[split:]

    write_list(train, os.path.join(args.out_dir, "train.txt"))
    write_list(test, os.path.join(args.out_dir, "test.txt"))

    print(f"Диалогов всего: {len(dialogs)},  train={len(train)}, test={len(test)}")


if __name__ == "__main__":
    main()
