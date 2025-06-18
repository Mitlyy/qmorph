# src/data_loader.py

import json
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocessing.lemmatizer import Lemmatizer
from src.preprocessing.tokenizer import Tokenizer


class QuantumNLPDataset(Dataset):
    """
    PyTorch Dataset для загрузки предложений и преобразования их в ID лемм и морфологических тегов.
    """

    def __init__(
        self,
        data_file: str,
        lemma_vocab_file: str,
        morph_vocab_file: str,
        max_length: int = None,
    ):
        """
        :param data_file: путь к файлу (train.txt или test.txt), каждая строка — одно предложение
        :param lemma_vocab_file: путь к lemma_vocab.json
        :param morph_vocab_file: путь к morph_vocab.json
        :param max_length: максимальная длина последовательности (для обрезки)
        """
        with open(lemma_vocab_file, "r", encoding="utf-8") as f:
            self.lemma_vocab: Dict[str, int] = json.load(f)
        with open(morph_vocab_file, "r", encoding="utf-8") as f:
            self.morph_vocab: Dict[str, int] = json.load(f)

        self.pad_lemma_id = self.lemma_vocab.get("<pad>", 0)
        self.unk_lemma_id = self.lemma_vocab.get("<unk>", 1)
        self.pad_morph_id = self.morph_vocab.get("<pad>", 0)
        self.unk_morph_id = self.morph_vocab.get("<unk>", 1)

        self.max_length = max_length

        self.tokenizer = Tokenizer()
        self.lemmatizer = Lemmatizer()

        self.samples: List[Tuple[List[int], List[int]]] = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                tokens = self.tokenizer.tokenize(line)
                lemma_ids: List[int] = []
                morph_ids: List[int] = []

                for tok in tokens:
                    lemma, morph_tag = self.lemmatizer.lemmatize_token(tok)
                    lemma_id = self.lemma_vocab.get(lemma, self.unk_lemma_id)
                    morph_id = self.morph_vocab.get(morph_tag, self.unk_morph_id)
                    lemma_ids.append(lemma_id)
                    morph_ids.append(morph_id)

                if self.max_length:
                    lemma_ids = lemma_ids[: self.max_length]
                    morph_ids = morph_ids[: self.max_length]

                self.samples.append((lemma_ids, morph_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        lemma_ids, morph_ids = self.samples[idx]
        length = len(lemma_ids)
        return {
            "lemma_ids": torch.tensor(lemma_ids, dtype=torch.long),
            "morph_ids": torch.tensor(morph_ids, dtype=torch.long),
            "length": length,
        }


def collate_fn(
    batch: List[Dict[str, Any]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Функция для объединения батча с динамическим паддингом.
    :param batch: список элементов, полученных из __getitem__
    :return:
        - batch_lemmas: Tensor (batch_size, max_seq_len)
        - batch_morphs: Tensor (batch_size, max_seq_len)
        - lengths:     Tensor (batch_size,)
    """
    lengths = [item["length"] for item in batch]
    max_len = max(lengths)

    batch_size = len(batch)
    batch_lemmas = torch.full((batch_size, max_len), 0, dtype=torch.long)
    batch_morphs = torch.full((batch_size, max_len), 0, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["length"]
        batch_lemmas[i, :seq_len] = item["lemma_ids"]
        batch_morphs[i, :seq_len] = item["morph_ids"]

    return batch_lemmas, batch_morphs, torch.tensor(lengths, dtype=torch.long)


def get_dataloader(
    data_file: str,
    lemma_vocab_file: str,
    morph_vocab_file: str,
    batch_size: int,
    shuffle: bool = True,
    max_length: int = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Утилита для создания DataLoader'а.
    :param data_file: путь к train.txt или test.txt
    :param lemma_vocab_file: путь к lemma_vocab.json
    :param morph_vocab_file: путь к morph_vocab.json
    :param batch_size: размер батча
    :param shuffle: перемешивать ли данные (для train)
    :param max_length: максимальная длина последовательности
    :param num_workers: количество воркеров для DataLoader
    :return: DataLoader
    """
    dataset = QuantumNLPDataset(
        data_file=data_file,
        lemma_vocab_file=lemma_vocab_file,
        morph_vocab_file=morph_vocab_file,
        max_length=max_length,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DataLoader для quantum_nlp_project")
    parser.add_argument(
        "--data_file",
        type=str,
        default="../data/train.txt",
        help="Путь к файлу train.txt или test.txt",
    )
    parser.add_argument(
        "--lemma_vocab",
        type=str,
        default="../vocab/lemma_vocab.json",
        help="Путь к lemma_vocab.json",
    )
    parser.add_argument(
        "--morph_vocab",
        type=str,
        default="../vocab/morph_vocab.json",
        help="Путь к morph_vocab.json",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Размер батча")
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Максимальная длина последовательности (опционально)",
    )
    args = parser.parse_args()

    loader = get_dataloader(
        data_file=args.data_file,
        lemma_vocab_file=args.lemma_vocab,
        morph_vocab_file=args.morph_vocab,
        batch_size=args.batch_size,
        shuffle=False,
        max_length=args.max_length,
    )

    for batch_idx, (lemmas, morphs, lengths) in enumerate(loader):
        print(f"Batch {batch_idx}:")
        print("  Lemmas:", lemmas.shape)
        print("  Morphs:", morphs.shape)
        print("  Lengths:", lengths)
        break
