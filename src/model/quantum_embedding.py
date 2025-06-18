from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumEmbedding(nn.Module):
    """
    Квантово-подобные эмбеддинги: каждое слово представлено суперпозицией
    K "sense"-векторов с амплитудами вероятностей.
    При инференсе возвращает и вектор коллапса (psi), и распределение вероятностей.
    """

    def __init__(
        self, vocab_size: int, embed_dim: int, n_senses: int = 4, pad_idx: int = 0
    ):
        """
        :param vocab_size: размер словаря (количество лемм)
        :param embed_dim: размерность векторного пространства
        :param n_senses: число базисных смыслов (sense-vectors) на слово
        :param pad_idx: индекс паддинга (будет возвращать нулевой вектор)
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.n_senses = n_senses
        self.pad_idx = pad_idx

        # Амплитуды (сырые логиты) для каждого sense
        self.amplitude_embeddings = nn.Embedding(
            vocab_size, n_senses, padding_idx=pad_idx
        )
        # Базисные эмбеддинги: для каждой леммы и каждого sense вектор размерности embed_dim
        self.sense_embeddings = nn.Embedding(
            vocab_size * n_senses, embed_dim, padding_idx=pad_idx
        )

        nn.init.xavier_uniform_(self.amplitude_embeddings.weight)
        nn.init.xavier_uniform_(self.sense_embeddings.weight)

    def forward(self, token_ids: torch.LongTensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param token_ids: [batch_size, seq_len] — индексы лемм
        :return:
            psi:   [batch_size, seq_len, embed_dim] — вектор после коллапса (суперпозиция)
            probs: [batch_size, seq_len, n_senses]  — вероятности каждого sense
        """
        # получаем логиты амплитуд
        logits = self.amplitude_embeddings(token_ids)  # [B, S, K]
        probs = F.softmax(logits, dim=-1)  # нормировка в вероятности

        # строим индексы для доступа к sense_embeddings
        # для каждого token_id i генерируем [i * n_senses + 0, ..., i * n_senses + (K-1)]
        device = token_ids.device

        sense_offset = torch.arange(self.n_senses, device=device)  # [K]
        base = token_ids.unsqueeze(-1) * self.n_senses  # [B, S, 1]
        sense_ids = base + sense_offset.reshape(1, 1, -1)  # [B, S, K]

        sense_vecs = self.sense_embeddings(sense_ids)  # [B, S, K, D]

        # коллапс: взвешенная суперпозиция
        # psi[b,s,d] = sum_k sqrt(p[b,s,k]) * sense_vecs[b,s,k,d]
        psi = torch.einsum(
            "bsk, bskd -> bsd", torch.sqrt(probs), sense_vecs
        )  # [B, S, D]
        psi = F.normalize(psi, dim=-1)
        return psi, probs
