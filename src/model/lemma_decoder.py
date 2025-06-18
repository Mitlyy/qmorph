from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LemmaDecoder(nn.Module):
    """
    Head для предсказания следующих лемм:
    - линейный слой embed_dim → vocab_size
    - методы автодекодинга (greedy и beam search)
    """

    def __init__(
        self, embed_dim: int, vocab_size: int, pad_idx: int = 0, dropout: float = 0.1
    ):
        """
        :param embed_dim: размерность входного вектора (последний скрытый слой трансформера)
        :param vocab_size: размер словаря лемм
        :param pad_idx: индекс паддинга (не генерировать)
        :param dropout: дропаут перед классификацией
        """
        super().__init__()
        self.classifier = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx

    def forward(self, contextual_embeddings: torch.Tensor) -> torch.Tensor:
        """
        :param contextual_embeddings: [batch, seq_len, embed_dim]
        :return: logits по леммам [batch, seq_len, vocab_size]
        """
        x = self.dropout(contextual_embeddings)
        return self.classifier(x)

    def greedy_decode(
        self,
        embedding: nn.Module,
        transformer: nn.Module,
        start_ids: torch.LongTensor,
        max_length: int,
        device: torch.device,
    ) -> List[int]:
        """
        Автодекодинг жадным методом (greedy).
        :param embedding: QuantumEmbedding
        :param transformer: TransformerCore (в autoregressive режиме)
        :param start_ids: [1, init_len] tensor с начальными леммами
        :param max_length: максимальная общая длина секвенции
        :param device: устройство
        :return: список сгенерированных ID лемм (длина = max_length)
        """
        generated = start_ids.tolist()[0]
        input_ids = start_ids
        for _ in range(max_length - input_ids.size(1)):
            # получаем psi и пропускаем через трансформер
            psi, _ = embedding(input_ids)
            contextual = transformer(psi, padding_mask=input_ids.eq(self.pad_idx))
            last_hidden = contextual[:, -1, :]  # [1, embed_dim]
            logits = self.classifier(last_hidden)  # [1, vocab_size]
            next_id = torch.argmax(F.log_softmax(logits, dim=-1), dim=-1).item()
            generated.append(next_id)
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=device)], dim=1
            )
        return generated

    def beam_search_decode(
        self,
        embedding: nn.Module,
        transformer: nn.Module,
        start_ids: torch.LongTensor,
        max_length: int,
        beam_size: int,
        device: torch.device,
    ) -> List[int]:
        """
        Автодекодинг с beam search.
        :param embedding: QuantumEmbedding
        :param transformer: TransformerCore (в autoregressive режиме)
        :param start_ids: [1, init_len]
        :param max_length: максимальная длина
        :param beam_size: число лучей
        :param device: устройство
        :return: лучшая сгенерированная последовательность ID лемм
        """
        beams: List[Tuple[List[int], float]] = [(start_ids.tolist()[0], 0.0)]

        for _ in range(max_length - start_ids.size(1)):
            all_candidates: List[Tuple[List[int], float]] = []
            for seq, score in beams:
                seq_tensor = torch.tensor([seq], device=device)
                psi, _ = embedding(seq_tensor)
                contextual = transformer(psi, padding_mask=seq_tensor.eq(self.pad_idx))
                logits = self.classifier(contextual[:, -1, :])
                log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
                topk_probs, topk_indices = torch.topk(log_probs, beam_size)
                for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
                    candidate_seq = seq + [idx]
                    candidate_score = score + prob
                    all_candidates.append((candidate_seq, candidate_score))

            all_candidates.sort(key=lambda x: x[1], reverse=True)
            beams = all_candidates[:beam_size]

        best_seq, _ = max(beams, key=lambda x: x[1])
        return best_seq
