import json
from typing import Dict, List

import torch
import torch.nn as nn


class MorphDecoder(nn.Module):
    """
    Генератор морфологической формы: предсказывает морфологический тег по
    контекстуальному вектору, а затем восстанавливает поверхность слова
    по заранее построенному мэппингу (form_mapping.json).
    """

    def __init__(
        self,
        embed_dim: int,
        morph_vocab_size: int,
        form_mapping_file: str,
        dropout: float = 0.1,
        pad_idx: int = 0,
    ):
        """
        :param embed_dim: размерность входного вектора (трансформер-выход)
        :param morph_vocab_size: число морф.тегов
        :param form_mapping_file: путь к vocab/form_mapping.json
        :param dropout: вероятность дропаут перед классификацией
        :param pad_idx: индекс паддинга (чтобы всегда выдавать пустую строку)
        """
        super().__init__()
        self.classifier = nn.Linear(embed_dim, morph_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.pad_idx = pad_idx

        # Загрузка сопоставления (lemma_id -> {morph_id -> surface_form})
        with open(form_mapping_file, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.form_mapping: Dict[int, Dict[int, str]] = {
            int(lemma_id): {int(m_id): form for m_id, form in forms.items()}
            for lemma_id, forms in raw.items()
        }

    def forward(self, contextual_embeddings: torch.Tensor) -> torch.Tensor:
        """
        :param contextual_embeddings: [batch, seq_len, embed_dim] — выход TransformerCore
        :return: logits по морфологическим тегам [batch, seq_len, morph_vocab_size]
        """
        x = self.dropout(contextual_embeddings)
        logits = self.classifier(x)
        return logits

    def decode_forms(
        self, lemma_ids: torch.LongTensor, morph_logits: torch.Tensor
    ) -> List[List[str]]:
        """
        Декодирует батч в поверхности слов:
        1) находит argmax по тегам
        2) маппит (lemma_id, morph_id) -> surface_form
        :param lemma_ids:    [batch, seq_len] тензор лемм-ID
        :param morph_logits: [batch, seq_len, M] логиты тегов
        :return: список списков строк формы слов
        """
        batch_size, seq_len, _ = morph_logits.size()
        pred_tags = torch.argmax(morph_logits, dim=-1)  # [B, S]

        forms: List[List[str]] = []
        for b in range(batch_size):
            sent_forms: List[str] = []
            for s in range(seq_len):
                lid = int(lemma_ids[b, s].item())
                mid = int(pred_tags[b, s].item())

                if lid == self.pad_idx:
                    sent_forms.append("")
                else:
                    form = self.form_mapping.get(lid, {}).get(mid, None)
                    if form is None:
                        form = f"<lemma_{lid}>"
                    sent_forms.append(form)
            forms.append(sent_forms)
        return forms
