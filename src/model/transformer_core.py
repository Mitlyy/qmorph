from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Синусно-косинусные позиционные кодировки (как в оригинальном Transformer).
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [batch_size, seq_len, d_model]
        :return: x + positional_encoding
        """
        return x + self.pe[:, : x.size(1), :]


class TransformerCore(nn.Module):
    """
    Основная часть трансформера: self-attention + FFN.
    Поддерживает как «полнослойный» (для кодирования всех позиций), так и
    автодекодерный режим (сделанный через causal mask).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int = 2048,
        num_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        is_autoregressive: bool = False,
    ):
        """
        :param embed_dim: размерность входных эмбеддингов
        :param num_heads: число голов в Multi-Head Attention
        :param ff_dim: размер внутреннего слоя FFN
        :param num_layers: число энкодер-слоёв
        :param dropout: дропаут
        :param max_seq_len: макс. длина для позиции
        :param is_autoregressive: если True — применяет causal mask для автодекодинга
        """
        super().__init__()
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len)
        self.is_autoregressive = is_autoregressive

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _generate_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Генерирует верхнетреугольную матрицу маски (causal mask),
        закрывая доступ к будущим позициям.
        """
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )

    def forward(
        self, x: torch.Tensor, padding_mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """
        :param x: [batch_size, seq_len, embed_dim]
        :param padding_mask: [batch_size, seq_len] — True на позициях PAD
        :return: [batch_size, seq_len, embed_dim]
        """
        x = self.pos_encoder(x)

        mask = None
        if self.is_autoregressive:
            seq_len = x.size(1)
            mask = self._generate_causal_mask(seq_len, x.device)

        out = self.transformer(x, mask=mask, src_key_padding_mask=padding_mask)
        return out
