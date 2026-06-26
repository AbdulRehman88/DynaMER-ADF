from __future__ import annotations

import torch
from torch import nn


class TemporalConvEncoder(nn.Module):
    """Lightweight temporal encoder for [B, T, F] sequences."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        layers = []
        in_ch = input_dim
        for layer_idx in range(num_layers):
            dilation = 2 ** layer_idx
            layers.extend(
                [
                    nn.Conv1d(in_ch, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_ch = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        x = x.transpose(1, 2)  # [B, F, T]
        z = self.net(x)
        return z.transpose(1, 2)  # [B, T, H]


class BiGRUEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1, dropout: float = 0.25):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.gru(x)
        return z
