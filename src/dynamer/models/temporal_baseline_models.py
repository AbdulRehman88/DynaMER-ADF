
from __future__ import annotations

from typing import Dict, List, Optional

import torch
from torch import nn


def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    mask = mask.to(dtype=x.dtype, device=x.device)
    if mask.ndim == 2:
        mask = mask.unsqueeze(-1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return (x * mask).sum(dim=1) / denom


class SmallTCNBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        y = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + y)


class BaselineTemporalEncoder(nn.Module):
    """
    Classical temporal encoder used for baseline comparisons.

    Supported variants:
      - temporal_mlp
      - lstm
      - gru
      - bilstm
      - tcn
      - cnn_lstm
    """

    def __init__(self, variant: str, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.variant = str(variant).lower()
        self.hidden_dim = int(hidden_dim)

        self.input_proj = nn.LazyLinear(hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        if self.variant == "temporal_mlp":
            self.backbone = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
            )
            self.out_proj = nn.Identity()

        elif self.variant == "lstm":
            self.backbone = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
            self.out_proj = nn.Identity()

        elif self.variant == "gru":
            self.backbone = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
            self.out_proj = nn.Identity()

        elif self.variant == "bilstm":
            self.backbone = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.out_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        elif self.variant == "tcn":
            self.backbone = SmallTCNBlock(hidden_dim=hidden_dim, dropout=dropout)
            self.out_proj = nn.Identity()

        elif self.variant == "cnn_lstm":
            self.conv = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.backbone = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
            self.out_proj = nn.Identity()

        else:
            raise ValueError(f"Unsupported baseline variant: {variant}")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: [B, T, F]
        x = x.float().contiguous()
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)

        if mask is not None:
            mask = mask.to(device=x.device).contiguous()

        if self.variant == "temporal_mlp":
            pooled = masked_mean(x, mask)
            return self.backbone(pooled)

        if self.variant == "tcn":
            y = self.backbone(x)
            return self.out_proj(masked_mean(y, mask))

        if self.variant == "cnn_lstm":
            y = self.conv(x.transpose(1, 2)).transpose(1, 2).contiguous()
            if y.is_cuda:
                with torch.backends.cudnn.flags(enabled=False):
                    y, _ = self.backbone(y)
            else:
                y, _ = self.backbone(y)
            return self.out_proj(masked_mean(y, mask))

        # RNN variants
        if x.is_cuda:
            with torch.backends.cudnn.flags(enabled=False):
                y, _ = self.backbone(x)
        else:
            y, _ = self.backbone(x)

        y = self.out_proj(y)
        return masked_mean(y, mask)


class BaselineEmotionModel(nn.Module):
    """
    Baseline classifier.

    It intentionally avoids DynaMER's gated-attention fusion and spike-inspired head.
    Each modality is temporally encoded with the selected classical backbone.
    Modality vectors are then concatenated and passed through a simple linear classifier.
    """

    def __init__(
        self,
        modality_keys: List[str],
        num_classes: int,
        variant: str,
        hidden_dim: int = 128,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        self.modality_keys = list(modality_keys)
        self.num_classes = int(num_classes)
        self.variant = str(variant).lower()
        self.hidden_dim = int(hidden_dim)

        self.encoders = nn.ModuleDict({
            key: BaselineTemporalEncoder(
                variant=self.variant,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            for key in self.modality_keys
        })

        fusion_in = hidden_dim * len(self.modality_keys)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        masks = masks or {}
        vectors = []

        for key in self.modality_keys:
            if key not in x:
                continue
            vectors.append(self.encoders[key](x[key], masks.get(key)))

        if not vectors:
            raise RuntimeError("No valid modality inputs were provided to BaselineEmotionModel.")

        fused = torch.cat(vectors, dim=-1)
        logits = self.classifier(fused)

        return {
            "logits": logits,
            "baseline_variant": self.variant,
        }
