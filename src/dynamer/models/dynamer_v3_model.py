
from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from src.dynamer.models.modality_encoders import TemporalModalityEncoder
from src.dynamer.models.dynamer_v2_model import (
    BiLSTMTCNTemporalEncoder,
    ModalityAdaptiveGatedFusion,
    HybridLinearSpikeHead,
)


class DualPathTemporalEncoder(nn.Module):
    """
    DynaMER-v3 temporal encoder:
    combines the original DynaMER-v1 temporal path with the stronger BiLSTM-TCN path.
    A learnable residual gate decides how much to use from each path.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.20,
        temporal_backbone_v1: str = "bigru",
        tcn_layers: int = 2,
    ) -> None:
        super().__init__()

        self.v1_encoder = TemporalModalityEncoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
            temporal_backbone=temporal_backbone_v1,
        )

        self.v2_encoder = BiLSTMTCNTemporalEncoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
            tcn_layers=tcn_layers,
        )

        self.mix_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )

        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        v1 = self.v1_encoder(x, mask)
        v2 = self.v2_encoder(x, mask)

        gate = self.mix_gate(torch.cat([v1, v2], dim=-1))
        fused = gate * v2 + (1.0 - gate) * v1

        return self.out_norm(fused)


class DynaMERv3Model(nn.Module):
    """
    DynaMER-v3:
      1. dual-path temporal encoder: DynaMER-v1 path + BiLSTM-TCN path
      2. modality-adaptive gated fusion
      3. linear-dominant hybrid spike head
    """

    def __init__(
        self,
        modality_keys: List[str],
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.20,
        temporal_backbone_v1: str = "bigru",
        tcn_layers: int = 2,
        modality_dropout: float = 0.00,
        spike_steps: int = 6,
        spike_decay: float = 0.85,
        spike_threshold: float = 1.0,
        spike_slope: float = 5.0,
        spike_mix: float = 0.10,
    ) -> None:
        super().__init__()

        self.modality_keys = list(modality_keys)
        self.num_classes = int(num_classes)

        self.encoders = nn.ModuleDict({
            key: DualPathTemporalEncoder(
                hidden_dim=hidden_dim,
                dropout=dropout,
                temporal_backbone_v1=temporal_backbone_v1,
                tcn_layers=tcn_layers,
            )
            for key in self.modality_keys
        })

        self.fusion = ModalityAdaptiveGatedFusion(
            hidden_dim=hidden_dim,
            modality_keys=self.modality_keys,
            dropout=dropout,
            modality_dropout=modality_dropout,
        )

        self.head = HybridLinearSpikeHead(
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            spike_steps=spike_steps,
            spike_decay=spike_decay,
            spike_threshold=spike_threshold,
            spike_slope=spike_slope,
            spike_mix=spike_mix,
        )

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        masks = masks or {}
        modality_vectors = {}

        for key in self.modality_keys:
            if key not in x:
                continue
            modality_vectors[key] = self.encoders[key](x[key], masks.get(key))

        fused, gate_weights, active_keys = self.fusion(modality_vectors)
        logits = self.head(fused)

        return {
            "logits": logits,
            "fused": fused,
            "gate_weights": gate_weights,
            "active_modalities": active_keys,
        }
