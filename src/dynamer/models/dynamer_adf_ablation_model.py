
from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn

from src.dynamer.models.modality_encoders import TemporalModalityEncoder
from src.dynamer.models.dynamer_bitcn_model import (
    BiLSTMTCNTemporalEncoder,
    ModalityAdaptiveGatedFusion,
    HybridLinearSpikeHead,
)


class AblationDualPathTemporalEncoder(nn.Module):
    """
    Dual-path temporal encoder with controlled path ablations:
      - learned_dual: original DynaMER-v3 learned path gate
      - v1_only: original DynaMER-v1 temporal path only
      - v2_only: BiLSTM-TCN path only
      - fixed_mean: non-learned average of v1 and v2 paths
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.20,
        temporal_backbone_v1: str = "bigru",
        tcn_layers: int = 2,
        path_mode: str = "learned_dual",
    ) -> None:
        super().__init__()

        self.path_mode = str(path_mode)

        self.v1_encoder = TemporalModalityEncoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
            temporal_backbone=temporal_backbone_v1,
        )

        self.v2_encoder = BiLSTMTCNTemporalEncoder(
            hidden_dim=hidden_dim,
            dropout=dropout,
            tcn_layers=max(1, int(tcn_layers)),
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

        if self.path_mode == "v1_only":
            fused = v1
        elif self.path_mode == "v2_only":
            fused = v2
        elif self.path_mode == "fixed_mean":
            fused = 0.5 * (v1 + v2)
        else:
            gate = self.mix_gate(torch.cat([v1, v2], dim=-1))
            fused = gate * v2 + (1.0 - gate) * v1

        return self.out_norm(fused)


class MeanModalityFusion(nn.Module):
    """
    Simple non-gated modality fusion.
    Used to test whether gated modality-adaptive fusion is really useful.
    """

    def __init__(self, hidden_dim: int, modality_keys: List[str]) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.modality_keys = list(modality_keys)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, modality_vectors: Dict[str, torch.Tensor]):
        active_keys = [k for k in self.modality_keys if k in modality_vectors]
        if len(active_keys) == 0:
            raise ValueError("No active modality vectors were provided to MeanModalityFusion.")

        stacked = torch.stack([modality_vectors[k] for k in active_keys], dim=1)
        fused = stacked.mean(dim=1)
        fused = self.out_norm(fused)

        batch = fused.shape[0]
        weights = torch.full(
            (batch, len(active_keys)),
            1.0 / float(len(active_keys)),
            device=fused.device,
            dtype=fused.dtype,
        )

        return fused, weights, active_keys


class DynaMERADFAblationModel(nn.Module):
    """
    DynaMER-v3 component-ablation model.

    Supports:
      - modality ablations through modality_keys
      - temporal path ablations through path_mode
      - fusion ablation through fusion_mode
      - spike/head sensitivity through spike_mix
      - depth/regularization sensitivity through tcn_layers/modality_dropout/dropout
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
        path_mode: str = "learned_dual",
        fusion_mode: str = "gated",
    ) -> None:
        super().__init__()

        self.modality_keys = list(modality_keys)
        self.num_classes = int(num_classes)
        self.path_mode = str(path_mode)
        self.fusion_mode = str(fusion_mode)

        self.encoders = nn.ModuleDict({
            key: AblationDualPathTemporalEncoder(
                hidden_dim=hidden_dim,
                dropout=dropout,
                temporal_backbone_v1=temporal_backbone_v1,
                tcn_layers=tcn_layers,
                path_mode=path_mode,
            )
            for key in self.modality_keys
        })

        if self.fusion_mode == "mean":
            self.fusion = MeanModalityFusion(
                hidden_dim=hidden_dim,
                modality_keys=self.modality_keys,
            )
        else:
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



# Backward-compatible alias
DynaMERv3AblationModel = DynaMERADFAblationModel
