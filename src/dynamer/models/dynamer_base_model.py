from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from src.dynamer.models.fusion import GatedAttentionFusion
from src.dynamer.models.heads import LinearClassificationHead, SpikeReadoutHead
from src.dynamer.models.modality_encoders import TemporalModalityEncoder


class DynaMERBaseModel(nn.Module):
    """
    Dynamic Neuromorphic Adaptive Multimodal Emotion Recognition model.

    Components:
      1. modality-specific temporal encoders
      2. gated modality fusion
      3. classification head, optionally spike-inspired

    The model accepts a dictionary:
      x[modality] = tensor [B, T, F]
      masks[modality] = tensor [B, T]
    """

    def __init__(
        self,
        modality_keys: list[str],
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        temporal_backbone: str = "bigru",
        fusion: str = "gated_attention",
        head: str = "spike_readout",
        spike_steps: int = 8,
        spike_decay: float = 0.85,
        spike_threshold: float = 1.0,
        spike_slope: float = 10.0,
    ) -> None:
        super().__init__()

        self.modality_keys = list(modality_keys)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)

        self.encoders = nn.ModuleDict(
            {
                key: TemporalModalityEncoder(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    temporal_backbone=temporal_backbone,
                )
                for key in self.modality_keys
            }
        )

        fusion = str(fusion).lower()
        if fusion != "gated_attention":
            raise ValueError(f"Unsupported fusion type: {fusion}")
        self.fusion = GatedAttentionFusion(hidden_dim=hidden_dim, modality_keys=self.modality_keys)

        head = str(head).lower()
        if head == "spike_readout":
            self.head = SpikeReadoutHead(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                spike_steps=spike_steps,
                decay=spike_decay,
                threshold=spike_threshold,
                slope=spike_slope,
                dropout=dropout,
            )
        elif head == "linear":
            self.head = LinearClassificationHead(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unsupported head type: {head}")

    def forward(self, x: Dict[str, torch.Tensor], masks: Dict[str, torch.Tensor] | None = None) -> Dict[str, torch.Tensor]:
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
DynaMERModel = DynaMERBaseModel
