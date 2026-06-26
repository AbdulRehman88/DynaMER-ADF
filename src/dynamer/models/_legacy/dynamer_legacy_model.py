from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.dynamer.models.encoders import TemporalConvEncoder, BiGRUEncoder
from src.dynamer.models.fusion import GatedModalityFusion
from src.dynamer.models.heads import NonSpikingHead, SpikeReadoutHead


class DynaMERModel(nn.Module):
    def __init__(
        self,
        modality_input_dims: dict[str, int],
        num_classes: int,
        cfg: Any,
    ):
        super().__init__()
        self.modalities = list(modality_input_dims.keys())
        hidden_dim = int(cfg.model.hidden_dim)
        dropout = float(cfg.model.dropout)
        num_layers = int(cfg.model.num_layers)

        encoder_cls = TemporalConvEncoder if cfg.model.backbone == "tcn" else BiGRUEncoder
        self.encoders = nn.ModuleDict(
            {
                name: encoder_cls(input_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
                for name, in_dim in modality_input_dims.items()
            }
        )

        self.fusion = GatedModalityFusion(self.modalities, hidden_dim=hidden_dim)

        if cfg.model.spike.enabled:
            self.head = SpikeReadoutHead(
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                threshold=float(cfg.model.spike.threshold),
                steps=int(cfg.model.spike.steps),
                dropout=dropout,
            )
        else:
            self.head = NonSpikingHead(hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)

    def forward(
        self,
        batch_modalities: dict[str, torch.Tensor],
        modality_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        encoded = {}
        for name, x in batch_modalities.items():
            if name in self.encoders:
                encoded[name] = self.encoders[name](x)

        fused = self.fusion(encoded, modality_mask=modality_mask)
        logits = self.head(fused)
        return logits
