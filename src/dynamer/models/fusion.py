from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


class GatedAttentionFusion(nn.Module):
    """
    Learns modality reliability weights and fuses available modality vectors.

    Inputs:
      modality_vectors: dict of {modality_name: [B, H]}

    Output:
      fused vector [B, H]
      gate weights [B, M]
    """

    def __init__(self, hidden_dim: int, modality_keys: list[str]) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.modality_keys = list(modality_keys)

        self.gates = nn.ModuleDict(
            {
                key: nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim // 2, 1),
                )
                for key in self.modality_keys
            }
        )

    def forward(self, modality_vectors: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, list[str]]:
        active_keys = [k for k in self.modality_keys if k in modality_vectors]
        if len(active_keys) == 0:
            raise RuntimeError("No modality vectors were provided to fusion module.")

        vectors = torch.stack([modality_vectors[k] for k in active_keys], dim=1)
        gate_logits = torch.cat([self.gates[k](modality_vectors[k]) for k in active_keys], dim=1)
        gate_weights = torch.softmax(gate_logits, dim=1)
        fused = (vectors * gate_weights.unsqueeze(-1)).sum(dim=1)

        return fused, gate_weights, active_keys
