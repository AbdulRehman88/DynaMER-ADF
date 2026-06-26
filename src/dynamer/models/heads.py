from __future__ import annotations

import torch
from torch import nn


class LinearClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpikeReadoutHead(nn.Module):
    """
    Lightweight spike-inspired readout.

    This is not used to claim biological spiking realism. It is a compact
    neuromorphic-inspired temporal decision head that converts the fused
    latent vector into repeated leaky-integrator spike states before logits.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        spike_steps: int = 8,
        decay: float = 0.85,
        threshold: float = 1.0,
        slope: float = 10.0,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.spike_steps = int(spike_steps)
        self.decay = float(decay)
        self.threshold = float(threshold)
        self.slope = float(slope)

        self.current = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.readout = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current = self.current(x)
        mem = torch.zeros_like(current)
        spike_sum = torch.zeros_like(current)

        for _ in range(self.spike_steps):
            mem = self.decay * mem + current
            spike = torch.sigmoid(self.slope * (mem - self.threshold))
            spike_sum = spike_sum + spike
            mem = mem * (1.0 - spike.detach())

        spike_rate = spike_sum / float(self.spike_steps)
        return self.readout(spike_rate)
