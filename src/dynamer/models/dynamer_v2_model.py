
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

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


def masked_attention_pool(x: torch.Tensor, attn: nn.Module, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    # x: [B, T, H]
    scores = attn(x).squeeze(-1)  # [B, T]

    if mask is not None:
        mask = mask.to(device=x.device)
        scores = scores.masked_fill(mask <= 0, -1e4)

    weights = torch.softmax(scores, dim=1).unsqueeze(-1)
    pooled = (x * weights).sum(dim=1)

    # Safety fallback if a row was fully masked.
    if not torch.isfinite(pooled).all():
        pooled = masked_mean(x, mask)

    return pooled


class ResidualTCNBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, dilation: int = 1) -> None:
        super().__init__()
        padding = dilation
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(x + y)


class BiLSTMTCNTemporalEncoder(nn.Module):
    """
    Stronger temporal encoder for DynaMER-v2.

    It intentionally absorbs the best SEED-IV baseline lesson:
    BiLSTM/TCN temporal modeling is stronger than the original DynaMER-v1 encoder
    under subject-LOSO evaluation.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.25,
        tcn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)

        self.input_proj = nn.LazyLinear(hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Bidirectional LSTM outputs hidden_dim total.
        self.bilstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=max(1, hidden_dim // 2),
            batch_first=True,
            bidirectional=True,
        )

        self.bilstm_proj = nn.Linear((hidden_dim // 2) * 2, hidden_dim)
        self.bilstm_norm = nn.LayerNorm(hidden_dim)

        dilations = [1, 2, 4, 1][:max(1, int(tcn_layers))]
        self.tcn = nn.ModuleList([
            ResidualTCNBlock(hidden_dim=hidden_dim, dropout=dropout, dilation=d)
            for d in dilations
        ])

        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x.float().contiguous()

        if mask is not None:
            mask = mask.to(device=x.device).contiguous()

        x = self.input_proj(x)
        x = self.input_norm(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)

        if x.is_cuda:
            with torch.backends.cudnn.flags(enabled=False):
                y, _ = self.bilstm(x)
        else:
            y, _ = self.bilstm(x)

        y = self.bilstm_proj(y)
        y = self.bilstm_norm(y)
        y = torch.nn.functional.gelu(y)
        y = self.dropout(y)

        for block in self.tcn:
            y = block(y)

        pooled = masked_attention_pool(y, self.attn, mask)
        pooled = self.out_norm(pooled)
        return pooled


class ModalityAdaptiveGatedFusion(nn.Module):
    """
    Modality-adaptive fusion with optional modality dropout.

    This keeps the central DynaMER idea:
    the model should learn how much to trust each available modality.
    """

    def __init__(
        self,
        hidden_dim: int,
        modality_keys: List[str],
        dropout: float = 0.25,
        modality_dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.modality_keys = list(modality_keys)
        self.modality_dropout = float(modality_dropout)

        self.modality_embeddings = nn.ParameterDict({
            key: nn.Parameter(torch.zeros(hidden_dim))
            for key in self.modality_keys
        })

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def _make_keep_mask(self, batch_size: int, n_modalities: int, device: torch.device) -> torch.Tensor:
        if (not self.training) or self.modality_dropout <= 0.0 or n_modalities <= 1:
            return torch.ones(batch_size, n_modalities, dtype=torch.bool, device=device)

        keep = torch.rand(batch_size, n_modalities, device=device) > self.modality_dropout

        # Ensure at least one modality remains per sample.
        empty = ~keep.any(dim=1)
        if empty.any():
            fallback = torch.randint(0, n_modalities, size=(int(empty.sum()),), device=device)
            keep[empty] = False
            keep[empty, fallback] = True

        return keep

    def forward(self, modality_vectors: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        active_keys = [k for k in self.modality_keys if k in modality_vectors]
        if not active_keys:
            raise RuntimeError("No modality vectors provided to fusion.")

        vectors = []
        for key in active_keys:
            v = modality_vectors[key]
            v = v + self.modality_embeddings[key].to(device=v.device).unsqueeze(0)
            vectors.append(v)

        stack = torch.stack(vectors, dim=1)  # [B, M, H]
        b, m, _ = stack.shape

        gate_logits = self.gate(stack).squeeze(-1)  # [B, M]
        keep_mask = self._make_keep_mask(b, m, stack.device)
        gate_logits = gate_logits.masked_fill(~keep_mask, -1e4)

        weights = torch.softmax(gate_logits, dim=1)
        fused = (stack * weights.unsqueeze(-1)).sum(dim=1)
        fused = self.out(fused)

        return fused, weights, active_keys


class HybridLinearSpikeHead(nn.Module):
    """
    Hybrid head:
    - linear/MLP branch provides stable discriminative classification
    - small spike-inspired branch preserves lightweight neuromorphic identity
    - spike_mix keeps spike branch helpful but not dominant
    """

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.25,
        spike_steps: int = 6,
        spike_decay: float = 0.85,
        spike_threshold: float = 1.0,
        spike_slope: float = 5.0,
        spike_mix: float = 0.15,
    ) -> None:
        super().__init__()
        self.spike_steps = int(spike_steps)
        self.spike_decay = float(spike_decay)
        self.spike_threshold = float(spike_threshold)
        self.spike_slope = float(spike_slope)
        self.spike_mix = float(spike_mix)

        self.linear_branch = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.spike_current = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.spike_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_logits = self.linear_branch(x)

        current = self.spike_current(x)
        membrane = torch.zeros_like(current)
        spike_sum = torch.zeros_like(current)

        for _ in range(max(1, self.spike_steps)):
            membrane = self.spike_decay * membrane + current
            spike = torch.sigmoid((membrane - self.spike_threshold) * self.spike_slope)
            spike_sum = spike_sum + spike
            membrane = membrane * (1.0 - spike.detach())

        spike_logits = self.spike_scale * (spike_sum / max(1, self.spike_steps))

        return (1.0 - self.spike_mix) * linear_logits + self.spike_mix * spike_logits


class DynaMERv2Model(nn.Module):
    """
    DynaMER-v2:
      1. BiLSTM-TCN modality-specific temporal encoders
      2. modality-adaptive gated fusion
      3. hybrid linear/spike-inspired classification head

    This is a principled upgrade over DynaMER-v1 based on SEED-IV baseline evidence.
    """

    def __init__(
        self,
        modality_keys: List[str],
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.25,
        tcn_layers: int = 2,
        modality_dropout: float = 0.10,
        spike_steps: int = 6,
        spike_decay: float = 0.85,
        spike_threshold: float = 1.0,
        spike_slope: float = 5.0,
        spike_mix: float = 0.15,
    ) -> None:
        super().__init__()
        self.modality_keys = list(modality_keys)
        self.num_classes = int(num_classes)

        self.encoders = nn.ModuleDict({
            key: BiLSTMTCNTemporalEncoder(
                hidden_dim=hidden_dim,
                dropout=dropout,
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
