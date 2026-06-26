from __future__ import annotations

import torch
from torch import nn


class MaskedMeanPool(nn.Module):
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            return x.mean(dim=1)

        mask = mask.float()
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom


class TemporalModalityEncoder(nn.Module):
    """
    Converts one modality sequence [B, T, F] into a fixed vector [B, H].

    LazyLinear allows the input feature dimension to remain dataset-specific
    and fully dynamic. No feature dimension is hard-coded.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        dropout: float = 0.2,
        temporal_backbone: str = "bigru",
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.temporal_backbone = str(temporal_backbone).lower()

        self.input_proj = nn.LazyLinear(self.hidden_dim)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.pool = MaskedMeanPool()

        if self.temporal_backbone == "bigru":
            if self.hidden_dim % 2 != 0:
                raise ValueError("hidden_dim must be even for bidirectional GRU.")
            self.backbone = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
        elif self.temporal_backbone == "tcn":
            self.backbone = nn.Sequential(
                nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Conv1d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
        elif self.temporal_backbone == "none":
            self.backbone = nn.Identity()
        else:
            raise ValueError(f"Unsupported temporal_backbone: {temporal_backbone}")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x.float()
        x = self.input_proj(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.dropout(x)

        if self.temporal_backbone == "bigru":
            x = x.contiguous()
            if mask is not None:
                mask = mask.contiguous()
                lengths = mask.sum(dim=1).clamp_min(1).long().cpu()

                packed = nn.utils.rnn.pack_padded_sequence(
                    x,
                    lengths=lengths,
                    batch_first=True,
                    enforce_sorted=False,
                )

                # Safer CUDA path: some very long/variable DREAMER sequences can
                # trigger cuDNN GRU support errors. Disabling cuDNN only for this
                # recurrent call preserves correctness and keeps the architecture
                # smoke-test deterministic.
                if x.is_cuda:
                    with torch.backends.cudnn.flags(enabled=False):
                        packed_out, _ = self.backbone(packed)
                else:
                    packed_out, _ = self.backbone(packed)

                x, _ = nn.utils.rnn.pad_packed_sequence(
                    packed_out,
                    batch_first=True,
                    total_length=x.shape[1],
                )
            else:
                if x.is_cuda:
                    with torch.backends.cudnn.flags(enabled=False):
                        x, _ = self.backbone(x)
                else:
                    x, _ = self.backbone(x)

        elif self.temporal_backbone == "tcn":
            x = x.transpose(1, 2)
            x = self.backbone(x)
            x = x.transpose(1, 2)

        return self.pool(x, mask)
