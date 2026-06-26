
from __future__ import annotations

import inspect
from typing import Any, Dict, List

import torch
from torch import nn

from src.dynamer.models.dynamer_model import DynaMERModel
from src.dynamer.models.dynamer_v3_model import DynaMERv3Model


def _make_supported_instance(cls, kwargs: Dict[str, Any]):
    sig = inspect.signature(cls.__init__)
    params = sig.parameters

    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return cls(**kwargs)

    allowed = {k: v for k, v in kwargs.items() if k in params}
    return cls(**allowed)


def _extract_logits(out):
    if isinstance(out, dict):
        return out["logits"]
    return out


class DynaMERv5Model(nn.Module):
    """
    DynaMER-v5: v1-anchored residual DynaMER.

    Anchor branch:
        exact DynaMER-v1 model path.

    Residual branch:
        DynaMER-v3 dual-path temporal model.

    Final prediction:
        logits = anchor_logits + alpha * (residual_logits - anchor_logits)

    alpha is learnable but capped, so the residual branch can improve the anchor
    without freely destroying the v1 cross-session behavior.
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
        residual_init_scale: float = 0.10,
        residual_max_scale: float = 0.35,
    ) -> None:
        super().__init__()

        self.modality_keys = list(modality_keys)
        self.num_classes = int(num_classes)
        self.residual_max_scale = float(residual_max_scale)

        anchor_kwargs = {
            "modality_keys": self.modality_keys,
            "num_classes": self.num_classes,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "temporal_backbone": temporal_backbone_v1,
            "temporal_backbone_v1": temporal_backbone_v1,
            "spike_steps": spike_steps,
            "spike_decay": spike_decay,
            "spike_threshold": spike_threshold,
            "spike_slope": spike_slope,
        }

        self.anchor = _make_supported_instance(DynaMERModel, anchor_kwargs)

        self.residual = DynaMERv3Model(
            modality_keys=self.modality_keys,
            num_classes=self.num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            temporal_backbone_v1=temporal_backbone_v1,
            tcn_layers=tcn_layers,
            modality_dropout=modality_dropout,
            spike_steps=spike_steps,
            spike_decay=spike_decay,
            spike_threshold=spike_threshold,
            spike_slope=spike_slope,
            spike_mix=spike_mix,
        )

        init_ratio = max(1e-4, min(0.999, float(residual_init_scale) / max(float(residual_max_scale), 1e-6)))
        init_logit = torch.logit(torch.tensor(init_ratio, dtype=torch.float32))
        self.raw_alpha = nn.Parameter(torch.full((self.num_classes,), float(init_logit)))

    def forward(
        self,
        x: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor] | None = None,
    ) -> Dict[str, torch.Tensor]:
        masks = masks or {}

        anchor_out = self.anchor(x, masks)
        residual_out = self.residual(x, masks)

        anchor_logits = _extract_logits(anchor_out)
        residual_logits = _extract_logits(residual_out)

        alpha = self.residual_max_scale * torch.sigmoid(self.raw_alpha)
        logits = anchor_logits + alpha.view(1, -1) * (residual_logits - anchor_logits)

        out = {
            "logits": logits,
            "anchor_logits": anchor_logits,
            "residual_logits": residual_logits,
            "residual_alpha": alpha.detach(),
        }

        if isinstance(anchor_out, dict):
            for k, v in anchor_out.items():
                if k != "logits":
                    out[f"anchor_{k}"] = v

        if isinstance(residual_out, dict):
            for k, v in residual_out.items():
                if k != "logits":
                    out[f"residual_{k}"] = v

        return out
