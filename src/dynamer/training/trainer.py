from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class TrainResult:
    best_epoch: int
    best_val_metric: float
    checkpoint_path: str | None


class Trainer:
    """Minimal trainer scaffold.

    The concrete dataloader and fold-specific preprocessing will be implemented
    after dataset integration is verified.
    """

    def __init__(self, cfg: Any, model: nn.Module):
        self.cfg = cfg
        self.model = model

    def fit(self, train_loader: Any, val_loader: Any) -> TrainResult:
        raise NotImplementedError("Trainer will be implemented after the data loaders are finalized.")

    @staticmethod
    def device_from_cfg(cfg: Any) -> torch.device:
        if cfg.project.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(cfg.project.device)
