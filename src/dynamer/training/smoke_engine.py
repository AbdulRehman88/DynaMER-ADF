from __future__ import annotations

import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {
        "x": {k: v.to(device, non_blocking=True) for k, v in batch["x"].items()},
        "masks": {k: v.to(device, non_blocking=True) for k, v in batch["masks"].items()},
        "y": batch["y"].to(device, non_blocking=True),
        "dataset": batch["dataset"],
        "task": batch["task"],
        "trial_uid": batch["trial_uid"],
        "subject_id": batch["subject_id"],
        "split": batch["split"],
    }


def compute_basic_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1": float("nan"),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_one_epoch_smoke(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    max_batches: int,
    gradient_clip_norm: float,
) -> Dict[str, float]:
    model.train()

    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for batch_idx, batch in enumerate(loader, start=1):
        if batch_idx > max_batches:
            break

        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        out = model(batch["x"], batch["masks"])
        logits = out["logits"]
        loss = criterion(logits, batch["y"])

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss detected.")

        loss.backward()

        if gradient_clip_norm and gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clip_norm))

        optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        preds = torch.argmax(logits.detach(), dim=1)
        y_true.extend(batch["y"].detach().cpu().numpy().astype(int).tolist())
        y_pred.extend(preds.detach().cpu().numpy().astype(int).tolist())

    metrics = compute_basic_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["batches"] = int(len(losses))
    return metrics


@torch.no_grad()
def evaluate_smoke(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    device: torch.device,
    max_batches: int,
) -> Dict[str, float]:
    model.eval()

    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for batch_idx, batch in enumerate(loader, start=1):
        if batch_idx > max_batches:
            break

        batch = move_batch_to_device(batch, device)

        out = model(batch["x"], batch["masks"])
        logits = out["logits"]
        loss = criterion(logits, batch["y"])

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite evaluation loss detected.")

        losses.append(float(loss.detach().cpu().item()))
        preds = torch.argmax(logits.detach(), dim=1)
        y_true.extend(batch["y"].detach().cpu().numpy().astype(int).tolist())
        y_pred.extend(preds.detach().cpu().numpy().astype(int).tolist())

    metrics = compute_basic_metrics(y_true, y_pred)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["batches"] = int(len(losses))
    return metrics
