from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def make_class_weights(labels: np.ndarray, num_classes: int, device: torch.device) -> Optional[torch.Tensor]:
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)

    if np.any(counts == 0):
        return None

    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def safe_metric_value(value: float) -> float:
    try:
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return float("nan")
        return value
    except Exception:
        return float("nan")


def compute_metrics(y_true: List[int], y_pred: List[int], y_prob: List[List[float]], num_classes: int) -> Dict[str, float]:
    if len(y_true) == 0:
        return {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_precision": float("nan"),
            "macro_recall": float("nan"),
            "macro_f1": float("nan"),
            "roc_auc": float("nan"),
            "pr_auc": float("nan"),
        }

    y_true_np = np.asarray(y_true, dtype=int)
    y_pred_np = np.asarray(y_pred, dtype=int)
    y_prob_np = np.asarray(y_prob, dtype=np.float32)

    metrics = {
        "accuracy": safe_metric_value(accuracy_score(y_true_np, y_pred_np)),
        "balanced_accuracy": safe_metric_value(balanced_accuracy_score(y_true_np, y_pred_np)),
        "macro_precision": safe_metric_value(precision_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "macro_recall": safe_metric_value(recall_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "macro_f1": safe_metric_value(f1_score(y_true_np, y_pred_np, average="macro", zero_division=0)),
        "roc_auc": float("nan"),
        "pr_auc": float("nan"),
    }

    present_classes = sorted(np.unique(y_true_np).tolist())

    try:
        if len(present_classes) >= 2:
            if num_classes == 2:
                metrics["roc_auc"] = safe_metric_value(roc_auc_score(y_true_np, y_prob_np[:, 1]))
                metrics["pr_auc"] = safe_metric_value(average_precision_score(y_true_np, y_prob_np[:, 1]))
            else:
                y_onehot = np.eye(num_classes, dtype=np.float32)[y_true_np]
                metrics["roc_auc"] = safe_metric_value(
                    roc_auc_score(y_onehot, y_prob_np, average="macro", multi_class="ovr")
                )
                metrics["pr_auc"] = safe_metric_value(
                    average_precision_score(y_onehot, y_prob_np, average="macro")
                )
    except Exception:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


def run_train_epoch(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    scaler,
    use_amp: bool,
    gradient_clip_norm: float,
    num_classes: int,
) -> Dict[str, float]:
    model.train()

    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=bool(use_amp and device.type == "cuda")):
            out = model(batch["x"], batch["masks"])
            logits = out["logits"]
            loss = criterion(logits, batch["y"])

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss detected.")

        if scaler is not None and bool(use_amp and device.type == "cuda"):
            scaler.scale(loss).backward()
            if gradient_clip_norm and gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clip_norm))
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_norm and gradient_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(gradient_clip_norm))
            optimizer.step()

        probs = torch.softmax(logits.detach(), dim=1)
        preds = torch.argmax(probs, dim=1)

        losses.append(float(loss.detach().cpu().item()))
        y_true.extend(batch["y"].detach().cpu().numpy().astype(int).tolist())
        y_pred.extend(preds.detach().cpu().numpy().astype(int).tolist())
        y_prob.extend(probs.detach().cpu().numpy().astype(float).tolist())

    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes=num_classes)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["batches"] = int(len(losses))
    return metrics


@torch.no_grad()
def run_eval_epoch(
    model: torch.nn.Module,
    loader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    collect_predictions: bool = False,
) -> tuple[Dict[str, float], List[Dict]]:
    model.eval()

    losses: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    y_prob: List[List[float]] = []
    pred_rows: List[Dict] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        out = model(batch["x"], batch["masks"])
        logits = out["logits"]
        loss = criterion(logits, batch["y"])

        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite evaluation loss detected.")

        probs = torch.softmax(logits.detach(), dim=1)
        preds = torch.argmax(probs, dim=1)

        losses.append(float(loss.detach().cpu().item()))
        y_true_batch = batch["y"].detach().cpu().numpy().astype(int).tolist()
        y_pred_batch = preds.detach().cpu().numpy().astype(int).tolist()
        y_prob_batch = probs.detach().cpu().numpy().astype(float).tolist()

        y_true.extend(y_true_batch)
        y_pred.extend(y_pred_batch)
        y_prob.extend(y_prob_batch)

        if collect_predictions:
            for i in range(len(y_true_batch)):
                row = {
                    "trial_uid": batch["trial_uid"][i],
                    "subject_id": batch["subject_id"][i],
                    "split": batch["split"][i],
                    "y_true": int(y_true_batch[i]),
                    "y_pred": int(y_pred_batch[i]),
                }
                for c in range(num_classes):
                    row[f"prob_class_{c}"] = float(y_prob_batch[i][c])
                pred_rows.append(row)

    metrics = compute_metrics(y_true, y_pred, y_prob, num_classes=num_classes)
    metrics["loss"] = float(np.mean(losses)) if losses else float("nan")
    metrics["batches"] = int(len(losses))
    return metrics, pred_rows


class EarlyStopper:
    def __init__(self, mode: str = "max", patience: int = 10, min_delta: float = 0.0) -> None:
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best_score: Optional[float] = None
        self.bad_epochs = 0

    def improved(self, value: float) -> bool:
        if value is None or math.isnan(float(value)):
            return False

        value = float(value)

        if self.best_score is None:
            self.best_score = value
            self.bad_epochs = 0
            return True

        if self.mode == "max":
            is_better = value > self.best_score + self.min_delta
        elif self.mode == "min":
            is_better = value < self.best_score - self.min_delta
        else:
            raise ValueError(f"Unsupported early-stop mode: {self.mode}")

        if is_better:
            self.best_score = value
            self.bad_epochs = 0
            return True

        self.bad_epochs += 1
        return False

    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience
