from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import KFold

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.dynamer.data.temporal_data_modules import DynaMERTemporalSplitDataModule
from src.dynamer.models.dynamer_adf_ablation_model import DynaMERADFAblationModel
from src.dynamer.training.full_engine import (
    EarlyStopper,
    count_parameters,
    make_class_weights,
    run_eval_epoch,
    run_train_epoch,
    set_global_seed,
)


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Logger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def info(self, msg: str) -> None:
        line = f"[{now()}] [INFO] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def warn(self, msg: str) -> None:
        line = f"[{now()}] [WARN] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def error(self, msg: str) -> None:
        line = f"[{now()}] [ERROR] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def as_path(x: str) -> Path:
    return Path(str(x).replace("\\", "/")).expanduser().resolve()


def sanitize(x: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(x))


def add_check(checks: List[Dict[str, Any]], name: str, observed: Any, expected: Any, passed: Optional[bool] = None) -> None:
    if passed is None:
        passed = observed == expected
    checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed)})


def parse_int_list(text: Optional[str]) -> Optional[List[int]]:
    if text is None or str(text).strip() == "":
        return None
    out: List[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or None


def stable_run_seed(base_seed: int, *parts: Any) -> int:
    key = "::".join([str(int(base_seed))] + [str(p) for p in parts])
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2_147_483_647


CAPACITY_VARIANT_DISPLAY = {
    "v3_adf_default": "DynaMER-ADF default",
    "v3_no_spike": "DynaMER-ADF no spike",
    "v3_low_spike": "DynaMER-ADF low spike",
    "v3_dropout_0p10": "DynaMER-ADF dropout 0.10",
    "v3_tcn_depth_1": "DynaMER-ADF TCN depth 1",
    "v3_tcn_depth_3": "DynaMER-ADF TCN depth 3",
}

DEFAULT_NESTED_VARIANTS = list(CAPACITY_VARIANT_DISPLAY.keys())

BASE_MODEL_CONFIG: Dict[str, Any] = {
    "hidden_dim": 128,
    "dropout": 0.20,
    "temporal_backbone_v1": "bigru",
    "tcn_layers": 2,
    "modality_dropout": 0.00,
    "spike_steps": 6,
    "spike_decay": 0.85,
    "spike_threshold": 1.0,
    "spike_slope": 5.0,
    "spike_mix": 0.10,
    "path_mode": "learned_dual",
    "fusion_mode": "gated",
}

CAPACITY_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "v3_adf_default": {},
    "v3_no_spike": {"spike_mix": 0.00},
    "v3_low_spike": {"spike_mix": 0.05},
    "v3_dropout_0p10": {"dropout": 0.10},
    "v3_tcn_depth_1": {"tcn_layers": 1},
    "v3_tcn_depth_3": {"tcn_layers": 3},
}

VARIANT_PRIORITY = {v: i for i, v in enumerate(DEFAULT_NESTED_VARIANTS)}

TRAINING_DEFAULTS = {
    "epochs": 50,
    "patience": 10,
    "min_delta": 1e-4,
    "monitor_metric": "val_macro_f1",
    "monitor_mode": "max",
    "learning_rate": 5e-4,
    "weight_decay": 2e-4,
    "gradient_clip_norm": 1.0,
    "use_amp": True,
    "class_weighting": "balanced_train",
}

LOADER_DEFAULTS = {
    "batch_size": 16,
    "num_workers": 0,
    "pin_memory": True,
}


def model_config_for_variant(variant: str) -> Dict[str, Any]:
    variant = str(variant).lower()
    if variant not in CAPACITY_OVERRIDES:
        raise ValueError(f"Unsupported nested-LOSO variant: {variant}")
    cfg = dict(BASE_MODEL_CONFIG)
    cfg.update(CAPACITY_OVERRIDES[variant])
    return cfg


def make_model(modality_keys: List[str], num_classes: int, variant: str, device: torch.device) -> torch.nn.Module:
    cfg = model_config_for_variant(variant)
    return DynaMERADFAblationModel(
        modality_keys=modality_keys,
        num_classes=int(num_classes),
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg["dropout"]),
        temporal_backbone_v1=str(cfg["temporal_backbone_v1"]),
        tcn_layers=int(cfg["tcn_layers"]),
        modality_dropout=float(cfg["modality_dropout"]),
        spike_steps=int(cfg["spike_steps"]),
        spike_decay=float(cfg["spike_decay"]),
        spike_threshold=float(cfg["spike_threshold"]),
        spike_slope=float(cfg["spike_slope"]),
        spike_mix=float(cfg["spike_mix"]),
        path_mode=str(cfg["path_mode"]),
        fusion_mode=str(cfg["fusion_mode"]),
    ).to(device)


def monitor_value_from_val(val_metrics: Dict[str, float], monitor_metric: str) -> float:
    key = str(monitor_metric)
    if key.startswith("val_"):
        key = key[4:]
    return float(val_metrics.get(key, float("nan")))


def make_split_rows(
    manifest: pd.DataFrame,
    label_col: str,
    split_assignments: Dict[str, str],
    split_id: str,
    protocol: str,
    outer_fold: int,
    inner_fold: Optional[int],
    leakage_note: str,
    duplicate_val_as_test: bool = False,
) -> pd.DataFrame:
    base_cols = [
        "split_id", "dataset", "task", "protocol", "outer_fold", "inner_fold", "fold_index",
        "trial_uid", "subject_id", "subject_index", "session_id", "session_index", "trial_id", "trial_index",
        label_col, "label", "split", "is_primary_protocol", "leakage_note",
    ]

    rows = manifest.copy()
    rows[label_col] = pd.to_numeric(rows[label_col], errors="coerce").astype(int)
    rows["label"] = rows[label_col].astype(int)
    rows["split"] = rows["trial_uid"].astype(str).map(split_assignments)
    rows = rows[rows["split"].notna()].copy()

    if duplicate_val_as_test:
        val_rows = rows[rows["split"] == "val"].copy()
        val_rows["split"] = "test"
        rows = pd.concat([rows, val_rows], ignore_index=True)

    rows["split_id"] = split_id
    rows["dataset"] = "SEED-IV"
    rows["task"] = label_col
    rows["protocol"] = protocol
    rows["outer_fold"] = int(outer_fold)
    rows["inner_fold"] = int(inner_fold) if inner_fold is not None else -1
    rows["fold_index"] = int(outer_fold)
    rows["is_primary_protocol"] = 0
    rows["leakage_note"] = leakage_note

    for c in base_cols:
        if c not in rows.columns:
            rows[c] = np.nan
    return rows[base_cols].copy()


def split_counts(df: pd.DataFrame) -> Dict[str, int]:
    return {str(k): int(v) for k, v in df["split"].value_counts().sort_index().to_dict().items()}


def label_counts_by_split(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for split_name, part in df.groupby("split"):
        out[str(split_name)] = {str(int(k)): int(v) for k, v in part["label"].value_counts().sort_index().to_dict().items()}
    return out


def generate_nested_split_files(
    manifest: pd.DataFrame,
    label_col: str,
    out_dir: Path,
    seed: int,
    selected_outer_folds: Optional[List[int]],
    max_outer: Optional[int],
    max_inner_folds: Optional[int],
    logger: Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_dir = out_dir / "split_files"
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = sorted(manifest["subject_id"].astype(str).unique().tolist())
    if selected_outer_folds is not None:
        outer_subject_pairs = [(i + 1, s) for i, s in enumerate(subject_ids) if (i + 1) in set(selected_outer_folds)]
    else:
        outer_subject_pairs = [(i + 1, s) for i, s in enumerate(subject_ids)]
    if max_outer is not None:
        outer_subject_pairs = outer_subject_pairs[: int(max_outer)]

    split_index_rows: List[Dict[str, Any]] = []
    inner_registry_rows: List[Dict[str, Any]] = []
    final_registry_rows: List[Dict[str, Any]] = []

    for outer_fold, test_subject in outer_subject_pairs:
        dev_subjects = [s for s in subject_ids if s != test_subject]
        kf = KFold(n_splits=3, shuffle=True, random_state=int(seed) + outer_fold * 1009)
        inner_subject_arrays = list(kf.split(np.array(dev_subjects)))
        if max_inner_folds is not None:
            inner_subject_arrays = inner_subject_arrays[: int(max_inner_folds)]

        for inner_fold, (train_rel, val_rel) in enumerate(inner_subject_arrays, start=1):
            inner_train_subjects = [dev_subjects[i] for i in train_rel]
            inner_val_subjects = [dev_subjects[i] for i in val_rel]
            assignments: Dict[str, str] = {}
            for _, row in manifest.iterrows():
                uid = str(row["trial_uid"])
                sid = str(row["subject_id"])
                if sid in inner_train_subjects:
                    assignments[uid] = "train"
                elif sid in inner_val_subjects:
                    assignments[uid] = "val"
                # Outer test subject is intentionally excluded from all inner-loop split files.

            split_id = f"SEED-IV__seed_iv_label__nested_loso__outer_{outer_fold:03d}__inner_{inner_fold:02d}"
            note = (
                "Nested LOSO inner-loop split. Outer test subject is excluded. Inner validation subjects are used only "
                "for candidate selection. The test split duplicates the inner validation rows only to satisfy the existing "
                "DataModule interface and is not used for outer evaluation."
            )
            split_df = make_split_rows(
                manifest=manifest,
                label_col=label_col,
                split_assignments=assignments,
                split_id=split_id,
                protocol="nested_loso_inner_cv",
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                leakage_note=note,
                duplicate_val_as_test=True,
            )
            split_path = split_dir / f"23A_inner_outer_{outer_fold:03d}_inner_{inner_fold:02d}.csv"
            split_df.to_csv(split_path, index=False)

            split_index_rows.append({
                "split_id": split_id,
                "split_type": "inner_cv",
                "dataset": "SEED-IV",
                "task": label_col,
                "protocol": "nested_loso_inner_cv",
                "outer_fold": int(outer_fold),
                "inner_fold": int(inner_fold),
                "test_subject": test_subject,
                "train_subjects": "|".join(inner_train_subjects),
                "val_subjects": "|".join(inner_val_subjects),
                "n_rows": int(len(split_df)),
                "split_counts": json.dumps(split_counts(split_df), ensure_ascii=False),
                "label_counts_by_split": json.dumps(label_counts_by_split(split_df), ensure_ascii=False),
                "split_file": str(split_path),
            })
            inner_registry_rows.append({
                "run_id": f"23A_inner__outer_{outer_fold:03d}__inner_{inner_fold:02d}",
                "dataset": "SEED-IV",
                "task": label_col,
                "protocol": "nested_loso_inner_cv",
                "outer_fold": int(outer_fold),
                "inner_fold": int(inner_fold),
                "fold_index": int(outer_fold),
                "test_subject": test_subject,
                "label_column": label_col,
                "modality_keys": "eeg_combined|eye_features",
                "num_classes": 4,
                "split_file": str(split_path),
            })

        # Final outer split: all development subjects are train rows. They are duplicated as val rows only because
        # the existing DataModule requires a non-empty validation split; final training does not use validation to
        # select hyperparameters or checkpoints.
        final_assignments: Dict[str, str] = {}
        for _, row in manifest.iterrows():
            uid = str(row["trial_uid"])
            sid = str(row["subject_id"])
            if sid == test_subject:
                final_assignments[uid] = "test"
            else:
                final_assignments[uid] = "train"

        final_split_id = f"SEED-IV__seed_iv_label__nested_loso__outer_{outer_fold:03d}__final"
        final_note = (
            "Nested LOSO final outer split. Candidate configuration and final epoch count are selected only from "
            "inner subject-level validation. All non-test subjects are used for final training. Development rows are "
            "duplicated as validation rows only to satisfy the current DataModule interface and are not used for outer "
            "model selection."
        )
        final_df = make_split_rows(
            manifest=manifest,
            label_col=label_col,
            split_assignments=final_assignments,
            split_id=final_split_id,
            protocol="nested_loso_outer_final",
            outer_fold=outer_fold,
            inner_fold=None,
            leakage_note=final_note,
            duplicate_val_as_test=False,
        )
        # Duplicate all development training rows as val rows to satisfy DataModule setup, but final training ignores val.
        train_dup = final_df[final_df["split"] == "train"].copy()
        train_dup["split"] = "val"
        final_df = pd.concat([final_df, train_dup], ignore_index=True)
        final_path = split_dir / f"23A_outer_{outer_fold:03d}_final.csv"
        final_df.to_csv(final_path, index=False)
        split_index_rows.append({
            "split_id": final_split_id,
            "split_type": "outer_final",
            "dataset": "SEED-IV",
            "task": label_col,
            "protocol": "nested_loso_outer_final",
            "outer_fold": int(outer_fold),
            "inner_fold": -1,
            "test_subject": test_subject,
            "train_subjects": "|".join(dev_subjects),
            "val_subjects": "DUPLICATED_DEVELOPMENT_ROWS_NOT_USED_FOR_SELECTION",
            "n_rows": int(len(final_df)),
            "split_counts": json.dumps(split_counts(final_df), ensure_ascii=False),
            "label_counts_by_split": json.dumps(label_counts_by_split(final_df), ensure_ascii=False),
            "split_file": str(final_path),
        })
        final_registry_rows.append({
            "run_id": f"23A_outer_final__outer_{outer_fold:03d}",
            "dataset": "SEED-IV",
            "task": label_col,
            "protocol": "nested_loso_outer_final",
            "outer_fold": int(outer_fold),
            "inner_fold": -1,
            "fold_index": int(outer_fold),
            "test_subject": test_subject,
            "label_column": label_col,
            "modality_keys": "eeg_combined|eye_features",
            "num_classes": 4,
            "split_file": str(final_path),
        })

        logger.info(f"Prepared nested LOSO outer fold {outer_fold:03d} test_subject={test_subject}")

    split_index = pd.DataFrame(split_index_rows)
    inner_registry = pd.DataFrame(inner_registry_rows)
    final_registry = pd.DataFrame(final_registry_rows)
    split_index.to_csv(out_dir / "23A_nested_loso_split_index.csv", index=False)
    inner_registry.to_csv(out_dir / "23A_nested_loso_inner_registry.csv", index=False)
    final_registry.to_csv(out_dir / "23A_nested_loso_final_registry.csv", index=False)
    return split_index, inner_registry, final_registry


def build_dm(
    project_root: Path,
    split_file: Path,
    temporal_index: pd.DataFrame,
    modality_keys: List[str],
    label_column: str,
    loader_cfg: Dict[str, Any],
) -> DynaMERTemporalSplitDataModule:
    dm = DynaMERTemporalSplitDataModule(
        project_root=project_root,
        split_file=split_file,
        temporal_view_index=temporal_index,
        modality_keys=modality_keys,
        label_column=label_column,
        batch_size=int(loader_cfg["batch_size"]),
        num_workers=int(loader_cfg["num_workers"]),
        pin_memory=bool(loader_cfg["pin_memory"]),
        fit_train_standardization=True,
        standardization_eps=1e-6,
    )
    dm.setup()
    return dm


def warmup_model(model: torch.nn.Module, train_loader: Any, device: torch.device) -> None:
    first_batch = next(iter(train_loader))
    with torch.no_grad():
        warm_x = {k: v.to(device) for k, v in first_batch["x"].items()}
        warm_masks = {k: v.to(device) for k, v in first_batch["masks"].items()}
        warm_out = model(warm_x, warm_masks)
        if not torch.isfinite(warm_out["logits"]).all():
            raise RuntimeError("Warm-up logits contain NaN or Inf.")


def run_inner_training(
    run: Dict[str, Any],
    variant: str,
    temporal_index: pd.DataFrame,
    project_root: Path,
    out_dir: Path,
    device: torch.device,
    overwrite: bool,
    max_epochs_override: Optional[int],
    base_seed: int,
) -> Dict[str, Any]:
    variant = str(variant).lower()
    run_id = str(run["run_id"])
    outer_fold = int(run["outer_fold"])
    inner_fold = int(run["inner_fold"])
    run_seed = stable_run_seed(base_seed, "inner", variant, outer_fold, inner_fold, run_id)
    set_global_seed(run_seed)

    nested_run_id = f"{variant}__{run_id}"
    run_dir = out_dir / "inner_runs" / sanitize(variant) / f"outer_{outer_fold:03d}" / f"inner_{inner_fold:02d}"
    if run_dir.exists() and not overwrite:
        return {
            "stage": "inner",
            "variant": variant,
            "variant_display": CAPACITY_VARIANT_DISPLAY.get(variant, variant),
            "nested_run_id": nested_run_id,
            "source_run_id": run_id,
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "test_subject": run.get("test_subject", ""),
            "run_seed": int(run_seed),
            "status": "skipped_existing",
            "run_dir": str(run_dir),
            "epoch_metrics_file": str(run_dir / "epoch_metrics.csv"),
            "best_checkpoint": str(run_dir / "best_model.pt"),
            "best_epoch": None,
            "best_monitor_value": None,
            "epochs_completed": 0,
            "parameter_count": None,
            "error": "",
        }
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    loader_cfg = dict(LOADER_DEFAULTS)
    training_cfg = dict(TRAINING_DEFAULTS)
    if max_epochs_override is not None:
        training_cfg["epochs"] = int(max_epochs_override)

    modality_keys = str(run["modality_keys"]).split("|")
    num_classes = int(run["num_classes"])
    label_column = str(run["label_column"])
    dm = build_dm(
        project_root=project_root,
        split_file=Path(str(run["split_file"])),
        temporal_index=temporal_index,
        modality_keys=modality_keys,
        label_column=label_column,
        loader_cfg=loader_cfg,
    )
    train_loader = dm.dataloader("train", shuffle=True)
    val_loader = dm.dataloader("val", shuffle=False)
    test_loader = dm.dataloader("test", shuffle=False)

    model = make_model(modality_keys, num_classes, variant, device)
    warmup_model(model, train_loader, device)

    train_labels = dm.datasets["train"].rows[label_column].astype(int).to_numpy()
    class_weights = None
    if str(training_cfg["class_weighting"]).lower() == "balanced_train":
        class_weights = make_class_weights(train_labels, num_classes=num_classes, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training_cfg["use_amp"] and device.type == "cuda"))
    stopper = EarlyStopper(
        mode=str(training_cfg["monitor_mode"]),
        patience=int(training_cfg["patience"]),
        min_delta=float(training_cfg["min_delta"]),
    )

    cfg = model_config_for_variant(variant)
    best_epoch = None
    best_monitor = None
    best_ckpt = run_dir / "best_model.pt"
    epoch_rows: List[Dict[str, Any]] = []

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        train_metrics = run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=bool(training_cfg["use_amp"]),
            gradient_clip_norm=float(training_cfg["gradient_clip_norm"]),
            num_classes=num_classes,
        )
        val_metrics, _ = run_eval_epoch(model, val_loader, criterion, device, num_classes)
        # Inner test duplicates inner validation and is kept only for interface consistency.
        test_metrics, _ = run_eval_epoch(model, test_loader, criterion, device, num_classes)
        score = monitor_value_from_val(val_metrics, str(training_cfg["monitor_metric"]))
        improved = stopper.improved(score)
        row = {
            "stage": "inner",
            "variant": variant,
            "variant_display": CAPACITY_VARIANT_DISPLAY.get(variant, variant),
            "nested_run_id": nested_run_id,
            "source_run_id": run_id,
            "outer_fold": outer_fold,
            "inner_fold": inner_fold,
            "test_subject": run.get("test_subject", ""),
            "run_seed": int(run_seed),
            "epoch": int(epoch),
            "parameter_count": count_parameters(model),
            "dropout": float(cfg["dropout"]),
            "tcn_layers": int(cfg["tcn_layers"]),
            "modality_dropout": float(cfg["modality_dropout"]),
            "spike_mix": float(cfg["spike_mix"]),
            "train_loss": train_metrics["loss"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_roc_auc": train_metrics["roc_auc"],
            "train_pr_auc": train_metrics["pr_auc"],
            "val_loss": val_metrics["loss"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "test_loss": test_metrics["loss"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_pr_auc": test_metrics["pr_auc"],
            "monitor_metric": training_cfg["monitor_metric"],
            "monitor_value": float(score),
            "improved": bool(improved),
            "early_stop_bad_epochs": int(stopper.bad_epochs),
        }
        epoch_rows.append(row)
        if improved:
            best_epoch = int(epoch)
            best_monitor = float(score)
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_family": "DynaMER-ADF nested LOSO inner",
                "variant": variant,
                "variant_display": CAPACITY_VARIANT_DISPLAY.get(variant, variant),
                "model_config": cfg,
                "run": run,
                "epoch": int(epoch),
                "monitor_metric": training_cfg["monitor_metric"],
                "monitor_value": float(score),
                "run_seed": int(run_seed),
            }, best_ckpt)
        if stopper.should_stop():
            break

    epoch_df = pd.DataFrame(epoch_rows)
    epoch_path = run_dir / "epoch_metrics.csv"
    epoch_df.to_csv(epoch_path, index=False)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_family": "DynaMER-ADF nested LOSO inner",
        "variant": variant,
        "run": run,
        "epoch": int(epoch_df["epoch"].max()) if len(epoch_df) else 0,
        "run_seed": int(run_seed),
    }, run_dir / "last_model.pt")

    return {
        "stage": "inner",
        "variant": variant,
        "variant_display": CAPACITY_VARIANT_DISPLAY.get(variant, variant),
        "nested_run_id": nested_run_id,
        "source_run_id": run_id,
        "outer_fold": outer_fold,
        "inner_fold": inner_fold,
        "test_subject": run.get("test_subject", ""),
        "run_seed": int(run_seed),
        "status": "completed",
        "run_dir": str(run_dir),
        "epoch_metrics_file": str(epoch_path),
        "best_checkpoint": str(best_ckpt),
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_monitor_value": float(best_monitor) if best_monitor is not None else None,
        "epochs_completed": int(epoch_df["epoch"].max()) if len(epoch_df) else 0,
        "parameter_count": count_parameters(model),
        "dropout": float(cfg["dropout"]),
        "tcn_layers": int(cfg["tcn_layers"]),
        "modality_dropout": float(cfg["modality_dropout"]),
        "spike_mix": float(cfg["spike_mix"]),
        "error": "",
    }


def train_final_outer(
    final_run: Dict[str, Any],
    selected_variant: str,
    final_epochs: int,
    temporal_index: pd.DataFrame,
    project_root: Path,
    out_dir: Path,
    device: torch.device,
    overwrite: bool,
    base_seed: int,
) -> Dict[str, Any]:
    selected_variant = str(selected_variant).lower()
    outer_fold = int(final_run["outer_fold"])
    run_id = str(final_run["run_id"])
    run_seed = stable_run_seed(base_seed, "outer_final", selected_variant, outer_fold, run_id, final_epochs)
    set_global_seed(run_seed)

    run_dir = out_dir / "outer_final_runs" / f"outer_{outer_fold:03d}" / sanitize(selected_variant)
    if run_dir.exists() and not overwrite:
        metrics_path = run_dir / "outer_final_metrics.csv"
        return {
            "stage": "outer_final",
            "outer_fold": outer_fold,
            "test_subject": final_run.get("test_subject", ""),
            "selected_variant": selected_variant,
            "selected_variant_display": CAPACITY_VARIANT_DISPLAY.get(selected_variant, selected_variant),
            "final_epochs": int(final_epochs),
            "run_seed": int(run_seed),
            "status": "skipped_existing",
            "run_dir": str(run_dir),
            "outer_final_metrics_file": str(metrics_path),
            "error": "",
        }
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    loader_cfg = dict(LOADER_DEFAULTS)
    training_cfg = dict(TRAINING_DEFAULTS)
    modality_keys = str(final_run["modality_keys"]).split("|")
    num_classes = int(final_run["num_classes"])
    label_column = str(final_run["label_column"])

    dm = build_dm(
        project_root=project_root,
        split_file=Path(str(final_run["split_file"])),
        temporal_index=temporal_index,
        modality_keys=modality_keys,
        label_column=label_column,
        loader_cfg=loader_cfg,
    )
    train_loader = dm.dataloader("train", shuffle=True)
    # Validation rows duplicate development rows and are not used for model selection.
    val_loader = dm.dataloader("val", shuffle=False)
    test_loader = dm.dataloader("test", shuffle=False)

    model = make_model(modality_keys, num_classes, selected_variant, device)
    warmup_model(model, train_loader, device)

    train_labels = dm.datasets["train"].rows[label_column].astype(int).to_numpy()
    class_weights = None
    if str(training_cfg["class_weighting"]).lower() == "balanced_train":
        class_weights = make_class_weights(train_labels, num_classes=num_classes, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training_cfg["use_amp"] and device.type == "cuda"))

    cfg = model_config_for_variant(selected_variant)
    epoch_rows: List[Dict[str, Any]] = []
    for epoch in range(1, int(final_epochs) + 1):
        train_metrics = run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=bool(training_cfg["use_amp"]),
            gradient_clip_norm=float(training_cfg["gradient_clip_norm"]),
            num_classes=num_classes,
        )
        # Logged for traceability only. Not used for candidate or checkpoint selection.
        val_metrics, _ = run_eval_epoch(model, val_loader, criterion, device, num_classes)
        test_metrics, _ = run_eval_epoch(model, test_loader, criterion, device, num_classes)
        epoch_rows.append({
            "stage": "outer_final",
            "outer_fold": outer_fold,
            "test_subject": final_run.get("test_subject", ""),
            "selected_variant": selected_variant,
            "selected_variant_display": CAPACITY_VARIANT_DISPLAY.get(selected_variant, selected_variant),
            "final_epochs": int(final_epochs),
            "run_seed": int(run_seed),
            "epoch": int(epoch),
            "parameter_count": count_parameters(model),
            "dropout": float(cfg["dropout"]),
            "tcn_layers": int(cfg["tcn_layers"]),
            "modality_dropout": float(cfg["modality_dropout"]),
            "spike_mix": float(cfg["spike_mix"]),
            "train_loss": train_metrics["loss"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss_trace_only": val_metrics["loss"],
            "val_balanced_accuracy_trace_only": val_metrics["balanced_accuracy"],
            "val_macro_f1_trace_only": val_metrics["macro_f1"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_macro_precision": test_metrics["macro_precision"],
            "test_macro_recall": test_metrics["macro_recall"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_pr_auc": test_metrics["pr_auc"],
        })

    epoch_df = pd.DataFrame(epoch_rows)
    epoch_path = run_dir / "outer_final_epoch_metrics.csv"
    epoch_df.to_csv(epoch_path, index=False)
    final_row = epoch_df.sort_values("epoch").tail(1).copy()
    metrics_path = run_dir / "outer_final_metrics.csv"
    final_row.to_csv(metrics_path, index=False)
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_family": "DynaMER-ADF nested LOSO outer final",
        "selected_variant": selected_variant,
        "selected_variant_display": CAPACITY_VARIANT_DISPLAY.get(selected_variant, selected_variant),
        "model_config": cfg,
        "final_run": final_run,
        "final_epochs": int(final_epochs),
        "run_seed": int(run_seed),
        "note": "Final model trained on all development subjects for inner-selected fixed epoch count. Outer test subject held out until final evaluation.",
    }, run_dir / "outer_final_model.pt")

    return {
        "stage": "outer_final",
        "outer_fold": outer_fold,
        "test_subject": final_run.get("test_subject", ""),
        "selected_variant": selected_variant,
        "selected_variant_display": CAPACITY_VARIANT_DISPLAY.get(selected_variant, selected_variant),
        "final_epochs": int(final_epochs),
        "run_seed": int(run_seed),
        "status": "completed",
        "run_dir": str(run_dir),
        "outer_final_epoch_metrics_file": str(epoch_path),
        "outer_final_metrics_file": str(metrics_path),
        "test_balanced_accuracy": float(final_row["test_balanced_accuracy"].iloc[0]),
        "test_macro_f1": float(final_row["test_macro_f1"].iloc[0]),
        "test_roc_auc": float(final_row["test_roc_auc"].iloc[0]),
        "test_pr_auc": float(final_row["test_pr_auc"].iloc[0]),
        "parameter_count": int(final_row["parameter_count"].iloc[0]),
        "error": "",
    }


def best_inner_rows(all_inner_epochs: pd.DataFrame) -> pd.DataFrame:
    if all_inner_epochs.empty:
        return pd.DataFrame()
    df = all_inner_epochs.copy()
    df = df.sort_values(["outer_fold", "inner_fold", "variant", "val_macro_f1", "val_balanced_accuracy", "epoch"], ascending=[True, True, True, False, False, True])
    best = df.groupby(["outer_fold", "inner_fold", "variant"], as_index=False).head(1).reset_index(drop=True)
    return best


def select_variant_for_outer(best_inner: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    if best_inner.empty:
        return pd.DataFrame()
    for outer_fold, part in best_inner.groupby("outer_fold"):
        grouped = []
        for variant, vpart in part.groupby("variant"):
            grouped.append({
                "outer_fold": int(outer_fold),
                "variant": str(variant),
                "variant_display": CAPACITY_VARIANT_DISPLAY.get(str(variant), str(variant)),
                "n_inner_folds": int(vpart["inner_fold"].nunique()),
                "mean_inner_val_macro_f1": float(vpart["val_macro_f1"].mean()),
                "std_inner_val_macro_f1": float(vpart["val_macro_f1"].std(ddof=1)) if len(vpart) > 1 else 0.0,
                "mean_inner_val_ba": float(vpart["val_balanced_accuracy"].mean()),
                "median_best_epoch": float(vpart["epoch"].median()),
                "mean_best_epoch": float(vpart["epoch"].mean()),
                "variant_priority": int(VARIANT_PRIORITY.get(str(variant), 999)),
            })
        gdf = pd.DataFrame(grouped)
        gdf = gdf.sort_values(
            ["mean_inner_val_macro_f1", "mean_inner_val_ba", "variant_priority"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        selected = gdf.iloc[0].to_dict()
        selected["selected_variant"] = selected.pop("variant")
        selected["selected_variant_display"] = selected.pop("variant_display")
        selected["final_epochs_from_inner_median"] = int(max(1, round(float(selected["median_best_epoch"]))))
        rows.append(selected)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 23A: Nested LOSO for DynaMER-ADF family on SEED-IV.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--local-paths", default="configs/local_paths.yaml")
    parser.add_argument("--variants", default=",".join(DEFAULT_NESTED_VARIANTS), help="Comma-separated ADF-family variants for inner selection.")
    parser.add_argument("--outer-folds", default=None, help="Optional comma-separated outer fold indices, e.g. 1,2,3.")
    parser.add_argument("--max-outer", type=int, default=None, help="Optional smoke-test limit on number of outer folds.")
    parser.add_argument("--max-inner-folds", type=int, default=None, help="Optional smoke-test limit on inner folds per outer fold.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override for inner training and final training smoke tests.")
    parser.add_argument("--final-epochs", type=int, default=None, help="Optional fixed final training epoch override. Default uses median inner best epoch for selected variant.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-inner", action="store_true", help="Reuse existing inner epoch files and only perform selection/final training.")
    parser.add_argument("--skip-final", action="store_true", help="Run inner CV and selection only, without final outer training.")
    args = parser.parse_args()

    t0 = time.time()
    _main_cfg = load_yaml(Path(args.config))
    local_paths = load_yaml(Path(args.local_paths))
    project_root = as_path(local_paths["PROJECT_ROOT"])

    out_dir = project_root / "outputs" / "protocol_extension" / "23_nested_loso_dynamer_adf"
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(out_dir / "23A_train_nested_loso_dynamer_adf_log.txt")
    logger.info("Starting Stage 23A nested LOSO for DynaMER-ADF family.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {out_dir}")

    manifest_path = project_root / "outputs" / "manifests" / "02_prepare_dataset_manifests" / "02_seed_iv_trial_manifest.csv"
    temporal_index_path = project_root / "outputs" / "temporal_views" / "08_prepare_temporal_feature_views" / "08_temporal_view_index.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing SEED-IV manifest: {manifest_path}")
    if not temporal_index_path.exists():
        raise FileNotFoundError(f"Missing temporal view index: {temporal_index_path}")

    manifest = pd.read_csv(manifest_path)
    label_col = "seed_iv_label"
    manifest = manifest.dropna(subset=[label_col]).copy().reset_index(drop=True)
    manifest[label_col] = pd.to_numeric(manifest[label_col], errors="coerce").astype(int)
    temporal_index = pd.read_csv(temporal_index_path)

    selected_outer = parse_int_list(args.outer_folds)
    variants = [v.strip().lower() for v in str(args.variants).split(",") if v.strip()]
    unknown = sorted(set(variants) - set(CAPACITY_OVERRIDES))
    if unknown:
        raise ValueError(f"Unsupported variants: {unknown}. Supported: {sorted(CAPACITY_OVERRIDES)}")

    split_index, inner_registry, final_registry = generate_nested_split_files(
        manifest=manifest,
        label_col=label_col,
        out_dir=out_dir,
        seed=int(args.seed),
        selected_outer_folds=selected_outer,
        max_outer=args.max_outer,
        max_inner_folds=args.max_inner_folds,
        logger=logger,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Inner registry rows: {len(inner_registry)}")
    logger.info(f"Final registry rows: {len(final_registry)}")
    logger.info(f"Nested variants: {variants}")

    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(x, **kwargs):
            return x

    inner_run_results: List[Dict[str, Any]] = []
    inner_epoch_frames: List[pd.DataFrame] = []

    if not args.skip_inner:
        total_inner_runs = len(inner_registry) * len(variants)
        pbar = tqdm(total=total_inner_runs, desc="Stage 23A inner CV", unit="run")
        for run in inner_registry.to_dict(orient="records"):
            for variant in variants:
                try:
                    result = run_inner_training(
                        run=run,
                        variant=variant,
                        temporal_index=temporal_index,
                        project_root=project_root,
                        out_dir=out_dir,
                        device=device,
                        overwrite=bool(args.overwrite),
                        max_epochs_override=args.epochs,
                        base_seed=int(args.seed),
                    )
                except Exception as e:
                    logger.error(f"Inner run failed: outer={run.get('outer_fold')} inner={run.get('inner_fold')} variant={variant} error={repr(e)}")
                    result = {
                        "stage": "inner", "variant": variant, "variant_display": CAPACITY_VARIANT_DISPLAY.get(variant, variant),
                        "nested_run_id": f"{variant}__{run.get('run_id')}", "source_run_id": run.get("run_id"),
                        "outer_fold": int(run.get("outer_fold", -1)), "inner_fold": int(run.get("inner_fold", -1)),
                        "test_subject": run.get("test_subject", ""), "run_seed": None, "status": "failed",
                        "run_dir": "", "epoch_metrics_file": "", "best_checkpoint": "", "best_epoch": None,
                        "best_monitor_value": None, "epochs_completed": 0, "parameter_count": None, "error": repr(e),
                    }
                inner_run_results.append(result)
                ep_path = result.get("epoch_metrics_file", "")
                if ep_path and Path(ep_path).exists():
                    inner_epoch_frames.append(pd.read_csv(ep_path))
                pbar.update(1)
        pbar.close()
    else:
        logger.info("Skipping inner CV training and reusing existing inner epoch files.")
        for epoch_path in sorted((out_dir / "inner_runs").glob("*/outer_*/inner_*/epoch_metrics.csv")):
            inner_epoch_frames.append(pd.read_csv(epoch_path))
        # run report may already exist
        existing_report = out_dir / "23A_nested_loso_inner_run_report.csv"
        if existing_report.exists():
            inner_run_results = pd.read_csv(existing_report).to_dict(orient="records")

    inner_report = pd.DataFrame(inner_run_results)
    all_inner_epochs = pd.concat(inner_epoch_frames, ignore_index=True) if inner_epoch_frames else pd.DataFrame()
    inner_report_path = out_dir / "23A_nested_loso_inner_run_report.csv"
    inner_epochs_path = out_dir / "23A_nested_loso_inner_all_epoch_metrics.csv"
    inner_best_path = out_dir / "23A_nested_loso_inner_best_epoch_rows.csv"
    selection_path = out_dir / "23A_nested_loso_outer_selection.csv"

    if not inner_report.empty:
        inner_report.to_csv(inner_report_path, index=False)
    all_inner_epochs.to_csv(inner_epochs_path, index=False)
    best_inner = best_inner_rows(all_inner_epochs)
    best_inner.to_csv(inner_best_path, index=False)
    selection_df = select_variant_for_outer(best_inner)
    selection_df.to_csv(selection_path, index=False)

    outer_final_results: List[Dict[str, Any]] = []
    if not args.skip_final:
        if selection_df.empty:
            raise RuntimeError("Cannot run outer final training because no inner selection rows were produced.")
        final_by_outer = {int(r["outer_fold"]): r for r in final_registry.to_dict(orient="records")}
        pbar2 = tqdm(total=len(selection_df), desc="Stage 23A outer final", unit="fold")
        for _, sel in selection_df.iterrows():
            outer_fold = int(sel["outer_fold"])
            if outer_fold not in final_by_outer:
                raise RuntimeError(f"Outer fold {outer_fold} missing from final registry.")
            final_epochs = int(args.final_epochs) if args.final_epochs is not None else int(sel["final_epochs_from_inner_median"])
            if args.epochs is not None:
                final_epochs = int(args.epochs)
            try:
                result = train_final_outer(
                    final_run=final_by_outer[outer_fold],
                    selected_variant=str(sel["selected_variant"]),
                    final_epochs=int(final_epochs),
                    temporal_index=temporal_index,
                    project_root=project_root,
                    out_dir=out_dir,
                    device=device,
                    overwrite=bool(args.overwrite),
                    base_seed=int(args.seed),
                )
            except Exception as e:
                logger.error(f"Outer final failed: outer={outer_fold} variant={sel.get('selected_variant')} error={repr(e)}")
                result = {
                    "stage": "outer_final", "outer_fold": outer_fold, "test_subject": final_by_outer[outer_fold].get("test_subject", ""),
                    "selected_variant": str(sel.get("selected_variant", "")),
                    "selected_variant_display": str(sel.get("selected_variant_display", "")),
                    "final_epochs": int(final_epochs), "run_seed": None, "status": "failed", "run_dir": "",
                    "outer_final_metrics_file": "", "error": repr(e),
                }
            outer_final_results.append(result)
            pbar2.update(1)
        pbar2.close()

    outer_report = pd.DataFrame(outer_final_results)
    outer_report_path = out_dir / "23A_nested_loso_outer_final_run_report.csv"
    outer_metrics_path = out_dir / "23A_nested_loso_outer_final_metrics.csv"
    if not outer_report.empty:
        outer_report.to_csv(outer_report_path, index=False)
        frames = []
        for p in outer_report.get("outer_final_metrics_file", pd.Series(dtype=str)).dropna().astype(str).tolist():
            if p and Path(p).exists():
                frames.append(pd.read_csv(p))
        outer_metrics = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        outer_metrics = pd.DataFrame()
    outer_metrics.to_csv(outer_metrics_path, index=False)

    checks: List[Dict[str, Any]] = []
    add_check(checks, "manifest rows", int(len(manifest)), 1080, int(len(manifest)) == 1080)
    add_check(checks, "split index rows", int(len(split_index)), ">0", int(len(split_index)) > 0)
    add_check(checks, "inner registry rows", int(len(inner_registry)), ">0", int(len(inner_registry)) > 0)
    add_check(checks, "final registry rows", int(len(final_registry)), ">0", int(len(final_registry)) > 0)
    add_check(checks, "inner epoch rows", int(len(all_inner_epochs)), ">0", int(len(all_inner_epochs)) > 0)
    add_check(checks, "inner best rows", int(len(best_inner)), ">0", int(len(best_inner)) > 0)
    add_check(checks, "outer selection rows", int(len(selection_df)), int(len(final_registry)) if not args.skip_final else ">0", int(len(selection_df)) > 0)
    if not args.skip_final:
        add_check(checks, "outer final metrics rows", int(len(outer_metrics)), int(len(final_registry)))
        if "status" in outer_report.columns:
            add_check(checks, "outer final failed runs", int((outer_report["status"] == "failed").sum()), 0)
    if not inner_report.empty and "status" in inner_report.columns:
        add_check(checks, "inner failed runs", int((inner_report["status"] == "failed").sum()), 0)

    checks_df = pd.DataFrame(checks)
    checks_path = out_dir / "23A_nested_loso_checks.csv"
    checks_df.to_csv(checks_path, index=False)
    failed_checks = checks_df[checks_df["passed"] == False]
    overall_passed = len(failed_checks) == 0

    summary_path = out_dir / "23A_nested_loso_summary.json"
    summary = {
        "name": "23A_train_nested_loso_dynamer_adf",
        "created_at": now(),
        "overall_passed": bool(overall_passed),
        "elapsed_seconds": round(time.time() - t0, 3),
        "device": str(device),
        "variants": variants,
        "outer_folds": selected_outer if selected_outer is not None else "all",
        "max_outer": args.max_outer,
        "max_inner_folds": args.max_inner_folds,
        "epoch_override": args.epochs,
        "final_epoch_override": args.final_epochs,
        "row_counts": {
            "split_index": int(len(split_index)),
            "inner_registry": int(len(inner_registry)),
            "final_registry": int(len(final_registry)),
            "inner_epoch_rows": int(len(all_inner_epochs)),
            "inner_best_rows": int(len(best_inner)),
            "selection_rows": int(len(selection_df)),
            "outer_metric_rows": int(len(outer_metrics)),
        },
        "outputs": {
            "split_index": str(out_dir / "23A_nested_loso_split_index.csv"),
            "inner_registry": str(out_dir / "23A_nested_loso_inner_registry.csv"),
            "final_registry": str(out_dir / "23A_nested_loso_final_registry.csv"),
            "inner_run_report": str(inner_report_path),
            "inner_epoch_metrics": str(inner_epochs_path),
            "inner_best_epoch_rows": str(inner_best_path),
            "outer_selection": str(selection_path),
            "outer_final_run_report": str(outer_report_path),
            "outer_final_metrics": str(outer_metrics_path),
            "checks": str(checks_path),
            "summary": str(summary_path),
            "log": str(out_dir / "23A_train_nested_loso_dynamer_adf_log.txt"),
        },
        "failed_checks": failed_checks.to_dict(orient="records"),
        "scientific_statement": (
            "Nested LOSO uses held-out subjects as the outer test loop and inner subject-level cross-validation "
            "within the development subjects for ADF-family candidate selection. The outer test subject is never "
            "used in normalization, inner validation, candidate selection, or final epoch selection."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Overall Stage 23A passed: {overall_passed}")
    print("\nStage 23A outputs:")
    for i, p in enumerate(summary["outputs"].values(), start=1):
        print(f"{i}. {p}")

    if not overall_passed:
        logger.error("Stage 23A did not pass all checks. Inspect checks and reports before using results.")
        return 1
    logger.info("Stage 23A passed. It is safe to run Stage 23B summary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
