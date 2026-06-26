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
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import yaml

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from src.dynamer.data.temporal_data_modules import DynaMERTemporalSplitDataModule
from src.dynamer.models.temporal_baseline_models import BaselineEmotionModel
from src.dynamer.models.dynamer_bitcn_model import DynaMERBiTCNModel
from src.dynamer.models.dynamer_adf_model import DynaMERADFModel
from src.dynamer.models.dynamer_anchor_model import DynaMERAnchorModel
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


def stable_run_seed(base_seed: int, variant: str, fold_index: int, run_id: str) -> int:
    """Return a deterministic per-run seed independent of run order.

    The original Stage 22B set the seed once at process start. That makes
    results depend on whether DynaMER-ADF is trained alone or together with
    other models, because earlier runs consume the RNG state. This function
    prevents order-dependent training by deriving a stable seed for every
    model/fold pair.
    """
    key = f"{int(base_seed)}::{str(variant).lower()}::{int(fold_index)}::{str(run_id)}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 2_147_483_647


def add_check(checks: List[Dict[str, Any]], name: str, observed: Any, expected: Any, passed: Optional[bool] = None) -> None:
    if passed is None:
        passed = observed == expected
    checks.append({
        "check": name,
        "observed": observed,
        "expected": expected,
        "passed": bool(passed),
    })


BASELINE_VARIANTS = {"temporal_mlp", "lstm", "gru", "bilstm", "tcn", "cnn_lstm"}
SUPPORTED_VARIANTS = sorted(BASELINE_VARIANTS | {"dynamer_v2", "dynamer_v3", "dynamer_v5"})


DEFAULT_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "hidden_dim": 128,
        "dropout": 0.20,
    },
    "dynamer_v2": {
        "hidden_dim": 128,
        "dropout": 0.25,
        "tcn_layers": 2,
        "modality_dropout": 0.10,
        "spike_steps": 6,
        "spike_decay": 0.85,
        "spike_threshold": 1.0,
        "spike_slope": 5.0,
        "spike_mix": 0.15,
    },
    "dynamer_v3": {
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
    },
    "dynamer_v5": {
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
        "residual_init_scale": 0.10,
        "residual_max_scale": 0.35,
    },
}


TRAINING_DEFAULTS = {
    "epochs": 50,
    "patience": 10,
    "min_delta": 1e-4,
    "monitor_metric": "val_macro_f1",
    "monitor_mode": "max",
    "learning_rate": 5e-4,
    "weight_decay_baseline": 1e-4,
    "weight_decay_dynamer": 2e-4,
    "gradient_clip_norm": 1.0,
    "use_amp": True,
    "class_weighting": "balanced_train",
}


LOADER_DEFAULTS = {
    "batch_size": 16,
    "num_workers": 0,
    "pin_memory": True,
}


def make_model(run: Dict[str, Any], variant: str, device: torch.device) -> torch.nn.Module:
    modality_keys = str(run["modality_keys"]).split("|")
    num_classes = int(run["num_classes"])
    variant = variant.lower()

    if variant in BASELINE_VARIANTS:
        cfg = DEFAULT_MODEL_CONFIGS["baseline"]
        return BaselineEmotionModel(
            modality_keys=modality_keys,
            num_classes=num_classes,
            variant=variant,
            hidden_dim=int(cfg["hidden_dim"]),
            dropout=float(cfg["dropout"]),
        ).to(device)

    if variant == "dynamer_v2":
        cfg = DEFAULT_MODEL_CONFIGS[variant]
        return DynaMERBiTCNModel(
            modality_keys=modality_keys,
            num_classes=num_classes,
            hidden_dim=int(cfg["hidden_dim"]),
            dropout=float(cfg["dropout"]),
            tcn_layers=int(cfg["tcn_layers"]),
            modality_dropout=float(cfg["modality_dropout"]),
            spike_steps=int(cfg["spike_steps"]),
            spike_decay=float(cfg["spike_decay"]),
            spike_threshold=float(cfg["spike_threshold"]),
            spike_slope=float(cfg["spike_slope"]),
            spike_mix=float(cfg["spike_mix"]),
        ).to(device)

    if variant == "dynamer_v3":
        cfg = DEFAULT_MODEL_CONFIGS[variant]
        return DynaMERADFModel(
            modality_keys=modality_keys,
            num_classes=num_classes,
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
        ).to(device)

    if variant == "dynamer_v5":
        cfg = DEFAULT_MODEL_CONFIGS[variant]
        return DynaMERAnchorModel(
            modality_keys=modality_keys,
            num_classes=num_classes,
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
            residual_init_scale=float(cfg["residual_init_scale"]),
            residual_max_scale=float(cfg["residual_max_scale"]),
        ).to(device)

    raise ValueError(f"Unsupported model variant: {variant}. Supported: {SUPPORTED_VARIANTS}")


def monitor_value_from_val(val_metrics: Dict[str, float], monitor_metric: str) -> float:
    key = str(monitor_metric)
    if key.startswith("val_"):
        key = key[4:]
    return float(val_metrics.get(key, float("nan")))


def run_single_training(
    run: Dict[str, Any],
    variant: str,
    temporal_index: pd.DataFrame,
    project_root: Path,
    out_dir: Path,
    device: torch.device,
    overwrite: bool,
    save_predictions: bool,
    logger: Logger,
    max_epochs_override: Optional[int] = None,
    base_seed: int = 42,
) -> Dict[str, Any]:
    run_id = str(run["run_id"])
    variant = variant.lower()
    fold_index = int(run["fold_index"])
    run_seed = stable_run_seed(int(base_seed), variant, fold_index, run_id)
    set_global_seed(run_seed)
    variant_run_id = f"{variant}__{run_id}"
    run_dir = out_dir / "runs" / sanitize(variant) / sanitize(run_id)

    if run_dir.exists() and not overwrite:
        return {
            "model_variant": variant,
            "variant_run_id": variant_run_id,
            "source_run_id": run_id,
            "dataset": run["dataset"],
            "task": run["task"],
            "protocol": run["protocol"],
            "fold_index": int(run["fold_index"]),
            "run_seed": int(run_seed),
            "status": "skipped_existing",
            "run_dir": str(run_dir),
            "best_checkpoint": str(run_dir / "best_model.pt"),
            "last_checkpoint": str(run_dir / "last_model.pt"),
            "prediction_file": str(run_dir / "predictions_best_epoch.csv") if (run_dir / "predictions_best_epoch.csv").exists() else "",
            "epoch_metrics_file": str(run_dir / "epoch_metrics.csv") if (run_dir / "epoch_metrics.csv").exists() else "",
            "best_epoch": None,
            "best_monitor_value": None,
            "epochs_completed": 0,
            "parameter_count": None,
            "error": "",
        }

    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    loader_cfg = LOADER_DEFAULTS
    training_cfg = TRAINING_DEFAULTS.copy()
    if max_epochs_override is not None:
        training_cfg["epochs"] = int(max_epochs_override)

    dm = DynaMERTemporalSplitDataModule(
        project_root=project_root,
        split_file=Path(str(run["split_file"])),
        temporal_view_index=temporal_index,
        modality_keys=str(run["modality_keys"]).split("|"),
        label_column=str(run["label_column"]),
        batch_size=int(loader_cfg["batch_size"]),
        num_workers=int(loader_cfg["num_workers"]),
        pin_memory=bool(loader_cfg["pin_memory"]),
        fit_train_standardization=True,
        standardization_eps=1e-6,
    )
    dm.setup()
    train_loader = dm.dataloader("train", shuffle=True)
    val_loader = dm.dataloader("val", shuffle=False)
    test_loader = dm.dataloader("test", shuffle=False)

    model = make_model(run, variant, device)

    # Warm up LazyLinear parameters before optimizer construction.
    first_batch = next(iter(train_loader))
    with torch.no_grad():
        warm_x = {k: v.to(device) for k, v in first_batch["x"].items()}
        warm_masks = {k: v.to(device) for k, v in first_batch["masks"].items()}
        warm_out = model(warm_x, warm_masks)
        if not torch.isfinite(warm_out["logits"]).all():
            raise RuntimeError("Warm-up logits contain NaN or Inf.")

    num_classes = int(run["num_classes"])
    train_labels = dm.datasets["train"].rows[str(run["label_column"])].astype(int).to_numpy()
    class_weights = None
    if str(training_cfg["class_weighting"]).lower() == "balanced_train":
        class_weights = make_class_weights(train_labels, num_classes=num_classes, device=device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    weight_decay = training_cfg["weight_decay_baseline"] if variant in BASELINE_VARIANTS else training_cfg["weight_decay_dynamer"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(weight_decay),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(training_cfg["use_amp"] and device.type == "cuda"))
    stopper = EarlyStopper(
        mode=str(training_cfg["monitor_mode"]),
        patience=int(training_cfg["patience"]),
        min_delta=float(training_cfg["min_delta"]),
    )

    best_epoch = None
    best_monitor = None
    best_ckpt = run_dir / "best_model.pt"
    last_ckpt = run_dir / "last_model.pt"
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
        val_metrics, _ = run_eval_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )
        test_metrics, _ = run_eval_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )
        score = monitor_value_from_val(val_metrics, str(training_cfg["monitor_metric"]))
        improved = stopper.improved(score)

        row = {
            "model_variant": variant,
            "variant_run_id": variant_run_id,
            "source_run_id": run_id,
            "dataset": run["dataset"],
            "task": run["task"],
            "protocol": run["protocol"],
            "fold_index": int(run["fold_index"]),
            "epoch": int(epoch),
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_balanced_accuracy": train_metrics["balanced_accuracy"],
            "train_macro_precision": train_metrics["macro_precision"],
            "train_macro_recall": train_metrics["macro_recall"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_roc_auc": train_metrics["roc_auc"],
            "train_pr_auc": train_metrics["pr_auc"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_pr_auc": val_metrics["pr_auc"],
            "test_loss": test_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_macro_precision": test_metrics["macro_precision"],
            "test_macro_recall": test_metrics["macro_recall"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_pr_auc": test_metrics["pr_auc"],
            "monitor_metric": training_cfg["monitor_metric"],
            "monitor_value": score,
            "improved": bool(improved),
            "early_stop_bad_epochs": int(stopper.bad_epochs),
        }
        epoch_rows.append(row)

        if improved:
            best_epoch = int(epoch)
            best_monitor = float(score)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_variant": variant,
                    "model_config": DEFAULT_MODEL_CONFIGS["baseline" if variant in BASELINE_VARIANTS else variant],
                    "run": run,
                    "epoch": int(epoch),
                    "monitor_metric": training_cfg["monitor_metric"],
                    "monitor_value": float(score),
                },
                best_ckpt,
            )
        if stopper.should_stop():
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_variant": variant,
            "run": run,
            "epoch": int(epoch_rows[-1]["epoch"]),
        },
        last_ckpt,
    )

    epoch_df = pd.DataFrame(epoch_rows)
    epoch_path = run_dir / "epoch_metrics.csv"
    epoch_df.to_csv(epoch_path, index=False)

    prediction_path = ""
    if save_predictions:
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        _, val_preds = run_eval_epoch(model, val_loader, criterion, device, num_classes, collect_predictions=True)
        _, test_preds = run_eval_epoch(model, test_loader, criterion, device, num_classes, collect_predictions=True)
        pred_df = pd.DataFrame(
            [{**r, "split": "val"} for r in val_preds] +
            [{**r, "split": "test"} for r in test_preds]
        )
        pred_path = run_dir / "predictions_best_epoch.csv"
        pred_df.to_csv(pred_path, index=False)
        prediction_path = str(pred_path)

    return {
        "model_variant": variant,
        "variant_run_id": variant_run_id,
        "source_run_id": run_id,
        "dataset": run["dataset"],
        "task": run["task"],
        "protocol": run["protocol"],
        "fold_index": int(run["fold_index"]),
        "run_seed": int(run_seed),
        "status": "completed",
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_ckpt),
        "last_checkpoint": str(last_ckpt),
        "prediction_file": prediction_path,
        "epoch_metrics_file": str(epoch_path),
        "best_epoch": best_epoch,
        "best_monitor_value": best_monitor,
        "epochs_completed": int(epoch_df["epoch"].max()) if len(epoch_df) else 0,
        "parameter_count": count_parameters(model),
        "error": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 22B: train SEED-IV subject-mixed 5-fold diagnostic models.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--local-paths", default="configs/local_paths.yaml")
    parser.add_argument("--registry", default=None, help="Optional Stage 22A registry path. Defaults to the standard output path.")
    parser.add_argument("--models", default="dynamer_v3", help="Comma-separated models. Recommended first run: dynamer_v3. Full: dynamer_v3,bilstm,tcn,dynamer_v2,dynamer_v5")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override for quick smoke tests.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-save-predictions", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    _main_cfg = load_yaml(Path(args.config))
    local_paths = load_yaml(Path(args.local_paths))
    project_root = as_path(local_paths["PROJECT_ROOT"])
    out_dir = project_root / "outputs" / "protocol_extension" / "22_seed_iv_subject_mixed_5fold" / "training_order_safe"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(out_dir / "22B_train_seed_iv_subject_mixed_5fold_log.txt")
    logger.info("Starting Stage 22B order-safe subject-mixed 5-fold training.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {out_dir}")

    registry_path = Path(args.registry) if args.registry else project_root / "outputs" / "protocol_extension" / "22_seed_iv_subject_mixed_5fold" / "22_seed_iv_subject_mixed_5fold_registry.csv"
    temporal_index_path = project_root / "outputs" / "temporal_views" / "08_prepare_temporal_feature_views" / "08_temporal_view_index.csv"
    split_summary_path = project_root / "outputs" / "protocol_extension" / "22_seed_iv_subject_mixed_5fold" / "22_seed_iv_subject_mixed_5fold_summary.json"

    if not registry_path.exists():
        raise FileNotFoundError(f"Missing Stage 22A registry: {registry_path}")
    if not temporal_index_path.exists():
        raise FileNotFoundError(f"Missing temporal view index: {temporal_index_path}")
    if split_summary_path.exists():
        split_summary = json.loads(split_summary_path.read_text(encoding="utf-8"))
        if not bool(split_summary.get("overall_passed", False)):
            raise RuntimeError(f"Stage 22A summary exists but did not pass: {split_summary_path}")

    set_global_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    registry = pd.read_csv(registry_path)
    registry = registry.sort_values(["fold_index"]).reset_index(drop=True)
    if args.max_runs is not None:
        registry = registry.head(int(args.max_runs)).copy()

    models = [m.strip().lower() for m in str(args.models).split(",") if m.strip()]
    unknown = sorted(set(models) - set(SUPPORTED_VARIANTS))
    if unknown:
        raise ValueError(f"Unsupported model(s): {unknown}. Supported: {SUPPORTED_VARIANTS}")

    temporal_index = pd.read_csv(temporal_index_path)

    plan_rows = []
    for run in registry.to_dict(orient="records"):
        for model_variant in models:
            plan_rows.append({
                "model_variant": model_variant,
                "source_run_id": run["run_id"],
                "dataset": run["dataset"],
                "task": run["task"],
                "protocol": run["protocol"],
                "fold_index": int(run["fold_index"]),
            })
    plan_df = pd.DataFrame(plan_rows)
    plan_path = out_dir / "22B_subject_mixed_5fold_training_plan.csv"
    plan_df.to_csv(plan_path, index=False)

    logger.info(f"Selected registry rows: {len(registry)}")
    logger.info(f"Selected models: {models}")
    logger.info(f"Planned runs: {len(plan_df)}")

    run_results: List[Dict[str, Any]] = []
    all_epoch_frames: List[pd.DataFrame] = []

    try:
        from tqdm.auto import tqdm
    except Exception:
        def tqdm(x, **kwargs):
            return x

    total = len(registry) * len(models)
    pbar = tqdm(total=total, desc="Stage 22B training", unit="run")

    for run in registry.to_dict(orient="records"):
        for model_variant in models:
            try:
                result = run_single_training(
                    run=run,
                    variant=model_variant,
                    temporal_index=temporal_index,
                    project_root=project_root,
                    out_dir=out_dir,
                    device=device,
                    overwrite=bool(args.overwrite),
                    save_predictions=not bool(args.no_save_predictions),
                    logger=logger,
                    max_epochs_override=args.epochs,
                    base_seed=int(args.seed),
                )
            except Exception as e:
                logger.error(f"Run failed: model={model_variant} run={run.get('run_id')} error={repr(e)}")
                result = {
                    "model_variant": model_variant,
                    "variant_run_id": f"{model_variant}__{run.get('run_id')}",
                    "source_run_id": run.get("run_id"),
                    "dataset": run.get("dataset"),
                    "task": run.get("task"),
                    "protocol": run.get("protocol"),
                    "fold_index": int(run.get("fold_index", -1)),
                    "run_seed": None,
                    "status": "failed",
                    "run_dir": "",
                    "best_checkpoint": "",
                    "last_checkpoint": "",
                    "prediction_file": "",
                    "epoch_metrics_file": "",
                    "best_epoch": None,
                    "best_monitor_value": None,
                    "epochs_completed": 0,
                    "parameter_count": None,
                    "error": repr(e),
                }
            run_results.append(result)
            ep_path = result.get("epoch_metrics_file", "")
            if ep_path and Path(ep_path).exists():
                all_epoch_frames.append(pd.read_csv(ep_path))
            pbar.update(1)
    pbar.close()

    run_report = pd.DataFrame(run_results)
    all_epochs = pd.concat(all_epoch_frames, ignore_index=True) if all_epoch_frames else pd.DataFrame()

    run_report_path = out_dir / "22B_subject_mixed_5fold_training_run_report.csv"
    epoch_report_path = out_dir / "22B_subject_mixed_5fold_all_epoch_metrics.csv"
    checks_path = out_dir / "22B_subject_mixed_5fold_training_checks.csv"
    summary_path = out_dir / "22B_subject_mixed_5fold_training_summary.json"

    run_report.to_csv(run_report_path, index=False)
    all_epochs.to_csv(epoch_report_path, index=False)

    checks: List[Dict[str, Any]] = []
    add_check(checks, "selected registry rows", int(len(registry)), ">0", int(len(registry)) > 0)
    add_check(checks, "selected models", models, models, len(models) > 0)
    add_check(checks, "planned runs", int(len(plan_df)), int(len(registry)) * int(len(models)))
    add_check(checks, "runs completed or skipped", int(run_report["status"].isin(["completed", "skipped_existing"]).sum()), int(len(plan_df)))
    add_check(checks, "failed runs", int((run_report["status"] == "failed").sum()), 0)
    add_check(checks, "epoch metric rows", int(len(all_epochs)), ">0", int(len(all_epochs)) > 0)
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(checks_path, index=False)

    failed_checks = checks_df[checks_df["passed"] == False]
    failed_runs = run_report[run_report["status"] == "failed"]
    overall_passed = len(failed_checks) == 0 and len(failed_runs) == 0

    summary = {
        "name": "22B_train_seed_iv_subject_mixed_5fold_order_safe",
        "created_at": now(),
        "overall_passed": bool(overall_passed),
        "elapsed_seconds": round(time.time() - t0, 3),
        "device": str(device),
        "models": models,
        "row_counts": {
            "registry_rows": int(len(registry)),
            "planned_runs": int(len(plan_df)),
            "run_report_rows": int(len(run_report)),
            "epoch_metric_rows": int(len(all_epochs)),
        },
        "outputs": {
            "plan": str(plan_path),
            "run_report": str(run_report_path),
            "epoch_metrics": str(epoch_report_path),
            "checks": str(checks_path),
            "summary": str(summary_path),
            "log": str(out_dir / "22B_train_seed_iv_subject_mixed_5fold_log.txt"),
            "runs_dir": str(out_dir / "runs"),
        },
        "failed_checks": failed_checks.to_dict(orient="records"),
        "failed_runs": failed_runs.to_dict(orient="records"),
        "scientific_statement": "Stage 22B trains diagnostic SEED-IV subject-mixed 5-fold models using the same temporal views, train-only standardization, validation-based checkpointing, and test-only final metrics as the main pipeline. Results must be interpreted as subject-mixed capacity estimates, not as subject-independent deployment evidence.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Wrote run report: {run_report_path}")
    logger.info(f"Wrote epoch metrics: {epoch_report_path}")
    logger.info(f"Overall Stage 22B passed: {overall_passed}")

    print("\nStage 22B outputs:")
    print(f"1. {plan_path}")
    print(f"2. {run_report_path}")
    print(f"3. {epoch_report_path}")
    print(f"4. {checks_path}")
    print(f"5. {summary_path}")
    print(f"6. {out_dir / '22B_train_seed_iv_subject_mixed_5fold_log.txt'}")
    print(f"7. {out_dir / 'runs'}")

    if not overall_passed:
        logger.error("Stage 22B failed. Inspect failed runs before summarizing.")
        return 1
    logger.info("Stage 22B passed. It is safe to run Stage 22C summary.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
