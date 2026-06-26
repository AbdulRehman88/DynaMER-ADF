from __future__ import annotations

import argparse
import json
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

from dynamer.data.temporal_data_modules import DynaMERTemporalSplitDataModule
from dynamer.models.dynamer_base_model import DynaMERBaseModel
from dynamer.training.full_engine import (
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
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")

    def info(self, msg: str) -> None:
        line = f"[{now()}] [INFO] {msg}"
        print(line)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def error(self, msg: str) -> None:
        line = f"[{now()}] [ERROR] {msg}"
        print(line)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def as_path(x: str) -> Path:
    return Path(x).expanduser().resolve()


def require_passed_json(project_root: Path, rel_path: str, logger: Logger) -> None:
    path = project_root / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Required previous summary not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    passed = bool(data.get("overall_passed", False))
    logger.info(f"Required previous summary found: {path}")
    logger.info(f"Previous stage passed: {passed}")
    if not passed:
        raise RuntimeError(f"Previous stage did not pass: {path}")


def add_check(checks: List[Dict[str, Any]], check: str, observed: Any, expected: Any, passed: Optional[bool] = None) -> None:
    if passed is None:
        passed = observed == expected
    checks.append(
        {
            "check": check,
            "observed": json.dumps(observed, ensure_ascii=False),
            "expected": json.dumps(expected, ensure_ascii=False),
            "passed": bool(passed),
        }
    )


def safe_run_dir_name(run_id: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(run_id)).strip("_")


def select_registry_runs(
    registry: pd.DataFrame,
    config_phase: Optional[str],
    cli_phase: Optional[str],
    include_diagnostic: bool,
    max_runs: Optional[int],
) -> pd.DataFrame:
    phase = cli_phase if cli_phase else config_phase
    selected = registry.copy()

    if phase and str(phase).lower() not in {"all", "none", "null"}:
        selected = selected[selected["phase"] == phase].copy()

    if not include_diagnostic:
        selected = selected[selected["is_primary_claim_allowed"].astype(int) == 1].copy()

    selected = selected.sort_values(["phase_priority", "dataset", "task", "protocol", "fold_index"]).reset_index(drop=True)

    if max_runs is not None:
        selected = selected.head(int(max_runs)).copy()

    return selected


def make_model(run: Dict[str, Any], model_cfg: Dict[str, Any], device: torch.device) -> DynaMERBaseModel:
    return DynaMERBaseModel(
        modality_keys=str(run["modality_keys"]).split("|"),
        num_classes=int(run["num_classes"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        temporal_backbone=str(model_cfg["temporal_backbone"]),
        fusion=str(model_cfg["fusion"]),
        head=str(model_cfg["head"]),
        spike_steps=int(model_cfg["spike_steps"]),
        spike_decay=float(model_cfg["spike_decay"]),
        spike_threshold=float(model_cfg["spike_threshold"]),
        spike_slope=float(model_cfg["spike_slope"]),
    ).to(device)


def monitor_value(metrics_row: Dict[str, Any], monitor_metric: str) -> float:
    value = metrics_row.get(monitor_metric, np.nan)
    try:
        return float(value)
    except Exception:
        return float("nan")


def run_single_training(
    run: Dict[str, Any],
    temporal_index: pd.DataFrame,
    project_root: Path,
    out_dir: Path,
    cfg: Dict[str, Any],
    device: torch.device,
    logger: Logger,
) -> Dict[str, Any]:
    run_id = str(run["run_id"])
    run_dir = out_dir / "runs" / safe_run_dir_name(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    done_file = run_dir / "DONE.json"

    if done_file.exists() and not bool(cfg["execution"]["overwrite_existing"]):
        return {
            "run_id": run_id,
            "dataset": str(run["dataset"]),
            "task": str(run["task"]),
            "protocol": str(run["protocol"]),
            "fold_index": int(run["fold_index"]),
            "status": "skipped_existing",
            "run_dir": str(run_dir),
            "best_checkpoint": str(run_dir / "best_model.pt"),
            "last_checkpoint": str(run_dir / "last_model.pt"),
            "best_epoch": None,
            "best_monitor_value": None,
            "epochs_completed": None,
            "parameter_count": None,
            "error": "",
        }

    if bool(cfg["execution"]["overwrite_existing"]) and run_dir.exists():
        for child in run_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)

    training_cfg = cfg["training"]
    loader_cfg = cfg["loader"]

    dm = DynaMERTemporalSplitDataModule(
        split_file=Path(str(run["split_file"])),
        temporal_view_index=temporal_index,
        project_root=project_root,
        modality_keys=str(run["modality_keys"]).split("|"),
        label_column=str(run["label_column"]),
        batch_size=int(loader_cfg["batch_size"]),
        num_workers=int(loader_cfg["num_workers"]),
        pin_memory=bool(loader_cfg["pin_memory"]),
        fit_train_standardization=bool(cfg.get("normalization", {}).get("fit_train_split_zscore", True)),
        standardization_eps=float(cfg.get("normalization", {}).get("eps", 1e-6)),
    )
    dm.setup()

    train_loader = dm.dataloader("train", shuffle=True)
    val_loader = dm.dataloader("val", shuffle=False)
    test_loader = dm.dataloader("test", shuffle=False)

    model = make_model(run, cfg["model"], device)

    # Warm-up initializes LazyLinear layers before optimizer and class-weighted criterion.
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

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(training_cfg["use_amp"] and device.type == "cuda"))

    monitor_metric = str(training_cfg["monitor_metric"])
    monitor_mode = str(training_cfg["monitor_mode"])
    stopper = EarlyStopper(
        mode=monitor_mode,
        patience=int(training_cfg["patience"]),
        min_delta=float(training_cfg["min_delta"]),
    )

    epoch_rows: List[Dict[str, Any]] = []
    best_epoch = None
    best_monitor = None
    best_ckpt = run_dir / "best_model.pt"
    last_ckpt = run_dir / "last_model.pt"

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
            collect_predictions=False,
        )

        test_metrics, _ = run_eval_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            collect_predictions=False,
        )

        row = {
            "run_id": run_id,
            "dataset": str(run["dataset"]),
            "task": str(run["task"]),
            "protocol": str(run["protocol"]),
            "fold_index": int(run["fold_index"]),
            "epoch": epoch,
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
        }

        score = monitor_value(row, monitor_metric)
        improved = stopper.improved(score)
        row["monitor_metric"] = monitor_metric
        row["monitor_value"] = score
        row["improved"] = int(improved)
        row["early_stop_bad_epochs"] = int(stopper.bad_epochs)

        epoch_rows.append(row)

        if improved:
            best_epoch = epoch
            best_monitor = score

            if bool(cfg["execution"]["save_best_checkpoint"]):
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": cfg["model"],
                        "run": run,
                        "epoch": epoch,
                        "monitor_metric": monitor_metric,
                        "monitor_value": score,
                        "class_weights": class_weights.detach().cpu().numpy().tolist() if class_weights is not None else None,
                        "created_at": now(),
                    },
                    best_ckpt,
                )

        if stopper.should_stop():
            logger.info(f"Early stopping: {run_id} at epoch {epoch}")
            break

    if bool(cfg["execution"]["save_last_checkpoint"]):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": cfg["model"],
                "run": run,
                "epoch": int(epoch_rows[-1]["epoch"]),
                "created_at": now(),
            },
            last_ckpt,
        )

    epoch_df = pd.DataFrame(epoch_rows)
    epoch_path = run_dir / "epoch_metrics.csv"
    epoch_df.to_csv(epoch_path, index=False)

    prediction_path = ""

    if bool(cfg["execution"]["save_predictions"]):
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

        _, val_preds = run_eval_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            collect_predictions=True,
        )
        _, test_preds = run_eval_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            collect_predictions=True,
        )

        pred_rows = []
        for r in val_preds:
            r["run_id"] = run_id
            r["dataset"] = str(run["dataset"])
            r["task"] = str(run["task"])
            r["protocol"] = str(run["protocol"])
            r["fold_index"] = int(run["fold_index"])
            pred_rows.append(r)
        for r in test_preds:
            r["run_id"] = run_id
            r["dataset"] = str(run["dataset"])
            r["task"] = str(run["task"])
            r["protocol"] = str(run["protocol"])
            r["fold_index"] = int(run["fold_index"])
            pred_rows.append(r)

        pred_df = pd.DataFrame(pred_rows)
        pred_path = run_dir / "val_test_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        prediction_path = str(pred_path)

    done_payload = {
        "run_id": run_id,
        "status": "completed",
        "best_epoch": best_epoch,
        "best_monitor_value": best_monitor,
        "epochs_completed": int(epoch_df["epoch"].max()) if len(epoch_df) else 0,
        "created_at": now(),
    }
    done_file.write_text(json.dumps(done_payload, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "dataset": str(run["dataset"]),
        "task": str(run["task"]),
        "protocol": str(run["protocol"]),
        "fold_index": int(run["fold_index"]),
        "status": "completed",
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_ckpt) if best_ckpt.exists() else "",
        "last_checkpoint": str(last_ckpt) if last_ckpt.exists() else "",
        "prediction_file": prediction_path,
        "epoch_metrics_file": str(epoch_path),
        "best_epoch": best_epoch,
        "best_monitor_value": best_monitor,
        "epochs_completed": int(epoch_df["epoch"].max()) if len(epoch_df) else 0,
        "parameter_count": count_parameters(model),
        "error": "",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="12_controlled_full_training: full training with registry-controlled runs.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-paths", required=True)
    parser.add_argument("--full-training-config", default="configs/12_controlled_full_training.yaml")
    parser.add_argument("--phase", default=None, help="Override phase_filter. Use all for all registry rows.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional max runs override.")
    parser.add_argument("--include-diagnostic", action="store_true", help="Include subject-mixed diagnostic runs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing run outputs.")
    args = parser.parse_args()

    t0 = time.time()

    _main_cfg = load_yaml(Path(args.config))
    local_paths = load_yaml(Path(args.local_paths))
    full_cfg = load_yaml(Path(args.full_training_config))["full_training"]

    project_root = as_path(local_paths["PROJECT_ROOT"])
    out_dir = project_root / full_cfg["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(out_dir / "12_controlled_full_training_log.txt")
    logger.info("Starting 12_controlled_full_training.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {out_dir}")

    req = full_cfg["required_previous_steps"]
    if req.get("require_passed", True):
        require_passed_json(project_root, req["registry_summary_json"], logger)
        require_passed_json(project_root, req["temporal_data_module_summary_json"], logger)

    if args.overwrite:
        full_cfg["execution"]["overwrite_existing"] = True
    if args.include_diagnostic:
        full_cfg["execution"]["include_diagnostic_upper_bound"] = True

    seed = int(full_cfg["random_seed"])
    set_global_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    registry = pd.read_csv(project_root / full_cfg["inputs"]["experiment_registry"])
    temporal_index = pd.read_csv(project_root / full_cfg["inputs"]["temporal_view_index"])

    config_max_runs = full_cfg["execution"].get("max_runs", None)
    max_runs = args.max_runs if args.max_runs is not None else config_max_runs

    selected = select_registry_runs(
        registry=registry,
        config_phase=full_cfg["execution"].get("phase_filter", None),
        cli_phase=args.phase,
        include_diagnostic=bool(full_cfg["execution"]["include_diagnostic_upper_bound"]),
        max_runs=max_runs,
    )

    selected_path = out_dir / "12_selected_training_runs.csv"
    selected.to_csv(selected_path, index=False)

    logger.info(f"Selected runs: {len(selected)}")
    logger.info(f"Wrote selected runs: {selected_path}")

    run_rows: List[Dict[str, Any]] = []

    from tqdm.auto import tqdm

    for run in tqdm(selected.to_dict(orient="records"), desc="Controlled full training", unit="run"):
        try:
            result = run_single_training(
                run=run,
                temporal_index=temporal_index,
                project_root=project_root,
                out_dir=out_dir,
                cfg=full_cfg,
                device=device,
                logger=logger,
            )
        except Exception as exc:
            result = {
                "run_id": str(run.get("run_id", "")),
                "dataset": str(run.get("dataset", "")),
                "task": str(run.get("task", "")),
                "protocol": str(run.get("protocol", "")),
                "fold_index": int(run.get("fold_index", -1)),
                "status": "failed",
                "run_dir": "",
                "best_checkpoint": "",
                "last_checkpoint": "",
                "prediction_file": "",
                "epoch_metrics_file": "",
                "best_epoch": None,
                "best_monitor_value": None,
                "epochs_completed": None,
                "parameter_count": None,
                "error": f"{type(exc).__name__}: {exc}",
            }
            logger.error(f"Run failed: {result['run_id']} :: {result['error']}")

        run_rows.append(result)

    run_report = pd.DataFrame(run_rows)

    all_epoch_rows = []
    for path in run_report["epoch_metrics_file"].dropna().astype(str):
        if path and Path(path).exists():
            df = pd.read_csv(path)
            all_epoch_rows.append(df)

    all_epochs = pd.concat(all_epoch_rows, ignore_index=True) if all_epoch_rows else pd.DataFrame()

    checks: List[Dict[str, Any]] = []
    completed_or_skipped = int(run_report["status"].isin(["completed", "skipped_existing"]).sum())
    add_check(checks, "selected runs", int(len(selected)), int(len(run_report)))
    add_check(checks, "runs completed or skipped", completed_or_skipped, int(len(run_report)))
    add_check(checks, "failed runs", int((run_report["status"] == "failed").sum()), 0)
    add_check(checks, "epoch metric rows generated", int(len(all_epochs)), ">0", int(len(all_epochs)) > 0 or int(len(selected)) == 0)
    add_check(checks, "best checkpoints present for completed runs", int(sum(Path(p).exists() for p in run_report.loc[run_report["status"] == "completed", "best_checkpoint"].astype(str) if p)), int((run_report["status"] == "completed").sum()))
    add_check(checks, "datasets trained", sorted(run_report["dataset"].dropna().unique().tolist()), sorted(selected["dataset"].dropna().unique().tolist()))

    checks_df = pd.DataFrame(checks)

    run_report_path = out_dir / "12_training_run_report.csv"
    epoch_report_path = out_dir / "12_all_epoch_metrics.csv"
    checks_path = out_dir / "12_full_training_checks.csv"
    summary_path = out_dir / "12_full_training_summary.json"

    run_report.to_csv(run_report_path, index=False)
    all_epochs.to_csv(epoch_report_path, index=False)
    checks_df.to_csv(checks_path, index=False)

    failed_checks = checks_df[checks_df["passed"] == False]
    failed_runs = run_report[run_report["status"] == "failed"]
    overall_passed = len(failed_checks) == 0 and len(failed_runs) == 0

    summary = {
        "name": full_cfg["name"],
        "created_at": now(),
        "overall_passed": bool(overall_passed),
        "elapsed_seconds": round(time.time() - t0, 3),
        "device": str(device),
        "selected_phase": args.phase if args.phase else full_cfg["execution"].get("phase_filter", None),
        "row_counts": {
            "selected_runs": int(len(selected)),
            "completed_runs": int((run_report["status"] == "completed").sum()),
            "skipped_existing_runs": int((run_report["status"] == "skipped_existing").sum()),
            "failed_runs": int((run_report["status"] == "failed").sum()),
            "epoch_metric_rows": int(len(all_epochs)),
        },
        "outputs": {
            "selected_runs": str(selected_path),
            "run_report": str(run_report_path),
            "epoch_metrics": str(epoch_report_path),
            "checks": str(checks_path),
            "summary": str(summary_path),
            "log": str(out_dir / "12_controlled_full_training_log.txt"),
            "runs_dir": str(out_dir / "runs"),
        },
        "failed_checks": failed_checks.to_dict(orient="records"),
        "failed_runs": failed_runs.to_dict(orient="records"),
        "leakage_statement": "Training uses only train split batches for optimization. Validation is used for early stopping and checkpoint selection. Test metrics are computed but not used for optimization, preprocessing, feature selection, calibration, or hyperparameter selection. Subject-mixed diagnostic runs remain excluded unless explicitly requested.",
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Wrote selected runs: {selected_path}")
    logger.info(f"Wrote run report: {run_report_path}")
    logger.info(f"Wrote epoch metrics: {epoch_report_path}")
    logger.info(f"Wrote checks: {checks_path}")
    logger.info(f"Wrote summary: {summary_path}")
    logger.info(f"Overall full training stage passed: {overall_passed}")
    logger.info(f"Elapsed seconds: {summary['elapsed_seconds']}")

    print("\nTARGETED OUTPUTS")
    print(f"1. {selected_path}")
    print(f"2. {run_report_path}")
    print(f"3. {epoch_report_path}")
    print(f"4. {checks_path}")
    print(f"5. {summary_path}")
    print(f"6. {out_dir / '12_controlled_full_training_log.txt'}")
    print(f"7. {out_dir / 'runs'}")

    if not overall_passed:
        logger.error("Controlled full training stage failed. Inspect failed runs before continuing.")
        return 1

    logger.info("Controlled full training stage passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
