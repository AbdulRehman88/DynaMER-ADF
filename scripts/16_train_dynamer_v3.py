
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
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
from dynamer.models.dynamer_v3_model import DynaMERv3Model
from dynamer.training.full_engine import (
    EarlyStopper,
    count_parameters,
    make_class_weights,
    run_eval_epoch,
    run_train_epoch,
    set_global_seed,
)


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


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
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def as_path(x: str) -> Path:
    return Path(str(x).replace("\\", "/"))


def require_passed_json(project_root: Path, rel_path: str, logger: Logger) -> None:
    path = project_root / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Required previous-stage summary not found: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not bool(obj.get("overall_passed", False)):
        raise RuntimeError(f"Required previous stage did not pass: {path}")
    logger.info(f"Verified previous stage passed: {path}")


def add_check(checks: List[Dict[str, Any]], name: str, observed: Any, expected: Any, passed: Optional[bool] = None) -> None:
    if passed is None:
        passed = observed == expected
    checks.append({
        "check": name,
        "observed": observed,
        "expected": expected,
        "passed": bool(passed),
    })


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


def make_model(run: Dict[str, Any], baseline_variant: str, model_cfg: Dict[str, Any], device: torch.device) -> DynaMERv3Model:
    return DynaMERv3Model(
        modality_keys=str(run["modality_keys"]).split("|"),
        num_classes=int(run["num_classes"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        temporal_backbone_v1=str(model_cfg.get("temporal_backbone_v1", "bigru")),
        tcn_layers=int(model_cfg.get("tcn_layers", 2)),
        modality_dropout=float(model_cfg.get("modality_dropout", 0.00)),
        spike_steps=int(model_cfg.get("spike_steps", 6)),
        spike_decay=float(model_cfg.get("spike_decay", 0.85)),
        spike_threshold=float(model_cfg.get("spike_threshold", 1.0)),
        spike_slope=float(model_cfg.get("spike_slope", 5.0)),
        spike_mix=float(model_cfg.get("spike_mix", 0.10)),
    ).to(device)


def sanitize(x: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(x))


def run_single_training(
    run: Dict[str, Any],
    baseline_variant: str,
    temporal_index: pd.DataFrame,
    project_root: Path,
    out_dir: Path,
    cfg: Dict[str, Any],
    device: torch.device,
    logger: Logger,
) -> Dict[str, Any]:

    baseline_run_id = f"{baseline_variant}__{run['run_id']}"
    run_dir = out_dir / "runs" / sanitize(baseline_variant) / sanitize(str(run["run_id"]))

    if run_dir.exists() and not bool(cfg["execution"]["overwrite_existing"]):
        return {
            "baseline_variant": baseline_variant,
            "baseline_run_id": baseline_run_id,
            "source_run_id": run["run_id"],
            "dataset": run["dataset"],
            "task": run["task"],
            "protocol": run["protocol"],
            "fold_index": int(run["fold_index"]),
            "status": "skipped_existing",
            "run_dir": str(run_dir),
            "best_checkpoint": str(run_dir / "best_model.pt"),
            "last_checkpoint": str(run_dir / "last_model.pt"),
            "best_epoch": None,
            "best_monitor_value": None,
            "epochs_completed": 0,
            "parameter_count": None,
            "error": "",
        }

    if run_dir.exists():
        for child in run_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)
    run_dir.mkdir(parents=True, exist_ok=True)

    training_cfg = cfg["training"]
    loader_cfg = cfg["loader"]

    dm = DynaMERTemporalSplitDataModule(
        project_root=project_root,
        split_file=Path(str(run["split_file"])),
        temporal_view_index=temporal_index,
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

    model = make_model(run, baseline_variant, cfg["model"], device)

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

    scaler = torch.amp.GradScaler("cuda", enabled=bool(training_cfg["use_amp"] and device.type == "cuda"))

    monitor_metric = str(training_cfg["monitor_metric"])
    monitor_mode = str(training_cfg["monitor_mode"])
    stopper = EarlyStopper(
        mode=monitor_mode,
        patience=int(training_cfg["patience"]),
        min_delta=float(training_cfg["min_delta"]),
    )

    epoch_rows = []
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
        )

        test_metrics, _ = run_eval_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
        )

        score = val_metrics.get(monitor_metric.replace("val_", ""), float("nan"))
        if monitor_metric.startswith("val_"):
            score = val_metrics.get(monitor_metric[4:], float("nan"))

        row = {
            "baseline_variant": baseline_variant,
            "baseline_run_id": baseline_run_id,
            "source_run_id": run["run_id"],
            "dataset": run["dataset"],
            "task": run["task"],
            "protocol": run["protocol"],
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

            "monitor_metric": monitor_metric,
            "monitor_value": score,
        }

        improved = stopper.improved(score)
        row["improved"] = bool(improved)
        row["early_stop_bad_epochs"] = int(stopper.bad_epochs)
        epoch_rows.append(row)

        if improved:
            best_epoch = epoch
            best_monitor = score

            if bool(cfg["execution"]["save_best_checkpoint"]):
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "baseline_variant": baseline_variant,
                        "model_config": cfg["model"],
                        "run": run,
                        "epoch": epoch,
                        "monitor_metric": monitor_metric,
                        "monitor_value": score,
                    },
                    best_ckpt,
                )

        if stopper.should_stop():
            break

    if bool(cfg["execution"]["save_last_checkpoint"]):
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "baseline_variant": baseline_variant,
                "model_config": cfg["model"],
                "run": run,
                "epoch": int(epoch_rows[-1]["epoch"]),
            },
            last_ckpt,
        )

    epoch_df = pd.DataFrame(epoch_rows)
    epoch_path = run_dir / "epoch_metrics.csv"
    epoch_df.to_csv(epoch_path, index=False)

    # Save predictions from the selected best checkpoint.
    prediction_path = ""
    if bool(cfg["execution"]["save_predictions"]):
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
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

        pred_df = pd.DataFrame(
            [{**r, "split": "val"} for r in val_preds] +
            [{**r, "split": "test"} for r in test_preds]
        )
        pred_path = run_dir / "predictions_best_epoch.csv"
        pred_df.to_csv(pred_path, index=False)
        prediction_path = str(pred_path)

    return {
        "baseline_variant": baseline_variant,
        "baseline_run_id": baseline_run_id,
        "source_run_id": run["run_id"],
        "dataset": run["dataset"],
        "task": run["task"],
        "protocol": run["protocol"],
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
    parser = argparse.ArgumentParser(description="16_train_dynamer_v3: DynaMER-v3 training using Stage 12 data/splits/engine.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-paths", required=True)
    parser.add_argument("--baseline-config", default="configs/16_train_dynamer_v3.yaml")
    parser.add_argument("--phase", default=None, help="Override phase_filter. Use all for all registry rows.")
    parser.add_argument("--baseline", default="all", help="Baseline variant name or all.")
    parser.add_argument("--max-runs", type=int, default=None, help="Optional max registry rows before expanding baselines.")
    parser.add_argument("--include-diagnostic", action="store_true", help="Include subject-mixed diagnostic runs.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    _main_cfg = load_yaml(Path(args.config))
    local_paths = load_yaml(Path(args.local_paths))
    cfg = load_yaml(Path(args.baseline_config))["baseline_training"]

    project_root = as_path(local_paths["PROJECT_ROOT"])
    out_dir = project_root / cfg["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(out_dir / "16_train_dynamer_v3_log.txt")
    logger.info("Starting 16_train_dynamer_v3.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {out_dir}")

    req = cfg["required_previous_steps"]
    if req.get("require_passed", True):
        require_passed_json(project_root, req["registry_summary_json"], logger)
        require_passed_json(project_root, req["temporal_data_module_summary_json"], logger)

    if args.overwrite:
        cfg["execution"]["overwrite_existing"] = True
    if args.include_diagnostic:
        cfg["execution"]["include_diagnostic_upper_bound"] = True

    set_global_seed(int(cfg["random_seed"]))

    device = torch.device("cuda" if torch.cuda.is_available() and bool(cfg["execution"]["use_cuda_if_available"]) else "cpu")
    logger.info(f"Using device: {device}")

    registry = pd.read_csv(project_root / cfg["inputs"]["experiment_registry"])
    temporal_index = pd.read_csv(project_root / cfg["inputs"]["temporal_view_index"])

    config_max_runs = cfg["execution"].get("max_runs", None)
    max_runs = args.max_runs if args.max_runs is not None else config_max_runs

    selected = select_registry_runs(
        registry=registry,
        config_phase=cfg["execution"].get("phase_filter", None),
        cli_phase=args.phase,
        include_diagnostic=bool(cfg["execution"]["include_diagnostic_upper_bound"]),
        max_runs=max_runs,
    )

    all_variants = [str(v).lower() for v in cfg["baselines"]]
    if str(args.baseline).lower() == "all":
        baseline_variants = all_variants
    else:
        baseline_variants = [str(args.baseline).lower()]
        unknown = sorted(set(baseline_variants) - set(all_variants))
        if unknown:
            raise ValueError(f"Unknown baseline(s): {unknown}. Allowed: {all_variants}")

    selected_path = out_dir / "14_selected_registry_runs.csv"
    selected.to_csv(selected_path, index=False)

    plan_rows = []
    for run in selected.to_dict(orient="records"):
        for baseline_variant in baseline_variants:
            plan_rows.append({
                "baseline_variant": baseline_variant,
                "source_run_id": run["run_id"],
                "dataset": run["dataset"],
                "task": run["task"],
                "protocol": run["protocol"],
                "fold_index": int(run["fold_index"]),
            })

    plan_df = pd.DataFrame(plan_rows)
    plan_df.to_csv(out_dir / "16_dynamer_v3_run_plan.csv", index=False)

    logger.info(f"Selected registry rows: {len(selected)}")
    logger.info(f"DynaMER-v3 variants: {baseline_variants}")
    logger.info(f"Total DynaMER-v3 training runs: {len(plan_df)}")

    run_results = []
    all_epoch_frames = []

    from tqdm.auto import tqdm

    total = len(selected) * len(baseline_variants)
    pbar = tqdm(total=total, desc="DynaMER-v3 training", unit="run")

    for run in selected.to_dict(orient="records"):
        for baseline_variant in baseline_variants:
            try:
                result = run_single_training(
                    run=run,
                    baseline_variant=baseline_variant,
                    temporal_index=temporal_index,
                    project_root=project_root,
                    out_dir=out_dir,
                    cfg=cfg,
                    device=device,
                    logger=logger,
                )
            except Exception as e:
                logger.error(f"DynaMER-v3 run failed: baseline={baseline_variant} run={run.get('run_id')} error={repr(e)}")
                result = {
                    "baseline_variant": baseline_variant,
                    "baseline_run_id": f"{baseline_variant}__{run.get('run_id')}",
                    "source_run_id": run.get("run_id"),
                    "dataset": run.get("dataset"),
                    "task": run.get("task"),
                    "protocol": run.get("protocol"),
                    "fold_index": int(run.get("fold_index", -1)),
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

    checks = []
    add_check(checks, "selected registry runs", int(len(selected)), ">0", int(len(selected)) > 0)
    add_check(checks, "DynaMER-v3 variants", baseline_variants, baseline_variants, len(baseline_variants) > 0)
    add_check(checks, "planned baseline runs", int(len(plan_df)), int(len(selected)) * int(len(baseline_variants)))
    add_check(checks, "runs completed or skipped", int((run_report["status"].isin(["completed", "skipped_existing"])).sum()), int(len(plan_df)))
    add_check(checks, "failed runs", int((run_report["status"] == "failed").sum()), 0)
    add_check(checks, "epoch metric rows generated", int(len(all_epochs)), ">0", int(len(all_epochs)) > 0 or int(len(plan_df)) == 0)
    add_check(
        checks,
        "best checkpoints present for completed runs",
        int(sum(Path(p).exists() for p in run_report.loc[run_report["status"] == "completed", "best_checkpoint"].astype(str) if p)),
        int((run_report["status"] == "completed").sum()),
    )
    add_check(checks, "datasets trained", sorted(run_report["dataset"].dropna().unique().tolist()), sorted(selected["dataset"].dropna().unique().tolist()))

    checks_df = pd.DataFrame(checks)

    run_report_path = out_dir / "16_dynamer_v3_training_run_report.csv"
    epoch_report_path = out_dir / "16_dynamer_v3_all_epoch_metrics.csv"
    checks_path = out_dir / "16_dynamer_v3_training_checks.csv"
    summary_path = out_dir / "16_dynamer_v3_training_summary.json"

    run_report.to_csv(run_report_path, index=False)
    all_epochs.to_csv(epoch_report_path, index=False)
    checks_df.to_csv(checks_path, index=False)

    failed_checks = checks_df[~checks_df["passed"]]
    failed_runs = run_report[run_report["status"] == "failed"]
    overall_passed = len(failed_checks) == 0 and len(failed_runs) == 0

    summary = {
        "name": cfg["name"],
        "created_at": now(),
        "overall_passed": bool(overall_passed),
        "device": str(device),
        "selected_phase": args.phase if args.phase else cfg["execution"].get("phase_filter", None),
        "baseline_variants": baseline_variants,
        "row_counts": {
            "selected_registry_runs": int(len(selected)),
            "planned_baseline_runs": int(len(plan_df)),
            "run_report_rows": int(len(run_report)),
            "epoch_metric_rows": int(len(all_epochs)),
        },
        "outputs": {
            "selected_registry_runs": str(selected_path),
            "run_plan": str(out_dir / "16_dynamer_v3_run_plan.csv"),
            "run_report": str(run_report_path),
            "epoch_metrics": str(epoch_report_path),
            "checks": str(checks_path),
            "summary": str(summary_path),
            "log": str(out_dir / "16_train_dynamer_v3_log.txt"),
            "runs_dir": str(out_dir / "runs"),
        },
        "failed_checks": failed_checks.to_dict(orient="records"),
        "failed_runs": failed_runs.to_dict(orient="records"),
        "leakage_statement": "Baselines use the same frozen registry, split files, temporal views, train-only normalization, class weighting, validation-based early stopping, and untouched test evaluation as DynaMER.",
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info(f"Wrote run report: {run_report_path}")
    logger.info(f"Wrote epoch metrics: {epoch_report_path}")
    logger.info(f"Wrote checks: {checks_path}")
    logger.info(f"Wrote summary: {summary_path}")
    logger.info(f"Overall DynaMER-v3 training stage passed: {overall_passed}")

    print("\nStage 14 DynaMER-v3 training outputs:")
    print(f"1. {selected_path}")
    print(f"2. {out_dir / '16_dynamer_v3_run_plan.csv'}")
    print(f"3. {run_report_path}")
    print(f"4. {epoch_report_path}")
    print(f"5. {checks_path}")
    print(f"6. {summary_path}")
    print(f"7. {out_dir / '16_train_dynamer_v3_log.txt'}")
    print(f"8. {out_dir / 'runs'}")

    if not overall_passed:
        logger.error("DynaMER-v3 training stage failed. Inspect failed runs before continuing.")
        return 1

    logger.info("DynaMER-v3 training stage passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
