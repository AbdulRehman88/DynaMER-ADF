from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
import yaml

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from dynamer.data.temporal_data_modules import DynaMERTemporalSplitDataModule
from dynamer.models.dynamer_base_model import DynaMERBaseModel
from dynamer.training.smoke_engine import (
    count_parameters,
    evaluate_smoke,
    set_global_seed,
    train_one_epoch_smoke,
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


def add_check(checks: List[Dict[str, Any]], check: str, observed: Any, expected: Any, passed: bool | None = None) -> None:
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


def select_representative_runs(split_index: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    selected_rows = []
    prefer_protocols = list(cfg["run_selection"]["prefer_protocols"])

    for item in cfg["run_selection"]["required_dataset_tasks"]:
        dataset = item["dataset"]
        task = item["task"]

        sub = split_index[(split_index["dataset"] == dataset) & (split_index["task"] == task)].copy()
        if sub.empty:
            raise RuntimeError(f"No split available for dataset={dataset}, task={task}")

        chosen = None
        for protocol in prefer_protocols:
            psub = sub[sub["protocol"] == protocol].sort_values("fold_index")
            if not psub.empty:
                chosen = psub.iloc[0]
                break

        if chosen is None:
            chosen = sub.sort_values(["protocol", "fold_index"]).iloc[0]

        selected_rows.append(chosen.to_dict())

    selected = pd.DataFrame(selected_rows)
    max_runs = cfg["run_selection"].get("max_runs", None)
    if max_runs is not None:
        selected = selected.head(int(max_runs)).copy()

    return selected


def make_model(task_cfg: Dict[str, Any], model_cfg: Dict[str, Any], device: torch.device) -> DynaMERBaseModel:
    return DynaMERBaseModel(
        modality_keys=list(task_cfg["modality_keys"]),
        num_classes=int(task_cfg["num_classes"]),
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


def main() -> int:
    parser = argparse.ArgumentParser(description="10_training_loop_smoke_test: tiny representative training-loop smoke test.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-paths", required=True)
    parser.add_argument("--training-smoke-config", default="configs/10_training_loop_smoke_test.yaml")
    args = parser.parse_args()

    t0 = time.time()

    _main_cfg = load_yaml(Path(args.config))
    local_paths = load_yaml(Path(args.local_paths))
    smoke_cfg = load_yaml(Path(args.training_smoke_config))["training_smoke"]

    project_root = as_path(local_paths["PROJECT_ROOT"])
    out_dir = project_root / smoke_cfg["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(out_dir / "10_training_loop_smoke_test_log.txt")
    logger.info("Starting 10_training_loop_smoke_test.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {out_dir}")

    req = smoke_cfg["required_previous_steps"]
    if req.get("require_passed", True):
        require_passed_json(project_root, req["temporal_data_module_summary_json"], logger)
        require_passed_json(project_root, req["architecture_summary_json"], logger)

    seed = int(smoke_cfg["random_seed"])
    set_global_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    temporal_view_index = pd.read_csv(project_root / smoke_cfg["inputs"]["temporal_view_index"])
    split_index = pd.read_csv(project_root / smoke_cfg["inputs"]["split_index"])
    selected_runs = select_representative_runs(split_index, smoke_cfg)

    selected_path = out_dir / "10_selected_smoke_runs.csv"
    selected_runs.to_csv(selected_path, index=False)

    logger.info(f"Selected {len(selected_runs)} representative smoke runs.")
    logger.info(f"Wrote selected runs: {selected_path}")

    run_rows: List[Dict[str, Any]] = []
    epoch_rows: List[Dict[str, Any]] = []

    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    from tqdm.auto import tqdm

    for run_idx, split_row in enumerate(tqdm(selected_runs.to_dict(orient="records"), desc="Training smoke runs", unit="run"), start=1):
        dataset = str(split_row["dataset"])
        task = str(split_row["task"])
        protocol = str(split_row["protocol"])
        split_id = str(split_row["split_id"])
        split_file = Path(str(split_row["split_file"]))

        task_cfg = smoke_cfg["tasks"][dataset][task]
        training_cfg = smoke_cfg["training"]
        loader_cfg = smoke_cfg["loader"]

        run_status = {
            "run_index": run_idx,
            "split_id": split_id,
            "dataset": dataset,
            "task": task,
            "protocol": protocol,
            "split_file": str(split_file),
            "num_classes": int(task_cfg["num_classes"]),
            "modalities": "|".join(task_cfg["modality_keys"]),
            "parameter_count": None,
            "checkpoint_file": "",
            "passed": False,
            "error": "",
        }

        try:
            dm = DynaMERTemporalSplitDataModule(
                split_file=split_file,
                temporal_view_index=temporal_view_index,
                project_root=project_root,
                modality_keys=list(task_cfg["modality_keys"]),
                label_column=str(task_cfg["label_column"]),
                batch_size=int(loader_cfg["batch_size"]),
                num_workers=int(loader_cfg["num_workers"]),
                pin_memory=bool(loader_cfg["pin_memory"]),
            )
            dm.setup()

            train_loader = dm.dataloader("train", shuffle=True)
            val_loader = dm.dataloader("val", shuffle=False)
            test_loader = dm.dataloader("test", shuffle=False)

            model = make_model(task_cfg, smoke_cfg["model"], device)
            criterion = torch.nn.CrossEntropyLoss()

            # Warm-up forward pass initializes LazyLinear parameters before optimizer creation.
            first_batch = next(iter(train_loader))
            with torch.no_grad():
                warm_x = {k: v.to(device) for k, v in first_batch["x"].items()}
                warm_masks = {k: v.to(device) for k, v in first_batch["masks"].items()}
                warm_out = model(warm_x, warm_masks)
                if not torch.isfinite(warm_out["logits"]).all():
                    raise RuntimeError("Warm-up logits contain NaN or Inf.")

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(training_cfg["learning_rate"]),
                weight_decay=float(training_cfg["weight_decay"]),
            )

            run_status["parameter_count"] = count_parameters(model)

            for epoch in range(1, int(training_cfg["epochs"]) + 1):
                train_metrics = train_one_epoch_smoke(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    max_batches=int(training_cfg["max_train_batches_per_epoch"]),
                    gradient_clip_norm=float(training_cfg["gradient_clip_norm"]),
                )

                val_metrics = evaluate_smoke(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    device=device,
                    max_batches=int(training_cfg["max_eval_batches"]),
                )

                test_metrics = evaluate_smoke(
                    model=model,
                    loader=test_loader,
                    criterion=criterion,
                    device=device,
                    max_batches=int(training_cfg["max_eval_batches"]),
                )

                epoch_rows.append(
                    {
                        "run_index": run_idx,
                        "split_id": split_id,
                        "dataset": dataset,
                        "task": task,
                        "protocol": protocol,
                        "epoch": epoch,
                        "train_loss": train_metrics["loss"],
                        "train_accuracy": train_metrics["accuracy"],
                        "train_balanced_accuracy": train_metrics["balanced_accuracy"],
                        "train_macro_f1": train_metrics["macro_f1"],
                        "train_batches": train_metrics["batches"],
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"],
                        "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                        "val_macro_f1": val_metrics["macro_f1"],
                        "val_batches": val_metrics["batches"],
                        "test_loss": test_metrics["loss"],
                        "test_accuracy": test_metrics["accuracy"],
                        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                        "test_macro_f1": test_metrics["macro_f1"],
                        "test_batches": test_metrics["batches"],
                    }
                )

            ckpt_path = checkpoint_dir / f"10_smoke_run_{run_idx:02d}_{dataset.replace('-', '_')}_{task}_{protocol}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": smoke_cfg["model"],
                    "dataset": dataset,
                    "task": task,
                    "protocol": protocol,
                    "split_id": split_id,
                    "task_config": task_cfg,
                    "smoke_only": True,
                    "created_at": now(),
                },
                ckpt_path,
            )

            run_status["checkpoint_file"] = str(ckpt_path)
            run_status["passed"] = bool(ckpt_path.exists())

        except Exception as exc:
            run_status["error"] = f"{type(exc).__name__}: {exc}"
            run_status["passed"] = False

        run_rows.append(run_status)

    run_df = pd.DataFrame(run_rows)
    epoch_df = pd.DataFrame(epoch_rows)

    checks: List[Dict[str, Any]] = []
    add_check(checks, "selected smoke runs", int(len(selected_runs)), int(len(smoke_cfg["run_selection"]["required_dataset_tasks"])))
    add_check(checks, "runs passed", int(run_df["passed"].sum()), int(len(selected_runs)))
    add_check(checks, "epoch rows generated", int(len(epoch_df)), int(len(selected_runs) * int(smoke_cfg["training"]["epochs"])))
    add_check(checks, "checkpoints generated", int((run_df["checkpoint_file"].astype(str).str.len() > 0).sum()), int(len(selected_runs)))
    add_check(checks, "datasets tested", sorted(run_df["dataset"].unique().tolist()), sorted(smoke_cfg["tasks"].keys()))
    add_check(checks, "tasks tested", sorted(run_df["task"].unique().tolist()), sorted({task for d in smoke_cfg["tasks"].values() for task in d.keys()}))

    checks_df = pd.DataFrame(checks)

    run_report_path = out_dir / "10_training_smoke_run_report.csv"
    epoch_report_path = out_dir / "10_training_smoke_epoch_metrics.csv"
    checks_path = out_dir / "10_training_smoke_checks.csv"
    summary_path = out_dir / "10_training_smoke_summary.json"

    run_df.to_csv(run_report_path, index=False)
    epoch_df.to_csv(epoch_report_path, index=False)
    checks_df.to_csv(checks_path, index=False)

    failed_checks = checks_df[checks_df["passed"] == False]
    failed_runs = run_df[run_df["passed"] == False]
    overall_passed = len(failed_checks) == 0 and len(failed_runs) == 0

    summary = {
        "name": smoke_cfg["name"],
        "created_at": now(),
        "overall_passed": bool(overall_passed),
        "elapsed_seconds": round(time.time() - t0, 3),
        "device": str(device),
        "row_counts": {
            "selected_runs": int(len(selected_runs)),
            "runs_passed": int(run_df["passed"].sum()),
            "epoch_rows": int(len(epoch_df)),
        },
        "outputs": {
            "selected_runs": str(selected_path),
            "run_report": str(run_report_path),
            "epoch_metrics": str(epoch_report_path),
            "checks": str(checks_path),
            "summary": str(summary_path),
            "log": str(out_dir / "10_training_loop_smoke_test_log.txt"),
            "checkpoint_dir": str(checkpoint_dir),
        },
        "failed_checks": failed_checks.to_dict(orient="records"),
        "failed_runs": failed_runs.to_dict(orient="records"),
        "leakage_statement": "This smoke stage trains only on training batches and evaluates on validation/test batches without fitting preprocessing, scaling, feature selection, calibration, balancing, or hyperparameter selection. Metrics are for engineering verification only and must not be reported as scientific results.",
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Wrote selected runs: {selected_path}")
    logger.info(f"Wrote run report: {run_report_path}")
    logger.info(f"Wrote epoch metrics: {epoch_report_path}")
    logger.info(f"Wrote checks: {checks_path}")
    logger.info(f"Wrote summary: {summary_path}")
    logger.info(f"Overall training smoke stage passed: {overall_passed}")
    logger.info(f"Elapsed seconds: {summary['elapsed_seconds']}")

    print("\nTARGETED OUTPUTS")
    print(f"1. {selected_path}")
    print(f"2. {run_report_path}")
    print(f"3. {epoch_report_path}")
    print(f"4. {checks_path}")
    print(f"5. {summary_path}")
    print(f"6. {out_dir / '10_training_loop_smoke_test_log.txt'}")
    print(f"7. {checkpoint_dir}")

    if not overall_passed:
        logger.error("Training-loop smoke stage failed. Do not proceed to full experiment design.")
        return 1

    logger.info("Training-loop smoke stage passed. It is safe to proceed to full experiment design.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
