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

from dynamer.data.data_modules import DynaMERSplitDataModule
from dynamer.models.dynamer_base_model import DynaMERBaseModel


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


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def main() -> int:
    parser = argparse.ArgumentParser(description="07_define_model_architecture: forward-pass smoke test for DynaMER model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-paths", required=True)
    parser.add_argument("--architecture-config", default="configs/07_define_model_architecture.yaml")
    args = parser.parse_args()

    t0 = time.time()

    _main_cfg = load_yaml(Path(args.config))
    local_paths = load_yaml(Path(args.local_paths))
    arch_cfg = load_yaml(Path(args.architecture_config))["architecture"]

    project_root = as_path(local_paths["PROJECT_ROOT"])
    out_dir = project_root / arch_cfg["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(out_dir / "07_define_model_architecture_log.txt")
    logger.info("Starting 07_define_model_architecture.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {out_dir}")

    req = arch_cfg["required_previous_steps"]
    if req.get("require_passed", True):
        require_passed_json(project_root, req["data_module_summary_json"], logger)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    inputs = arch_cfg["inputs"]
    model_ready_index = pd.read_csv(project_root / inputs["unified_model_ready_index"])
    split_index = pd.read_csv(project_root / inputs["split_index"])

    max_split_files = arch_cfg["smoke_test"].get("max_split_files", None)
    if max_split_files is not None:
        split_index = split_index.head(int(max_split_files)).copy()

    model_cfg = arch_cfg["model"]
    test_splits = list(arch_cfg["smoke_test"]["test_splits"])

    smoke_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []

    from tqdm.auto import tqdm

    for _, split_row in tqdm(split_index.iterrows(), total=len(split_index), desc="Architecture forward smoke tests", unit="split"):
        dataset = str(split_row["dataset"])
        task = str(split_row["task"])
        protocol = str(split_row["protocol"])
        split_id = str(split_row["split_id"])
        split_file = Path(str(split_row["split_file"]))

        task_cfg = arch_cfg["tasks"][dataset][task]
        modality_keys = list(task_cfg["modality_keys"])
        num_classes = int(task_cfg["num_classes"])

        row_report = {
            "split_id": split_id,
            "dataset": dataset,
            "task": task,
            "protocol": protocol,
            "split_file": str(split_file),
            "num_classes": num_classes,
            "modalities": "|".join(modality_keys),
            "parameter_count": None,
            "train_ok": False,
            "val_ok": False,
            "test_ok": False,
            "passed": False,
            "error": "",
        }

        try:
            dm = DynaMERSplitDataModule(
                split_file=split_file,
                model_ready_index=model_ready_index,
                project_root=project_root,
                batch_size=int(arch_cfg["smoke_test"]["batch_size"]),
                num_workers=int(arch_cfg["smoke_test"]["num_workers"]),
                pin_memory=False,
                modality_keys=modality_keys,
            )
            dm.setup()

            model = DynaMERBaseModel(
                modality_keys=modality_keys,
                num_classes=num_classes,
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

            model.eval()

            with torch.no_grad():
                for split_name in test_splits:
                    loader = dm.dataloader(split_name, shuffle=False)
                    batch = next(iter(loader))

                    x = {k: v.to(device) for k, v in batch["x"].items()}
                    masks = {k: v.to(device) for k, v in batch["masks"].items()}
                    y = batch["y"].to(device)

                    out = model(x, masks)
                    logits = out["logits"]

                    logits_shape = list(logits.shape)
                    expected_shape = [int(y.shape[0]), num_classes]
                    finite_ok = bool(torch.isfinite(logits).all().item())
                    shape_ok = logits_shape == expected_shape

                    if not shape_ok:
                        raise RuntimeError(f"{split_name} logits shape mismatch: observed={logits_shape}, expected={expected_shape}")
                    if not finite_ok:
                        raise RuntimeError(f"{split_name} logits contain NaN or Inf.")

                    row_report[f"{split_name}_ok"] = True

                    gate_weights = out["gate_weights"].detach().cpu()
                    active_modalities = list(out["active_modalities"])

                    for m_idx, modality in enumerate(active_modalities):
                        gate_rows.append(
                            {
                                "split_id": split_id,
                                "dataset": dataset,
                                "task": task,
                                "protocol": protocol,
                                "split": split_name,
                                "modality": modality,
                                "gate_mean": float(gate_weights[:, m_idx].mean().item()),
                                "gate_min": float(gate_weights[:, m_idx].min().item()),
                                "gate_max": float(gate_weights[:, m_idx].max().item()),
                            }
                        )

            row_report["parameter_count"] = count_parameters(model)
            row_report["passed"] = bool(row_report["train_ok"] and row_report["val_ok"] and row_report["test_ok"])

        except Exception as exc:
            row_report["error"] = f"{type(exc).__name__}: {exc}"
            row_report["passed"] = False

        smoke_rows.append(row_report)

    smoke_df = pd.DataFrame(smoke_rows)
    gate_df = pd.DataFrame(gate_rows)

    checks: List[Dict[str, Any]] = []
    add_check(checks, "split files tested", int(len(smoke_df)), int(len(split_index)))
    add_check(checks, "forward tests passed", int(smoke_df["passed"].sum()), int(len(split_index)))
    add_check(checks, "gate rows generated", int(len(gate_df)), ">0", int(len(gate_df)) > 0)
    add_check(checks, "datasets tested", sorted(smoke_df["dataset"].unique().tolist()), sorted(arch_cfg["tasks"].keys()))
    add_check(checks, "tasks tested", sorted(smoke_df["task"].unique().tolist()), sorted({task for d in arch_cfg["tasks"].values() for task in d.keys()}))

    checks_df = pd.DataFrame(checks)

    smoke_path = out_dir / "07_architecture_forward_smoke_report.csv"
    gate_path = out_dir / "07_fusion_gate_smoke_report.csv"
    checks_path = out_dir / "07_architecture_checks.csv"
    summary_path = out_dir / "07_architecture_summary.json"

    smoke_df.to_csv(smoke_path, index=False)
    gate_df.to_csv(gate_path, index=False)
    checks_df.to_csv(checks_path, index=False)

    failed_checks = checks_df[checks_df["passed"] == False]
    failed_smoke = smoke_df[smoke_df["passed"] == False]

    overall_passed = len(failed_checks) == 0 and len(failed_smoke) == 0

    summary = {
        "name": arch_cfg["name"],
        "created_at": now(),
        "overall_passed": bool(overall_passed),
        "elapsed_seconds": round(time.time() - t0, 3),
        "device": str(device),
        "model_config": model_cfg,
        "row_counts": {
            "split_files_tested": int(len(smoke_df)),
            "forward_tests_passed": int(smoke_df["passed"].sum()),
            "gate_rows": int(len(gate_df)),
        },
        "outputs": {
            "forward_smoke_report": str(smoke_path),
            "fusion_gate_report": str(gate_path),
            "checks": str(checks_path),
            "summary": str(summary_path),
            "log": str(out_dir / "07_define_model_architecture_log.txt"),
        },
        "failed_checks": failed_checks.to_dict(orient="records"),
        "failed_forward_tests": failed_smoke.to_dict(orient="records"),
        "leakage_statement": "This stage only defines the model architecture and runs forward passes on existing split batches. It performs no optimization, no fitting, no scaling, no feature selection, no calibration, and no result-driven model selection.",
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Wrote forward smoke report: {smoke_path}")
    logger.info(f"Wrote fusion gate report: {gate_path}")
    logger.info(f"Wrote checks: {checks_path}")
    logger.info(f"Wrote summary: {summary_path}")
    logger.info(f"Overall architecture stage passed: {overall_passed}")
    logger.info(f"Elapsed seconds: {summary['elapsed_seconds']}")

    print("\nTARGETED OUTPUTS")
    print(f"1. {smoke_path}")
    print(f"2. {gate_path}")
    print(f"3. {checks_path}")
    print(f"4. {summary_path}")
    print(f"5. {out_dir / '07_define_model_architecture_log.txt'}")

    if not overall_passed:
        logger.error("Architecture stage failed. Do not proceed to training-loop design.")
        return 1

    logger.info("Architecture stage passed. It is safe to proceed to training-loop design.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

