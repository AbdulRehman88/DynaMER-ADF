from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from dynamer.data.data_modules import DynaMERSplitDataModule


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


def main() -> int:
    parser = argparse.ArgumentParser(description="06_define_model_data_modules: smoke-test reusable DynaMER data modules.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-paths", required=True)
    parser.add_argument("--data-module-config", default="configs/06_define_model_data_modules.yaml")
    args = parser.parse_args()

    t0 = time.time()

    _main_cfg = load_yaml(Path(args.config))
    local_paths = load_yaml(Path(args.local_paths))
    dm_cfg = load_yaml(Path(args.data_module_config))["data_modules"]

    project_root = as_path(local_paths["PROJECT_ROOT"])
    out_dir = project_root / dm_cfg["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(out_dir / "06_define_model_data_modules_log.txt")
    logger.info("Starting 06_define_model_data_modules.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {out_dir}")

    req = dm_cfg["required_previous_steps"]
    if req.get("require_passed", True):
        require_passed_json(project_root, req["loader_summary_json"], logger)

    inputs = dm_cfg["inputs"]
    model_ready_index = pd.read_csv(project_root / inputs["unified_model_ready_index"])
    split_index = pd.read_csv(project_root / inputs["split_index"])

    smoke_rows: List[Dict[str, Any]] = []
    shape_rows: List[Dict[str, Any]] = []

    logger.info(f"Testing {len(split_index)} split files.")

    from tqdm.auto import tqdm

    for _, split_row in tqdm(split_index.iterrows(), total=len(split_index), desc="DataModule smoke tests", unit="split"):
        split_file = Path(str(split_row["split_file"]))
        dataset = str(split_row["dataset"])
        task = str(split_row["task"])
        protocol = str(split_row["protocol"])
        split_id = str(split_row["split_id"])

        modality_keys = dm_cfg["datasets"][dataset]["primary_modalities"]

        status = {
            "split_id": split_id,
            "dataset": dataset,
            "task": task,
            "protocol": protocol,
            "split_file": str(split_file),
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
                batch_size=int(dm_cfg["batch_size"]),
                num_workers=int(dm_cfg["num_workers"]),
                pin_memory=bool(dm_cfg["pin_memory"]),
                modality_keys=modality_keys,
            )
            dm.setup()

            for split_name in ["train", "val", "test"]:
                loader = dm.dataloader(split_name, shuffle=False)
                batch = next(iter(loader))

                y_shape = list(batch["y"].shape)
                modalities = sorted(batch["x"].keys())

                status[f"{split_name}_ok"] = True

                for key in modalities:
                    shape_rows.append(
                        {
                            "split_id": split_id,
                            "dataset": dataset,
                            "task": task,
                            "protocol": protocol,
                            "split": split_name,
                            "modality": key,
                            "x_shape": "x".join(map(str, batch["x"][key].shape)),
                            "mask_shape": "x".join(map(str, batch["masks"][key].shape)),
                            "y_shape": "x".join(map(str, y_shape)),
                            "batch_size_observed": int(batch["y"].shape[0]),
                        }
                    )

            status["passed"] = bool(status["train_ok"] and status["val_ok"] and status["test_ok"])

        except Exception as exc:
            status["error"] = f"{type(exc).__name__}: {exc}"
            status["passed"] = False

        smoke_rows.append(status)

    smoke_df = pd.DataFrame(smoke_rows)
    shape_df = pd.DataFrame(shape_rows)

    checks: List[Dict[str, Any]] = []
    add_check(checks, "split files tested", int(len(smoke_df)), int(len(split_index)))
    add_check(checks, "split files passed", int(smoke_df["passed"].sum()), int(len(split_index)))
    add_check(checks, "batch shape rows generated", int(len(shape_df)), ">0", int(len(shape_df)) > 0)
    add_check(checks, "datasets tested", sorted(smoke_df["dataset"].unique().tolist()), sorted(dm_cfg["datasets"].keys()))

    checks_df = pd.DataFrame(checks)

    smoke_path = out_dir / "06_data_module_smoke_report.csv"
    shape_path = out_dir / "06_batch_shape_report.csv"
    checks_path = out_dir / "06_data_module_checks.csv"
    summary_path = out_dir / "06_data_module_summary.json"

    smoke_df.to_csv(smoke_path, index=False)
    shape_df.to_csv(shape_path, index=False)
    checks_df.to_csv(checks_path, index=False)

    failed_checks = checks_df[checks_df["passed"] == False]
    failed_smoke = smoke_df[smoke_df["passed"] == False]
    overall_passed = len(failed_checks) == 0 and len(failed_smoke) == 0

    summary = {
        "name": dm_cfg["name"],
        "created_at": now(),
        "overall_passed": bool(overall_passed),
        "elapsed_seconds": round(time.time() - t0, 3),
        "row_counts": {
            "split_files_tested": int(len(smoke_df)),
            "split_files_passed": int(smoke_df["passed"].sum()),
            "batch_shape_rows": int(len(shape_df)),
        },
        "outputs": {
            "smoke_report": str(smoke_path),
            "batch_shape_report": str(shape_path),
            "checks": str(checks_path),
            "summary": str(summary_path),
            "log": str(out_dir / "06_define_model_data_modules_log.txt"),
        },
        "failed_checks": failed_checks.to_dict(orient="records"),
        "failed_smoke_tests": failed_smoke.to_dict(orient="records"),
        "leakage_statement": "This stage defines and smoke-tests data modules only. It performs no fitting, scaling, sampling, feature selection, calibration, model training, or result optimization.",
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Wrote smoke report: {smoke_path}")
    logger.info(f"Wrote batch shape report: {shape_path}")
    logger.info(f"Wrote checks: {checks_path}")
    logger.info(f"Wrote summary: {summary_path}")
    logger.info(f"Overall data-module stage passed: {overall_passed}")
    logger.info(f"Elapsed seconds: {summary['elapsed_seconds']}")

    print("\nTARGETED OUTPUTS")
    print(f"1. {smoke_path}")
    print(f"2. {shape_path}")
    print(f"3. {checks_path}")
    print(f"4. {summary_path}")
    print(f"5. {out_dir / '06_define_model_data_modules_log.txt'}")

    if not overall_passed:
        logger.error("Data-module stage failed. Do not proceed to model architecture.")
        return 1

    logger.info("Data-module stage passed. It is safe to proceed to model architecture definition.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

