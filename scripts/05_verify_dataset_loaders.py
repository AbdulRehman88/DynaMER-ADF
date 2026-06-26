from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x

import sys
PROJECT_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT_FOR_IMPORT))

from dynamer.data.trial_dataset import DynaMERTrialDataset


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

    def warn(self, msg: str) -> None:
        line = f"[{now()}] [WARN] {msg}"
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


def shape_text(arr: np.ndarray) -> str:
    return "x".join(str(v) for v in arr.shape)


def parse_shape_text(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return None
    return text


def scalar_int_from_npz(npz: Any, key: str) -> Optional[int]:
    if key not in npz.files:
        return None
    arr = np.asarray(npz[key])
    if arr.size != 1:
        return None
    return int(arr.reshape(-1)[0])


def json_dump(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False)


def add_check(
    checks: List[Dict[str, Any]],
    check: str,
    observed: Any,
    expected: Any,
    passed: Optional[bool] = None,
) -> None:
    if passed is None:
        passed = observed == expected
    checks.append(
        {
            "check": check,
            "observed": json_dump(observed),
            "expected": json_dump(expected),
            "passed": bool(passed),
        }
    )


def verify_trial_files(
    project_root: Path,
    index_df: pd.DataFrame,
    reqs: Dict[str, Any],
    logger: Logger,
) -> pd.DataFrame:
    logger.info("Verifying every prepared trial-store NPZ file.")

    reports: List[Dict[str, Any]] = []

    for _, row in tqdm(index_df.iterrows(), total=len(index_df), desc="Verifying NPZ trial files", unit="trial"):
        dataset = str(row["dataset"])
        trial_uid = str(row["trial_uid"])
        path = project_root / str(row["trial_store_file"])

        dataset_req = reqs[dataset]
        required_keys = dataset_req["required_npz_keys"]
        shape_columns = dataset_req.get("shape_columns", {})
        task_label_map = dataset_req.get("task_label_map", {})

        file_exists = path.exists()
        open_ok = False
        missing_keys: List[str] = []
        shape_mismatches: List[str] = []
        label_mismatches: List[str] = []
        npz_keys: List[str] = []

        if file_exists:
            try:
                with np.load(path, allow_pickle=False) as npz:
                    open_ok = True
                    npz_keys = list(npz.files)
                    missing_keys = [k for k in required_keys if k not in npz.files]

                    for key, shape_col in shape_columns.items():
                        expected_shape = parse_shape_text(row.get(shape_col, None))
                        if key in npz.files and expected_shape is not None:
                            observed_shape = shape_text(np.asarray(npz[key]))
                            if observed_shape != expected_shape:
                                shape_mismatches.append(f"{key}: observed={observed_shape}, expected={expected_shape}")

                    for task_name, index_label_col in task_label_map.items():
                        if index_label_col in row.index and index_label_col in npz.files and not pd.isna(row[index_label_col]):
                            npz_label = scalar_int_from_npz(npz, index_label_col)
                            index_label = int(row[index_label_col])
                            if npz_label != index_label:
                                label_mismatches.append(
                                    f"{index_label_col}: npz={npz_label}, index={index_label}"
                                )
            except Exception as exc:
                open_ok = False
                missing_keys = [f"OPEN_ERROR: {type(exc).__name__}: {exc}"]

        reports.append(
            {
                "dataset": dataset,
                "trial_uid": trial_uid,
                "trial_store_file": str(row["trial_store_file"]),
                "file_exists": bool(file_exists),
                "open_ok": bool(open_ok),
                "npz_keys": "|".join(npz_keys),
                "missing_required_keys": "|".join(missing_keys),
                "shape_mismatches": "|".join(shape_mismatches),
                "label_mismatches": "|".join(label_mismatches),
                "passed": bool(file_exists and open_ok and not missing_keys and not shape_mismatches and not label_mismatches),
            }
        )

    return pd.DataFrame(reports)


def verify_split_alignment(
    prepared_index: pd.DataFrame,
    all_splits: pd.DataFrame,
    reqs: Dict[str, Any],
    logger: Logger,
) -> pd.DataFrame:
    logger.info("Verifying split assignments against prepared model-ready index.")

    base_cols = ["dataset", "trial_uid", "trial_store_file"]
    label_cols = set()
    for dataset_req in reqs.values():
        for index_col in dataset_req.get("task_label_map", {}).values():
            label_cols.add(index_col)

    available_label_cols = [c for c in sorted(label_cols) if c in prepared_index.columns]
    prepared_small = prepared_index[base_cols + available_label_cols].copy()

    merged = all_splits.merge(
        prepared_small,
        on=["dataset", "trial_uid"],
        how="left",
        suffixes=("_split", "_prepared"),
    )

    rows: List[Dict[str, Any]] = []

    for split_id, group in tqdm(merged.groupby("split_id", sort=True), desc="Verifying split groups", unit="split"):
        dataset = str(group["dataset"].iloc[0])
        task = str(group["task"].iloc[0])
        protocol = str(group["protocol"].iloc[0])

        dataset_req = reqs[dataset]
        task_label_map = dataset_req.get("task_label_map", {})
        index_label_col = task_label_map.get(task)

        missing_prepared = int(group["trial_store_file"].isna().sum())
        missing_file_rows = int((group["trial_store_file"].astype(str).str.len() == 0).sum())

        label_mismatch_count = 0
        if index_label_col is None:
            label_mismatch_count = len(group)
        elif index_label_col not in group.columns:
            label_mismatch_count = len(group)
        else:
            split_labels = pd.to_numeric(group["label"], errors="coerce").astype("Int64")
            prepared_labels = pd.to_numeric(group[index_label_col], errors="coerce").astype("Int64")
            label_mismatch_count = int((split_labels != prepared_labels).sum())

        rows.append(
            {
                "split_id": split_id,
                "dataset": dataset,
                "task": task,
                "protocol": protocol,
                "rows": int(len(group)),
                "missing_prepared_trials": missing_prepared,
                "missing_trial_store_paths": missing_file_rows,
                "label_mismatch_count": label_mismatch_count,
                "passed": bool(missing_prepared == 0 and missing_file_rows == 0 and label_mismatch_count == 0),
            }
        )

    return pd.DataFrame(rows)


def verify_dataset_class(
    project_root: Path,
    prepared_index: pd.DataFrame,
    reqs: Dict[str, Any],
    logger: Logger,
) -> pd.DataFrame:
    logger.info("Verifying reusable DynaMERTrialDataset class on each dataset-task pair.")

    rows: List[Dict[str, Any]] = []

    for dataset in sorted(prepared_index["dataset"].unique()):
        dataset_df = prepared_index[prepared_index["dataset"] == dataset].copy()
        dataset_req = reqs[dataset]
        required_keys = dataset_req["required_npz_keys"]

        for task, label_col in dataset_req.get("task_label_map", {}).items():
            if label_col not in dataset_df.columns:
                rows.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "label_column": label_col,
                        "n_items": 0,
                        "first_item_loaded": False,
                        "last_item_loaded": False,
                        "required_keys_present_first": False,
                        "required_keys_present_last": False,
                        "passed": False,
                        "error": f"Missing label column: {label_col}",
                    }
                )
                continue

            try:
                ds = DynaMERTrialDataset(
                    index_df=dataset_df,
                    project_root=project_root,
                    dataset_name=dataset,
                    task=task,
                    label_column=label_col,
                    required_keys=required_keys,
                )

                first = ds[0]
                last = ds[len(ds) - 1]

                first_keys_ok = set(required_keys).issubset(set(first["arrays"].keys()))
                last_keys_ok = set(required_keys).issubset(set(last["arrays"].keys()))

                rows.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "label_column": label_col,
                        "n_items": len(ds),
                        "first_item_loaded": True,
                        "last_item_loaded": True,
                        "required_keys_present_first": bool(first_keys_ok),
                        "required_keys_present_last": bool(last_keys_ok),
                        "passed": bool(len(ds) > 0 and first_keys_ok and last_keys_ok),
                        "error": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "dataset": dataset,
                        "task": task,
                        "label_column": label_col,
                        "n_items": int(len(dataset_df)),
                        "first_item_loaded": False,
                        "last_item_loaded": False,
                        "required_keys_present_first": False,
                        "required_keys_present_last": False,
                        "passed": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="05_verify_dataset_loaders: verifies model-ready NPZ files and split alignment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-paths", required=True)
    parser.add_argument("--loader-config", default="configs/05_verify_dataset_loaders.yaml")
    args = parser.parse_args()

    t0 = time.time()

    _main_cfg = load_yaml(Path(args.config))
    local_paths = load_yaml(Path(args.local_paths))
    loader_cfg = load_yaml(Path(args.loader_config))["loader_verify"]

    project_root = as_path(local_paths["PROJECT_ROOT"])
    out_dir = project_root / loader_cfg["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(out_dir / "05_verify_dataset_loaders_log.txt")
    logger.info("Starting 05_verify_dataset_loaders.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {out_dir}")

    req_prev = loader_cfg["required_previous_steps"]
    if req_prev.get("require_passed", True):
        require_passed_json(project_root, req_prev["model_ready_summary_json"], logger)
        require_passed_json(project_root, req_prev["split_summary_json"], logger)

    input_cfg = loader_cfg["inputs"]
    prepared_index = pd.read_csv(project_root / input_cfg["unified_model_ready_index"])
    all_splits = pd.read_csv(project_root / input_cfg["all_split_assignments"])
    split_index = pd.read_csv(project_root / input_cfg["split_index"])

    reqs = loader_cfg["dataset_requirements"]

    trial_report = verify_trial_files(project_root, prepared_index, reqs, logger)
    split_report = verify_split_alignment(prepared_index, all_splits, reqs, logger)
    dataset_class_report = verify_dataset_class(project_root, prepared_index, reqs, logger)

    checks: List[Dict[str, Any]] = []

    add_check(checks, "prepared row count", int(len(prepared_index)), int(loader_cfg["expected"]["total_prepared_trials"]))
    add_check(checks, "prepared datasets", sorted(prepared_index["dataset"].unique().tolist()), loader_cfg["expected"]["total_datasets"])
    add_check(checks, "trial files all passed", int(trial_report["passed"].sum()), int(len(trial_report)))
    add_check(checks, "split groups all passed", int(split_report["passed"].sum()), int(len(split_report)))
    add_check(checks, "dataset class checks all passed", int(dataset_class_report["passed"].sum()), int(len(dataset_class_report)))
    add_check(checks, "split index rows match split groups", int(len(split_index)), int(split_report["split_id"].nunique()))
    add_check(checks, "all split assignment trial UIDs known", int(all_splits["trial_uid"].isin(set(prepared_index["trial_uid"])).sum()), int(len(all_splits)))

    checks_df = pd.DataFrame(checks)

    trial_report_path = out_dir / "05_trial_file_verification_report.csv"
    split_report_path = out_dir / "05_split_alignment_report.csv"
    dataset_class_report_path = out_dir / "05_dataset_class_report.csv"
    checks_path = out_dir / "05_loader_verification_checks.csv"
    summary_path = out_dir / "05_loader_verification_summary.json"

    trial_report.to_csv(trial_report_path, index=False)
    split_report.to_csv(split_report_path, index=False)
    dataset_class_report.to_csv(dataset_class_report_path, index=False)
    checks_df.to_csv(checks_path, index=False)

    failed_checks = checks_df[checks_df["passed"] == False]
    failed_trials = trial_report[trial_report["passed"] == False]
    failed_splits = split_report[split_report["passed"] == False]
    failed_dataset_classes = dataset_class_report[dataset_class_report["passed"] == False]

    overall_passed = (
        len(failed_checks) == 0
        and len(failed_trials) == 0
        and len(failed_splits) == 0
        and len(failed_dataset_classes) == 0
    )

    summary = {
        "name": loader_cfg["name"],
        "created_at": now(),
        "overall_passed": bool(overall_passed),
        "elapsed_seconds": round(time.time() - t0, 3),
        "row_counts": {
            "prepared_index_rows": int(len(prepared_index)),
            "trial_file_report_rows": int(len(trial_report)),
            "split_alignment_groups": int(len(split_report)),
            "dataset_class_checks": int(len(dataset_class_report)),
            "all_split_assignment_rows": int(len(all_splits)),
        },
        "outputs": {
            "trial_file_report": str(trial_report_path),
            "split_alignment_report": str(split_report_path),
            "dataset_class_report": str(dataset_class_report_path),
            "checks": str(checks_path),
            "summary": str(summary_path),
            "log": str(out_dir / "05_verify_dataset_loaders_log.txt"),
        },
        "failed_checks": failed_checks.to_dict(orient="records"),
        "failed_trial_files": failed_trials.head(50).to_dict(orient="records"),
        "failed_split_alignments": failed_splits.head(50).to_dict(orient="records"),
        "failed_dataset_class_checks": failed_dataset_classes.to_dict(orient="records"),
        "leakage_statement": "This stage only verifies prepared trial files, labels, split assignments, and dataset class loading. It performs no fitting, scaling, feature selection, calibration, sampling, or training.",
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Wrote trial file report: {trial_report_path}")
    logger.info(f"Wrote split alignment report: {split_report_path}")
    logger.info(f"Wrote dataset class report: {dataset_class_report_path}")
    logger.info(f"Wrote checks: {checks_path}")
    logger.info(f"Wrote summary: {summary_path}")
    logger.info(f"Overall loader verification passed: {overall_passed}")
    logger.info(f"Elapsed seconds: {summary['elapsed_seconds']}")

    print("\nTARGETED OUTPUTS")
    print(f"1. {trial_report_path}")
    print(f"2. {split_report_path}")
    print(f"3. {dataset_class_report_path}")
    print(f"4. {checks_path}")
    print(f"5. {summary_path}")
    print(f"6. {out_dir / '05_verify_dataset_loaders_log.txt'}")

    if not overall_passed:
        logger.error("Loader verification failed. Do not proceed to model/data module design.")
        return 1

    logger.info("Loader verification passed. It is safe to proceed to model/data module design.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

