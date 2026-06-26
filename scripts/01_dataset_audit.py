from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import scipy.io as sio
import yaml

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class AuditLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_path.write_text("", encoding="utf-8")

    def info(self, message: str) -> None:
        line = f"[{now()}] [INFO] {message}"
        print(line)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def warn(self, message: str) -> None:
        line = f"[{now()}] [WARN] {message}"
        print(line)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def error(self, message: str) -> None:
        line = f"[{now()}] [ERROR] {message}"
        print(line)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def to_path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def clean_zip_members(zip_path: Path, suffix: str = ".mat") -> List[str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            m for m in zf.namelist()
            if m.lower().endswith(suffix.lower())
            and "__macosx" not in m.lower()
            and not Path(m).name.startswith(".")
        ]
    return sorted(members)


def mat_load_from_zip(zip_path: Path, member: str) -> Dict[str, Any]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        payload = zf.read(member)
    return sio.loadmat(BytesIO(payload), simplify_cells=True)


def public_mat_keys(mat: Dict[str, Any]) -> List[str]:
    return sorted([k for k in mat.keys() if not k.startswith("__")])


def shape_list(x: Any) -> List[int]:
    return list(np.asarray(x).shape)


def get_field(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return [x.item()]
        return list(x.ravel())
    return [x]


def binary_counts_from_array(values: Any, threshold: Optional[float] = None) -> Dict[str, int]:
    arr = np.asarray(values).reshape(-1)
    arr = arr[~pd.isna(arr)]
    if threshold is None:
        low = int(np.sum(arr == 0))
        high = int(np.sum(arr == 1))
    else:
        low = int(np.sum(arr <= threshold))
        high = int(np.sum(arr > threshold))
    return {"low": low, "high": high}


def find_binary_column(df: pd.DataFrame, keyword: str) -> Optional[str]:
    keyword = keyword.lower()
    candidates = []
    for col in df.columns:
        c = str(col).lower()
        if keyword not in c:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna().unique()
        vals = sorted([int(v) for v in vals if float(v).is_integer()])
        if set(vals).issubset({0, 1}) and len(vals) > 0:
            candidates.append(col)
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda x: len(str(x)))
    return str(candidates[0])


def add_check(checks: List[Dict[str, Any]], name: str, observed: Any, expected: Any) -> None:
    passed = observed == expected
    checks.append(
        {
            "check": name,
            "observed": observed,
            "expected": expected,
            "passed": bool(passed),
        }
    )


def add_exists_check(checks: List[Dict[str, Any]], name: str, path: Path) -> None:
    checks.append(
        {
            "check": name,
            "observed": str(path),
            "expected": "exists",
            "passed": bool(path.exists()),
        }
    )


def audit_seed_iv(root: Path, cfg: Dict[str, Any], logger: AuditLogger) -> Dict[str, Any]:
    logger.info("Starting SEED-IV structural audit.")
    files = cfg["files"]
    expected = cfg["expected"]
    checks: List[Dict[str, Any]] = []

    paths = {k: root / v for k, v in files.items()}
    for key, path in paths.items():
        add_exists_check(checks, f"SEED-IV file exists: {key}", path)

    result: Dict[str, Any] = {
        "dataset": "SEED-IV",
        "root": str(root),
        "paths": {k: str(v) for k, v in paths.items()},
        "zip_counts": {},
        "feature_summaries": {},
        "checks": checks,
    }

    for zip_key in ["eeg_feature_zip", "eye_feature_zip", "eeg_raw_zip", "eye_raw_zip"]:
        zip_path = paths[zip_key]
        if zip_path.exists():
            members = clean_zip_members(zip_path)
            result["zip_counts"][zip_key] = {
                "mat_files": len(members),
                "first_members": members[:5],
            }
        else:
            result["zip_counts"][zip_key] = {"mat_files": 0, "first_members": []}

    add_check(
        checks,
        "SEED-IV EEG feature MAT files",
        result["zip_counts"]["eeg_feature_zip"]["mat_files"],
        expected["feature_mat_files"],
    )
    add_check(
        checks,
        "SEED-IV eye feature MAT files",
        result["zip_counts"]["eye_feature_zip"]["mat_files"],
        expected["feature_mat_files"],
    )

    feature_specs = [
        ("eeg_feature_zip", r"^(de_LDS|psd_LDS)\d+$"),
        ("eye_feature_zip", r"^eye_\d+$"),
    ]

    for zip_key, pattern in feature_specs:
        zip_path = paths[zip_key]
        regex = re.compile(pattern)
        mat_members = clean_zip_members(zip_path) if zip_path.exists() else []
        total_trial_vars = 0
        per_file_counts = []
        shape_examples = []

        for member in tqdm(mat_members, desc=f"Auditing SEED-IV {zip_key}", unit="mat"):
            mat = mat_load_from_zip(zip_path, member)
            keys = public_mat_keys(mat)
            trial_keys = [k for k in keys if regex.match(k)]
            total_trial_vars += len(trial_keys)
            per_file_counts.append(len(trial_keys))
            if trial_keys and len(shape_examples) < 5:
                shape_examples.append(
                    {
                        "member": member,
                        "variable": trial_keys[0],
                        "shape": shape_list(mat[trial_keys[0]]),
                    }
                )

        result["feature_summaries"][zip_key] = {
            "mat_files": len(mat_members),
            "total_trial_variables": total_trial_vars,
            "per_file_trial_variable_counts_unique": sorted(set(per_file_counts)),
            "shape_examples": shape_examples,
        }

        expected_total = expected["total_trials"] * 2 if zip_key == "eeg_feature_zip" else expected["total_trials"]
        expected_per_file = expected["trials_per_feature_mat"] * 2 if zip_key == "eeg_feature_zip" else expected["trials_per_feature_mat"]

        add_check(
            checks,
            f"SEED-IV {zip_key} total trial variables",
            total_trial_vars,
            expected_total,
        )
        add_check(
            checks,
            f"SEED-IV {zip_key} per-file trial count",
            sorted(set(per_file_counts)),
            [expected_per_file],
        )

    result["passed"] = all(c["passed"] for c in checks)
    logger.info(f"Finished SEED-IV audit. Passed={result['passed']}")
    return result


def audit_dreamer(root: Path, cfg: Dict[str, Any], logger: AuditLogger) -> Dict[str, Any]:
    logger.info("Starting DREAMER structural audit.")
    files = cfg["files"]
    expected = cfg["expected"]
    checks: List[Dict[str, Any]] = []

    mat_path = root / files["mat_file"]
    doc_path = root / files["documentation_file"]
    add_exists_check(checks, "DREAMER MAT file exists", mat_path)
    add_exists_check(checks, "DREAMER documentation file exists", doc_path)

    result: Dict[str, Any] = {
        "dataset": "DREAMER",
        "root": str(root),
        "paths": {"mat_file": str(mat_path), "documentation_file": str(doc_path)},
        "checks": checks,
    }

    if not mat_path.exists():
        result["passed"] = False
        logger.error("DREAMER.mat is missing.")
        return result

    mat = sio.loadmat(mat_path, simplify_cells=True)
    dreamer = mat.get("DREAMER")
    data = get_field(dreamer, "Data")
    subjects = as_list(data)

    n_subjects = len(subjects)
    trial_counts = []
    label_counts = {}
    eeg_channel_candidates = []
    ecg_channel_candidates = []
    eeg_duration_sec = []
    ecg_duration_sec = []

    for subj in tqdm(subjects, desc="Auditing DREAMER subjects", unit="subject"):
        valence = get_field(subj, "ScoreValence")
        arousal = get_field(subj, "ScoreArousal")
        dominance = get_field(subj, "ScoreDominance")

        scores = {
            "valence": np.asarray(valence).reshape(-1) if valence is not None else np.array([]),
            "arousal": np.asarray(arousal).reshape(-1) if arousal is not None else np.array([]),
            "dominance": np.asarray(dominance).reshape(-1) if dominance is not None else np.array([]),
        }

        if len(scores["valence"]) > 0:
            trial_counts.append(int(len(scores["valence"])))

        for label_name, arr in scores.items():
            if label_name not in label_counts:
                label_counts[label_name] = {"low": 0, "high": 0}
            c = binary_counts_from_array(arr, threshold=3)
            label_counts[label_name]["low"] += c["low"]
            label_counts[label_name]["high"] += c["high"]

        eeg = get_field(subj, "EEG")
        ecg = get_field(subj, "ECG")
        eeg_stimuli = as_list(get_field(eeg, "stimuli"))
        ecg_stimuli = as_list(get_field(ecg, "stimuli"))

        for arr in eeg_stimuli:
            a = np.asarray(arr)
            if a.ndim == 2:
                dims = list(a.shape)
                ch = min(dims)
                samples = max(dims)
                eeg_channel_candidates.append(int(ch))
                eeg_duration_sec.append(float(samples / expected["eeg_sampling_rate_hz"]))

        for arr in ecg_stimuli:
            a = np.asarray(arr)
            if a.ndim == 2:
                dims = list(a.shape)
                ch = min(dims)
                samples = max(dims)
                ecg_channel_candidates.append(int(ch))
                ecg_duration_sec.append(float(samples / expected["ecg_sampling_rate_hz"]))

    total_trials = int(sum(trial_counts))

    result.update(
        {
            "subjects": n_subjects,
            "trial_counts_unique": sorted(set(trial_counts)),
            "total_trials_from_scores": total_trials,
            "label_counts_threshold_gt_3": label_counts,
            "eeg_channel_candidates_unique": sorted(set(eeg_channel_candidates)),
            "ecg_channel_candidates_unique": sorted(set(ecg_channel_candidates)),
            "eeg_duration_sec": {
                "min": round(float(np.min(eeg_duration_sec)), 3) if eeg_duration_sec else None,
                "mean": round(float(np.mean(eeg_duration_sec)), 3) if eeg_duration_sec else None,
                "max": round(float(np.max(eeg_duration_sec)), 3) if eeg_duration_sec else None,
            },
            "ecg_duration_sec": {
                "min": round(float(np.min(ecg_duration_sec)), 3) if ecg_duration_sec else None,
                "mean": round(float(np.mean(ecg_duration_sec)), 3) if ecg_duration_sec else None,
                "max": round(float(np.max(ecg_duration_sec)), 3) if ecg_duration_sec else None,
            },
        }
    )

    add_check(checks, "DREAMER subjects", n_subjects, expected["subjects"])
    add_check(checks, "DREAMER trials per subject", sorted(set(trial_counts)), [expected["trials_per_subject"]])
    add_check(checks, "DREAMER total trials", total_trials, expected["total_trials"])
    add_check(checks, "DREAMER EEG channels", sorted(set(eeg_channel_candidates)), [expected["eeg_channels"]])
    add_check(checks, "DREAMER ECG channels", sorted(set(ecg_channel_candidates)), [expected["ecg_channels"]])

    for label_name, expected_counts in expected["label_counts"].items():
        add_check(
            checks,
            f"DREAMER {label_name} binary counts",
            label_counts.get(label_name),
            expected_counts,
        )

    result["passed"] = all(c["passed"] for c in checks)
    logger.info(f"Finished DREAMER audit. Passed={result['passed']}")
    return result


def find_npz_array(npz_obj: Any, preferred: List[str], ndim: int) -> Optional[str]:
    keys = list(npz_obj.keys())
    for k in preferred:
        if k in keys and np.asarray(npz_obj[k]).ndim == ndim:
            return k
    candidates = []
    for k in keys:
        arr = np.asarray(npz_obj[k])
        if arr.ndim == ndim:
            candidates.append((k, arr.size))
    if not candidates:
        return None
    return sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]


def audit_amigos(root: Path, cfg: Dict[str, Any], logger: AuditLogger) -> Dict[str, Any]:
    logger.info("Starting AMIGOS structural audit.")
    files = cfg["files"]
    expected = cfg["expected"]
    checks: List[Dict[str, Any]] = []

    paths = {k: root / v for k, v in files.items()}
    for key, path in paths.items():
        add_exists_check(checks, f"AMIGOS file exists: {key}", path)

    result: Dict[str, Any] = {
        "dataset": "AMIGOS",
        "root": str(root),
        "paths": {k: str(v) for k, v in paths.items()},
        "checks": checks,
    }

    trial_bag_path = paths["trial_bag_file"]
    feature_bag_path = paths["feature_bag_file"]
    trial_meta_path = paths["trial_metadata_file"]
    feature_meta_path = paths["feature_metadata_file"]
    feature_names_path = paths["feature_names_file"]
    window_meta_path = paths["window_metadata_file"]

    with np.load(trial_bag_path, allow_pickle=True) as trial_npz:
        trial_keys = list(trial_npz.keys())
        trial_array_key = find_npz_array(trial_npz, ["X_bags", "X", "data"], ndim=4)
        trial_shape = shape_list(trial_npz[trial_array_key]) if trial_array_key else None

    with np.load(feature_bag_path, allow_pickle=True) as feature_npz:
        feature_keys = list(feature_npz.keys())
        feature_array_key = find_npz_array(feature_npz, ["F_bags", "X_features", "features"], ndim=3)
        feature_shape = shape_list(feature_npz[feature_array_key]) if feature_array_key else None

    trial_meta = pd.read_csv(trial_meta_path)
    feature_meta = pd.read_csv(feature_meta_path)
    feature_names = pd.read_csv(feature_names_path)
    window_meta = pd.read_csv(window_meta_path)

    val_col = find_binary_column(trial_meta, "valence")
    aro_col = find_binary_column(trial_meta, "arousal")

    val_counts = None
    aro_counts = None

    if val_col:
        val_counts = binary_counts_from_array(trial_meta[val_col])
    if aro_col:
        aro_counts = binary_counts_from_array(trial_meta[aro_col])

    result.update(
        {
            "trial_npz_keys": trial_keys,
            "trial_array_key": trial_array_key,
            "trial_bag_shape": trial_shape,
            "feature_npz_keys": feature_keys,
            "feature_array_key": feature_array_key,
            "feature_bag_shape": feature_shape,
            "trial_metadata_rows": int(len(trial_meta)),
            "feature_metadata_rows": int(len(feature_meta)),
            "window_metadata_rows": int(len(window_meta)),
            "feature_names_rows": int(len(feature_names)),
            "trial_metadata_columns": list(map(str, trial_meta.columns)),
            "detected_valence_column": val_col,
            "detected_arousal_column": aro_col,
            "valence_counts": val_counts,
            "arousal_counts": aro_counts,
        }
    )

    add_check(checks, "AMIGOS trial bag shape", trial_shape, expected["trial_bag_shape"])
    add_check(checks, "AMIGOS feature bag shape", feature_shape, expected["feature_bag_shape"])
    add_check(checks, "AMIGOS trial metadata rows", int(len(trial_meta)), expected["trials"])
    add_check(checks, "AMIGOS feature metadata rows", int(len(feature_meta)), expected["trials"])
    add_check(checks, "AMIGOS feature names count", int(len(feature_names)), expected["feature_bag_shape"][-1])
    add_check(checks, "AMIGOS valence binary counts", val_counts, expected["valence_counts"])
    add_check(checks, "AMIGOS arousal binary counts", aro_counts, expected["arousal_counts"])

    result["passed"] = all(c["passed"] for c in checks)
    logger.info(f"Finished AMIGOS audit. Passed={result['passed']}")
    return result


def flatten_checks(dataset_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    dataset = dataset_result["dataset"]
    for c in dataset_result.get("checks", []):
        rows.append(
            {
                "dataset": dataset,
                "check": c["check"],
                "observed": json.dumps(c["observed"], ensure_ascii=False),
                "expected": json.dumps(c["expected"], ensure_ascii=False),
                "passed": c["passed"],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="01_dataset_audit: leakage-free structural dataset audit.")
    parser.add_argument("--config", required=True, help="Main project config YAML.")
    parser.add_argument("--local-paths", required=True, help="Machine-local path YAML.")
    parser.add_argument("--audit-config", default="configs/01_dataset_audit.yaml", help="Audit target YAML.")
    args = parser.parse_args()

    t0 = time.time()

    config = load_yaml(Path(args.config))
    local_paths = load_yaml(Path(args.local_paths))
    audit_config = load_yaml(Path(args.audit_config))["audit"]

    project_root = to_path(local_paths["PROJECT_ROOT"])
    out_dir = project_root / audit_config["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = AuditLogger(out_dir / "01_dataset_audit_log.txt")
    logger.info("Starting 01_dataset_audit.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {out_dir}")

    seed_root = to_path(local_paths[audit_config["seed_iv"]["root_key"]])
    dreamer_root = to_path(local_paths[audit_config["dreamer"]["root_key"]])
    amigos_root = to_path(local_paths[audit_config["amigos"]["root_key"]])

    results = {
        "audit_name": audit_config["name"],
        "created_at": now(),
        "project_root": str(project_root),
        "datasets": {},
    }

    results["datasets"]["SEED-IV"] = audit_seed_iv(seed_root, audit_config["seed_iv"], logger)
    results["datasets"]["DREAMER"] = audit_dreamer(dreamer_root, audit_config["dreamer"], logger)
    results["datasets"]["AMIGOS"] = audit_amigos(amigos_root, audit_config["amigos"], logger)

    all_checks = []
    for dataset_result in results["datasets"].values():
        all_checks.extend(flatten_checks(dataset_result))

    checks_df = pd.DataFrame(all_checks)
    summary_df = (
        checks_df.groupby("dataset")["passed"]
        .agg(total_checks="count", passed_checks="sum")
        .reset_index()
    )
    summary_df["failed_checks"] = summary_df["total_checks"] - summary_df["passed_checks"]
    summary_df["dataset_passed"] = summary_df["failed_checks"] == 0

    overall_passed = bool(summary_df["dataset_passed"].all())
    results["overall_passed"] = overall_passed
    results["elapsed_seconds"] = round(time.time() - t0, 3)

    json_path = out_dir / "01_dataset_audit_summary.json"
    checks_path = out_dir / "01_dataset_audit_checks.csv"
    summary_path = out_dir / "01_dataset_audit_dataset_summary.csv"

    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    checks_df.to_csv(checks_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    logger.info(f"Wrote JSON summary: {json_path}")
    logger.info(f"Wrote check table: {checks_path}")
    logger.info(f"Wrote dataset summary: {summary_path}")
    logger.info(f"Overall audit passed: {overall_passed}")
    logger.info(f"Elapsed seconds: {results['elapsed_seconds']}")

    print("\nTARGETED OUTPUTS")
    print(f"1. {json_path}")
    print(f"2. {checks_path}")
    print(f"3. {summary_path}")
    print(f"4. {out_dir / '01_dataset_audit_log.txt'}")

    if not overall_passed:
        logger.error("Audit failed. Do not proceed to preprocessing or training.")
        return 1

    logger.info("Audit passed. It is safe to proceed to the next stage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

