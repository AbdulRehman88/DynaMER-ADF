from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


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


def require_previous_manifest(project_root: Path, cfg: Dict[str, Any], logger: Logger) -> None:
    prev = cfg["required_manifest"]
    path = project_root / prev["summary_json"]
    if not path.exists():
        raise FileNotFoundError(f"Required manifest summary not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    passed = bool(data.get("overall_passed", False))
    logger.info(f"Previous manifest summary found: {path}")
    logger.info(f"Previous manifest stage passed: {passed}")
    if prev.get("require_passed", True) and not passed:
        raise RuntimeError("Previous manifest stage did not pass. Refusing to generate splits.")


def normalize_label_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def label_counts(df: pd.DataFrame, label_col: str) -> Dict[str, int]:
    counts = df[label_col].value_counts(dropna=False).sort_index()
    out = {}
    for k, v in counts.items():
        if pd.isna(k):
            out["NaN"] = int(v)
        else:
            out[str(int(k))] = int(v)
    return out


def add_check(checks: List[Dict[str, Any]], split_id: str, dataset: str, task: str, protocol: str, check: str, observed: Any, expected: Any, passed: bool | None = None) -> None:
    if passed is None:
        passed = observed == expected
    checks.append({
        "split_id": split_id,
        "dataset": dataset,
        "task": task,
        "protocol": protocol,
        "check": check,
        "observed": json.dumps(observed, ensure_ascii=False),
        "expected": json.dumps(expected, ensure_ascii=False),
        "passed": bool(passed),
    })


def make_split_frame(
    df: pd.DataFrame,
    label_col: str,
    dataset: str,
    protocol: str,
    fold_index: int,
    split_map: Dict[str, str],
    is_primary: bool,
    leakage_note: str,
) -> pd.DataFrame:
    sub = df.copy()
    sub["label"] = normalize_label_series(sub[label_col]).astype(int)
    sub["split"] = sub["trial_uid"].map(split_map)
    if sub["split"].isna().any():
        missing = int(sub["split"].isna().sum())
        raise RuntimeError(f"{dataset}-{label_col}-{protocol}-fold{fold_index}: {missing} trials have no split assignment.")

    split_id = f"{dataset}__{label_col}__{protocol}__fold_{fold_index:03d}"
    out_cols = [
        "split_id",
        "dataset",
        "task",
        "protocol",
        "fold_index",
        "trial_uid",
        "subject_id",
        "subject_index",
        "session_id",
        "session_index",
        "trial_id",
        "trial_index",
        "label",
        "split",
        "is_primary_protocol",
        "leakage_note",
    ]

    sub["split_id"] = split_id
    sub["task"] = label_col
    sub["protocol"] = protocol
    sub["fold_index"] = fold_index
    sub["is_primary_protocol"] = int(is_primary)
    sub["leakage_note"] = leakage_note

    for col in out_cols:
        if col not in sub.columns:
            sub[col] = np.nan

    return sub[out_cols]


def choose_validation_subjects(
    train_subjects: List[str],
    df: pd.DataFrame,
    label_col: str,
    fold_index: int,
    val_fraction: float,
    min_val_subjects: int,
    seed: int,
) -> List[str]:
    train_subjects = sorted(train_subjects)
    n_val = max(min_val_subjects, int(round(len(train_subjects) * val_fraction)))
    n_val = min(max(1, n_val), max(1, len(train_subjects) - 1))

    label_set_all = set(pd.to_numeric(df[label_col], errors="coerce").dropna().astype(int).unique().tolist())

    rng = np.random.default_rng(seed + fold_index * 9973)
    best_subjects = None
    best_score = -1

    for _ in range(500):
        candidate = sorted(rng.choice(train_subjects, size=n_val, replace=False).tolist())
        val_labels = set(
            pd.to_numeric(df[df["subject_id"].isin(candidate)][label_col], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )
        score = len(val_labels.intersection(label_set_all))
        if score > best_score:
            best_score = score
            best_subjects = candidate
        if val_labels == label_set_all:
            return candidate

    if best_subjects is None:
        best_subjects = train_subjects[:n_val]
    return sorted(best_subjects)


def build_loso_splits(
    df: pd.DataFrame,
    dataset: str,
    label_col: str,
    seed: int,
    val_fraction: float,
    min_val_subjects: int,
) -> List[pd.DataFrame]:
    df = df.copy()
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = normalize_label_series(df[label_col]).astype(int)

    subjects = sorted(df["subject_id"].unique().tolist())
    split_frames = []

    for fold_index, test_subject in enumerate(tqdm(subjects, desc=f"{dataset} {label_col} LOSO", unit="fold"), start=1):
        train_candidate_subjects = [s for s in subjects if s != test_subject]
        val_subjects = choose_validation_subjects(
            train_subjects=train_candidate_subjects,
            df=df[df["subject_id"].isin(train_candidate_subjects)],
            label_col=label_col,
            fold_index=fold_index,
            val_fraction=val_fraction,
            min_val_subjects=min_val_subjects,
            seed=seed,
        )
        train_subjects = [s for s in train_candidate_subjects if s not in set(val_subjects)]

        split_map = {}
        for _, row in df.iterrows():
            uid = row["trial_uid"]
            sid = row["subject_id"]
            if sid == test_subject:
                split_map[uid] = "test"
            elif sid in val_subjects:
                split_map[uid] = "val"
            else:
                split_map[uid] = "train"

        split_frames.append(
            make_split_frame(
                df=df,
                label_col=label_col,
                dataset=dataset,
                protocol="subject_loso",
                fold_index=fold_index,
                split_map=split_map,
                is_primary=True,
                leakage_note="Primary subject-independent split. Test subject is completely absent from train and validation.",
            )
        )

    return split_frames


def build_subject_mixed_upper_bound_split(
    df: pd.DataFrame,
    dataset: str,
    label_col: str,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> pd.DataFrame:
    df = df.copy()
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = normalize_label_series(df[label_col]).astype(int)

    if not math.isclose(train_fraction + val_fraction + test_fraction, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError("Mixed split fractions must sum to 1.0.")

    y = df[label_col].to_numpy()
    all_idx = np.arange(len(df))

    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(all_idx.reshape(-1, 1), y))

    train_val_df = df.iloc[train_val_idx].copy()
    train_val_y = train_val_df[label_col].to_numpy()
    relative_val_fraction = val_fraction / (train_fraction + val_fraction)

    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_fraction, random_state=seed + 1)
    train_rel_idx, val_rel_idx = next(sss_val.split(np.arange(len(train_val_df)).reshape(-1, 1), train_val_y))

    train_idx = train_val_df.iloc[train_rel_idx].index.tolist()
    val_idx = train_val_df.iloc[val_rel_idx].index.tolist()
    test_abs_idx = df.iloc[test_idx].index.tolist()

    split_map = {}
    for idx, row in df.iterrows():
        if idx in set(test_abs_idx):
            split_map[row["trial_uid"]] = "test"
        elif idx in set(val_idx):
            split_map[row["trial_uid"]] = "val"
        elif idx in set(train_idx):
            split_map[row["trial_uid"]] = "train"
        else:
            raise RuntimeError("Unassigned row in subject-mixed split.")

    return make_split_frame(
        df=df,
        label_col=label_col,
        dataset=dataset,
        protocol="subject_mixed_upper_bound",
        fold_index=1,
        split_map=split_map,
        is_primary=False,
        leakage_note="Diagnostic upper-bound split only. Subjects may appear across train/validation/test and must not be used as the main generalization claim.",
    )


def build_seed_cross_session_splits(df: pd.DataFrame, label_col: str, seed: int) -> List[pd.DataFrame]:
    dataset = "SEED-IV"
    df = df.copy()
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = normalize_label_series(df[label_col]).astype(int)

    sessions = sorted(pd.to_numeric(df["session_index"], errors="coerce").dropna().astype(int).unique().tolist())
    split_frames = []

    for fold_index, test_session in enumerate(tqdm(sessions, desc="SEED-IV cross-session", unit="fold"), start=1):
        remaining = [s for s in sessions if s != test_session]
        val_session = remaining[(fold_index - 1) % len(remaining)]
        train_sessions = [s for s in remaining if s != val_session]

        split_map = {}
        for _, row in df.iterrows():
            uid = row["trial_uid"]
            sess = int(row["session_index"])
            if sess == test_session:
                split_map[uid] = "test"
            elif sess == val_session:
                split_map[uid] = "val"
            elif sess in train_sessions:
                split_map[uid] = "train"
            else:
                raise RuntimeError("Unexpected session assignment.")

        split_frames.append(
            make_split_frame(
                df=df,
                label_col=label_col,
                dataset=dataset,
                protocol="cross_session",
                fold_index=fold_index,
                split_map=split_map,
                is_primary=True,
                leakage_note="Session-transfer split. Test session is absent from train/validation, but subjects appear across sessions by design.",
            )
        )

    return split_frames


def verify_split_frame(split_df: pd.DataFrame, checks: List[Dict[str, Any]]) -> None:
    split_id = str(split_df["split_id"].iloc[0])
    dataset = str(split_df["dataset"].iloc[0])
    task = str(split_df["task"].iloc[0])
    protocol = str(split_df["protocol"].iloc[0])

    total_rows = int(len(split_df))
    split_counts = split_df["split"].value_counts().to_dict()
    split_counts = {str(k): int(v) for k, v in sorted(split_counts.items())}

    add_check(checks, split_id, dataset, task, protocol, "contains train/val/test", sorted(split_counts.keys()), ["test", "train", "val"])
    add_check(checks, split_id, dataset, task, protocol, "non-empty train", int(split_counts.get("train", 0)), ">0", int(split_counts.get("train", 0)) > 0)
    add_check(checks, split_id, dataset, task, protocol, "non-empty val", int(split_counts.get("val", 0)), ">0", int(split_counts.get("val", 0)) > 0)
    add_check(checks, split_id, dataset, task, protocol, "non-empty test", int(split_counts.get("test", 0)), ">0", int(split_counts.get("test", 0)) > 0)
    add_check(checks, split_id, dataset, task, protocol, "unique trial rows", int(split_df["trial_uid"].nunique()), total_rows)

    all_classes = sorted(split_df["label"].astype(str).unique().tolist())

    for split_name in ["train", "val", "test"]:
        split_part = split_df[split_df["split"] == split_name]
        counts = label_counts(split_part, "label")
        observed_classes = sorted(counts.keys())

        if split_name in ["train", "val"]:
            add_check(
                checks,
                split_id,
                dataset,
                task,
                protocol,
                f"{split_name} label classes",
                observed_classes,
                all_classes,
            )
        else:
            add_check(
                checks,
                split_id,
                dataset,
                task,
                protocol,
                "test label classes valid",
                observed_classes,
                f"non-empty subset of {all_classes}",
                len(observed_classes) > 0 and set(observed_classes).issubset(set(all_classes)),
            )

    if protocol == "subject_loso":
        train_subjects = set(split_df.loc[split_df["split"] == "train", "subject_id"].tolist())
        val_subjects = set(split_df.loc[split_df["split"] == "val", "subject_id"].tolist())
        test_subjects = set(split_df.loc[split_df["split"] == "test", "subject_id"].tolist())

        add_check(checks, split_id, dataset, task, protocol, "test subjects absent from train", sorted(test_subjects.intersection(train_subjects)), [])
        add_check(checks, split_id, dataset, task, protocol, "test subjects absent from val", sorted(test_subjects.intersection(val_subjects)), [])
        add_check(checks, split_id, dataset, task, protocol, "train subjects absent from val", sorted(train_subjects.intersection(val_subjects)), [])
        add_check(checks, split_id, dataset, task, protocol, "number of test subjects", len(test_subjects), 1)

    if protocol == "cross_session":
        train_sessions = set(split_df.loc[split_df["split"] == "train", "session_id"].tolist())
        val_sessions = set(split_df.loc[split_df["split"] == "val", "session_id"].tolist())
        test_sessions = set(split_df.loc[split_df["split"] == "test", "session_id"].tolist())

        add_check(checks, split_id, dataset, task, protocol, "test sessions absent from train", sorted(test_sessions.intersection(train_sessions)), [])
        add_check(checks, split_id, dataset, task, protocol, "test sessions absent from val", sorted(test_sessions.intersection(val_sessions)), [])
        add_check(checks, split_id, dataset, task, protocol, "number of test sessions", len(test_sessions), 1)


def write_split_files(split_frames: List[pd.DataFrame], out_dir: Path, logger: Logger) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    split_index_rows = []
    all_split_rows = []

    for split_df in tqdm(split_frames, desc="Writing split files", unit="split"):
        split_id = str(split_df["split_id"].iloc[0])
        dataset = str(split_df["dataset"].iloc[0])
        task = str(split_df["task"].iloc[0])
        protocol = str(split_df["protocol"].iloc[0])
        fold_index = int(split_df["fold_index"].iloc[0])

        file_name = f"03_{dataset.lower().replace('-', '_')}__{task}__{protocol}__fold_{fold_index:03d}.csv"
        file_path = out_dir / "split_files" / file_name
        file_path.parent.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(file_path, index=False)

        counts = split_df["split"].value_counts().to_dict()
        counts = {str(k): int(v) for k, v in sorted(counts.items())}

        split_index_rows.append({
            "split_id": split_id,
            "dataset": dataset,
            "task": task,
            "protocol": protocol,
            "fold_index": fold_index,
            "is_primary_protocol": int(split_df["is_primary_protocol"].iloc[0]),
            "n_rows": int(len(split_df)),
            "n_train": counts.get("train", 0),
            "n_val": counts.get("val", 0),
            "n_test": counts.get("test", 0),
            "split_file": str(file_path),
            "leakage_note": str(split_df["leakage_note"].iloc[0]),
        })

        all_split_rows.append(split_df)
        logger.info(f"Wrote split file: {file_path}")

    unified_splits = pd.concat(all_split_rows, ignore_index=True)
    return split_index_rows, unified_splits


def main() -> int:
    parser = argparse.ArgumentParser(description="03_prepare_leakage_safe_splits: split index generation with leakage checks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-paths", required=True)
    parser.add_argument("--split-config", default="configs/03_prepare_leakage_safe_splits.yaml")
    args = parser.parse_args()

    t0 = time.time()

    _main_cfg = load_yaml(Path(args.config))
    local_paths = load_yaml(Path(args.local_paths))
    split_cfg = load_yaml(Path(args.split_config))["splits"]

    project_root = as_path(local_paths["PROJECT_ROOT"])
    out_dir = project_root / split_cfg["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(out_dir / "03_prepare_leakage_safe_splits_log.txt")
    logger.info("Starting 03_prepare_leakage_safe_splits.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {out_dir}")

    require_previous_manifest(project_root, split_cfg, logger)

    manifest_paths = split_cfg["manifest_paths"]
    seed_df = pd.read_csv(project_root / manifest_paths["seed_iv"])
    dreamer_df = pd.read_csv(project_root / manifest_paths["dreamer"])
    amigos_df = pd.read_csv(project_root / manifest_paths["amigos"])

    policy = split_cfg["split_policy"]
    seed = int(split_cfg["random_seed"])

    split_frames: List[pd.DataFrame] = []

    logger.info("Generating SEED-IV splits.")
    seed_label = split_cfg["datasets"]["seed_iv"]["primary_label_column"]
    split_frames.extend(
        build_loso_splits(
            df=seed_df,
            dataset="SEED-IV",
            label_col=seed_label,
            seed=seed,
            val_fraction=float(policy["val_subject_fraction_loso"]),
            min_val_subjects=int(policy["min_val_subjects_loso"]),
        )
    )
    split_frames.extend(build_seed_cross_session_splits(seed_df, seed_label, seed))
    split_frames.append(
        build_subject_mixed_upper_bound_split(
            df=seed_df,
            dataset="SEED-IV",
            label_col=seed_label,
            seed=seed,
            train_fraction=float(policy["train_fraction_mixed"]),
            val_fraction=float(policy["val_fraction_mixed"]),
            test_fraction=float(policy["test_fraction_mixed"]),
        )
    )

    logger.info("Generating DREAMER splits.")
    for label_col in split_cfg["datasets"]["dreamer"]["label_columns"]:
        split_frames.extend(
            build_loso_splits(
                df=dreamer_df,
                dataset="DREAMER",
                label_col=label_col,
                seed=seed,
                val_fraction=float(policy["val_subject_fraction_loso"]),
                min_val_subjects=int(policy["min_val_subjects_loso"]),
            )
        )
        split_frames.append(
            build_subject_mixed_upper_bound_split(
                df=dreamer_df,
                dataset="DREAMER",
                label_col=label_col,
                seed=seed,
                train_fraction=float(policy["train_fraction_mixed"]),
                val_fraction=float(policy["val_fraction_mixed"]),
                test_fraction=float(policy["test_fraction_mixed"]),
            )
        )

    logger.info("Generating AMIGOS splits.")
    for label_col in split_cfg["datasets"]["amigos"]["label_columns"]:
        split_frames.extend(
            build_loso_splits(
                df=amigos_df,
                dataset="AMIGOS",
                label_col=label_col,
                seed=seed,
                val_fraction=float(policy["val_subject_fraction_loso"]),
                min_val_subjects=int(policy["min_val_subjects_loso"]),
            )
        )
        split_frames.append(
            build_subject_mixed_upper_bound_split(
                df=amigos_df,
                dataset="AMIGOS",
                label_col=label_col,
                seed=seed,
                train_fraction=float(policy["train_fraction_mixed"]),
                val_fraction=float(policy["val_fraction_mixed"]),
                test_fraction=float(policy["test_fraction_mixed"]),
            )
        )

    checks: List[Dict[str, Any]] = []
    for split_df in tqdm(split_frames, desc="Verifying split integrity", unit="split"):
        verify_split_frame(split_df, checks)

    split_index_rows, unified_splits = write_split_files(split_frames, out_dir, logger)

    split_index_df = pd.DataFrame(split_index_rows)
    checks_df = pd.DataFrame(checks)

    split_index_path = out_dir / "03_split_index.csv"
    unified_splits_path = out_dir / "03_all_split_assignments.csv"
    checks_path = out_dir / "03_split_integrity_checks.csv"
    summary_path = out_dir / "03_split_summary.json"

    split_index_df.to_csv(split_index_path, index=False)
    unified_splits.to_csv(unified_splits_path, index=False)
    checks_df.to_csv(checks_path, index=False)

    failed = checks_df[checks_df["passed"] == False]
    overall_passed = len(failed) == 0

    summary = {
        "name": split_cfg["name"],
        "created_at": now(),
        "overall_passed": bool(overall_passed),
        "elapsed_seconds": round(time.time() - t0, 3),
        "n_split_files": int(len(split_index_df)),
        "n_split_assignment_rows": int(len(unified_splits)),
        "datasets": sorted(split_index_df["dataset"].unique().tolist()),
        "protocols": sorted(split_index_df["protocol"].unique().tolist()),
        "tasks": sorted(split_index_df["task"].unique().tolist()),
        "outputs": {
            "split_index": str(split_index_path),
            "all_split_assignments": str(unified_splits_path),
            "integrity_checks": str(checks_path),
            "summary": str(summary_path),
            "log": str(out_dir / "03_prepare_leakage_safe_splits_log.txt"),
            "split_files_dir": str(out_dir / "split_files"),
        },
        "failed_checks": failed.to_dict(orient="records"),
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Wrote split index: {split_index_path}")
    logger.info(f"Wrote unified split assignments: {unified_splits_path}")
    logger.info(f"Wrote integrity checks: {checks_path}")
    logger.info(f"Wrote summary: {summary_path}")
    logger.info(f"Number of split files: {summary['n_split_files']}")
    logger.info(f"Overall split stage passed: {overall_passed}")
    logger.info(f"Elapsed seconds: {summary['elapsed_seconds']}")

    print("\nTARGETED OUTPUTS")
    print(f"1. {split_index_path}")
    print(f"2. {unified_splits_path}")
    print(f"3. {checks_path}")
    print(f"4. {summary_path}")
    print(f"5. {out_dir / '03_prepare_leakage_safe_splits_log.txt'}")
    print(f"6. {out_dir / 'split_files'}")

    if not overall_passed:
        logger.error("Split stage failed. Do not proceed to preprocessing, feature preparation, or training.")
        return 1

    logger.info("Split stage passed. It is safe to proceed to dataset loading/preparation stage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

