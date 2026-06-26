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
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


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


def add_check(checks: List[Dict[str, Any]], check: str, observed: Any, expected: Any, passed: Optional[bool] = None) -> None:
    if passed is None:
        passed = observed == expected
    checks.append({
        "check": check,
        "observed": json.dumps(observed, ensure_ascii=False),
        "expected": json.dumps(expected, ensure_ascii=False),
        "passed": bool(passed),
    })


def normalize_label_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def label_counts(df: pd.DataFrame, label_col: str) -> Dict[str, int]:
    counts = df[label_col].value_counts(dropna=False).sort_index()
    out: Dict[str, int] = {}
    for k, v in counts.items():
        if pd.isna(k):
            out["NaN"] = int(v)
        else:
            out[str(int(k))] = int(v)
    return out


def make_split_frame(
    df: pd.DataFrame,
    label_col: str,
    fold_index: int,
    split_map: Dict[str, str],
) -> pd.DataFrame:
    sub = df.copy()
    sub[label_col] = normalize_label_series(sub[label_col]).astype(int)
    sub["label"] = sub[label_col].astype(int)
    sub["split"] = sub["trial_uid"].map(split_map)
    if sub["split"].isna().any():
        raise RuntimeError(f"Fold {fold_index}: split assignment missing for {int(sub['split'].isna().sum())} rows.")

    split_id = f"SEED-IV__seed_iv_label__subject_mixed_5fold__fold_{fold_index:03d}"
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
    sub["dataset"] = "SEED-IV"
    sub["task"] = "seed_iv_label"
    sub["protocol"] = "subject_mixed_5fold"
    sub["fold_index"] = int(fold_index)
    sub["is_primary_protocol"] = 0
    sub["leakage_note"] = (
        "Diagnostic subject-mixed 5-fold split. Subjects may appear across train/validation/test. "
        "Use only as an upper-bound capacity estimate, not as the main subject-independent claim. "
        "Validation is drawn only from the training portion of each fold, and all normalization must be fit on train only."
    )
    for col in out_cols:
        if col not in sub.columns:
            sub[col] = np.nan
    return sub[out_cols].copy()


def verify_split_frame(split_df: pd.DataFrame, checks: List[Dict[str, Any]]) -> None:
    split_id = str(split_df["split_id"].iloc[0])
    prefix = f"{split_id} :: "
    total_rows = int(len(split_df))
    split_counts = {str(k): int(v) for k, v in split_df["split"].value_counts().sort_index().to_dict().items()}
    add_check(checks, prefix + "contains train/val/test", sorted(split_counts.keys()), ["test", "train", "val"])
    add_check(checks, prefix + "unique trial rows", int(split_df["trial_uid"].nunique()), total_rows)
    for split_name in ["train", "val", "test"]:
        part = split_df[split_df["split"] == split_name]
        add_check(checks, prefix + f"non-empty {split_name}", int(len(part)), ">0", int(len(part)) > 0)
        add_check(checks, prefix + f"{split_name} class coverage", sorted(label_counts(part, "label").keys()), ["0", "1", "2", "3"])
    # Subject mixing is expected, not a failure. Record the overlap as evidence of diagnostic design.
    train_subjects = set(split_df.loc[split_df["split"] == "train", "subject_id"].astype(str).tolist())
    val_subjects = set(split_df.loc[split_df["split"] == "val", "subject_id"].astype(str).tolist())
    test_subjects = set(split_df.loc[split_df["split"] == "test", "subject_id"].astype(str).tolist())
    add_check(checks, prefix + "subject overlap train-test allowed", len(train_subjects.intersection(test_subjects)), ">=1", len(train_subjects.intersection(test_subjects)) >= 1)
    add_check(checks, prefix + "subject overlap train-val allowed", len(train_subjects.intersection(val_subjects)), ">=1", len(train_subjects.intersection(val_subjects)) >= 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 22A: create SEED-IV subject-mixed 5-fold diagnostic split files and registry.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--local-paths", default="configs/local_paths.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction-of-trainval", type=float, default=0.1875, help="Validation fraction within the 80 percent train+val pool. Default gives 65/15/20 overall.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    _main_cfg = load_yaml(Path(args.config))
    local_paths = load_yaml(Path(args.local_paths))
    project_root = as_path(local_paths["PROJECT_ROOT"])

    out_dir = project_root / "outputs" / "protocol_extension" / "22_seed_iv_subject_mixed_5fold"
    if out_dir.exists() and args.overwrite:
        import shutil
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir = out_dir / "split_files"
    split_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(out_dir / "22A_seed_iv_subject_mixed_5fold_splits_log.txt")
    logger.info("Starting Stage 22A SEED-IV subject-mixed 5-fold split generation.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {out_dir}")

    manifest_path = project_root / "outputs" / "manifests" / "02_prepare_dataset_manifests" / "02_seed_iv_trial_manifest.csv"
    temporal_index_path = project_root / "outputs" / "temporal_views" / "08_prepare_temporal_feature_views" / "08_temporal_view_index.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing SEED-IV manifest: {manifest_path}")
    if not temporal_index_path.exists():
        raise FileNotFoundError(f"Missing temporal view index: {temporal_index_path}")

    df = pd.read_csv(manifest_path)
    label_col = "seed_iv_label"
    required = ["trial_uid", "subject_id", "session_id", label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"SEED-IV manifest missing required columns: {missing}")

    df = df.dropna(subset=[label_col]).copy().reset_index(drop=True)
    df[label_col] = normalize_label_series(df[label_col]).astype(int)
    y = df[label_col].to_numpy(dtype=int)
    indices = np.arange(len(df))

    logger.info(f"SEED-IV rows available: {len(df)}")
    logger.info(f"SEED-IV label counts: {label_counts(df, label_col)}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(args.seed))
    split_index_rows: List[Dict[str, Any]] = []
    all_split_frames: List[pd.DataFrame] = []
    checks: List[Dict[str, Any]] = []

    for fold_index, (train_val_idx, test_idx) in enumerate(skf.split(indices, y), start=1):
        train_val_df = df.iloc[train_val_idx].copy().reset_index(drop=False).rename(columns={"index": "orig_index"})
        train_val_y = train_val_df[label_col].to_numpy(dtype=int)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=float(args.val_fraction_of_trainval), random_state=int(args.seed) + fold_index * 1009)
        train_rel, val_rel = next(sss.split(np.arange(len(train_val_df)).reshape(-1, 1), train_val_y))

        train_orig = set(train_val_df.iloc[train_rel]["orig_index"].astype(int).tolist())
        val_orig = set(train_val_df.iloc[val_rel]["orig_index"].astype(int).tolist())
        test_orig = set(test_idx.astype(int).tolist())

        split_map: Dict[str, str] = {}
        for idx, row in df.iterrows():
            uid = str(row["trial_uid"])
            if idx in test_orig:
                split_map[uid] = "test"
            elif idx in val_orig:
                split_map[uid] = "val"
            elif idx in train_orig:
                split_map[uid] = "train"
            else:
                raise RuntimeError(f"Fold {fold_index}: row {idx} unassigned.")

        split_df = make_split_frame(df=df, label_col=label_col, fold_index=fold_index, split_map=split_map)
        verify_split_frame(split_df, checks)

        file_name = f"22_seed_iv__seed_iv_label__subject_mixed_5fold__fold_{fold_index:03d}.csv"
        file_path = split_dir / file_name
        split_df.to_csv(file_path, index=False)
        counts = {str(k): int(v) for k, v in split_df["split"].value_counts().sort_index().to_dict().items()}

        split_index_rows.append({
            "split_id": str(split_df["split_id"].iloc[0]),
            "dataset": "SEED-IV",
            "task": "seed_iv_label",
            "protocol": "subject_mixed_5fold",
            "fold_index": int(fold_index),
            "is_primary_protocol": 0,
            "n_rows": int(len(split_df)),
            "n_train": int(counts.get("train", 0)),
            "n_val": int(counts.get("val", 0)),
            "n_test": int(counts.get("test", 0)),
            "split_file": str(file_path),
            "leakage_note": str(split_df["leakage_note"].iloc[0]),
        })
        all_split_frames.append(split_df)
        logger.info(f"Wrote fold {fold_index}: {file_path} counts={counts}")

    split_index = pd.DataFrame(split_index_rows)
    all_splits = pd.concat(all_split_frames, ignore_index=True)

    # Build a Stage-11-compatible registry for the new diagnostic protocol.
    registry_rows = []
    for _, row in split_index.iterrows():
        run_id = f"22__seed_iv_subject_mixed_5fold__SEED_IV__seed_iv_label__fold_{int(row['fold_index']):03d}"
        registry_rows.append({
            "run_id": run_id,
            "phase": "phase_5_seed_iv_subject_mixed_5fold",
            "phase_priority": 5,
            "phase_purpose": "Diagnostic subject-mixed 5-fold capacity evaluation requested after professor review.",
            "dataset": "SEED-IV",
            "task": "seed_iv_label",
            "protocol": "subject_mixed_5fold",
            "scientific_role": "diagnostic_subject_mixed_5fold_upper_bound_only",
            "is_primary_claim_allowed": 0,
            "fold_index": int(row["fold_index"]),
            "split_id": str(row["split_id"]),
            "split_file": str(row["split_file"]),
            "model_name": "DynaMER-ADF-compatible temporal framework",
            "hidden_dim": 128,
            "temporal_backbone": "mixed_by_model_variant",
            "fusion": "mixed_by_model_variant",
            "head": "mixed_by_model_variant",
            "label_column": "label_seed_iv",
            "num_classes": 4,
            "modality_keys": "eeg_combined|eye_features",
            "status": "queued",
            "notes": "Generated by Stage 22A. Subject-mixed diagnostic CV only; not a primary subject-independent claim.",
        })
    registry = pd.DataFrame(registry_rows)

    add_check(checks, "number of subject-mixed 5-fold split files", int(len(split_index)), 5)
    add_check(checks, "registry rows", int(len(registry)), 5)
    add_check(checks, "all split files exist", int(sum(Path(p).exists() for p in split_index["split_file"])), 5)
    add_check(checks, "registry marked non-primary", int(registry["is_primary_claim_allowed"].sum()), 0)
    add_check(checks, "temporal view index exists", temporal_index_path.exists(), True)

    checks_df = pd.DataFrame(checks)
    failed = checks_df[checks_df["passed"] == False]
    overall_passed = len(failed) == 0

    split_index_path = out_dir / "22_seed_iv_subject_mixed_5fold_split_index.csv"
    all_splits_path = out_dir / "22_seed_iv_subject_mixed_5fold_all_split_assignments.csv"
    registry_path = out_dir / "22_seed_iv_subject_mixed_5fold_registry.csv"
    checks_path = out_dir / "22_seed_iv_subject_mixed_5fold_checks.csv"
    summary_path = out_dir / "22_seed_iv_subject_mixed_5fold_summary.json"

    split_index.to_csv(split_index_path, index=False)
    all_splits.to_csv(all_splits_path, index=False)
    registry.to_csv(registry_path, index=False)
    checks_df.to_csv(checks_path, index=False)

    summary = {
        "name": "22A_seed_iv_subject_mixed_5fold_splits",
        "created_at": now(),
        "overall_passed": bool(overall_passed),
        "elapsed_seconds": round(time.time() - t0, 3),
        "protocol": "subject_mixed_5fold",
        "dataset": "SEED-IV",
        "task": "seed_iv_label",
        "n_folds": 5,
        "label_counts": label_counts(df, label_col),
        "outputs": {
            "split_index": str(split_index_path),
            "all_split_assignments": str(all_splits_path),
            "registry": str(registry_path),
            "checks": str(checks_path),
            "summary": str(summary_path),
            "log": str(out_dir / "22A_seed_iv_subject_mixed_5fold_splits_log.txt"),
            "split_files_dir": str(split_dir),
        },
        "failed_checks": failed.to_dict(orient="records"),
        "scientific_statement": "These splits quantify subject-mixed within-dataset capacity only. They intentionally allow subject overlap across train, validation, and test, and must be interpreted as diagnostic upper-bound evidence rather than subject-independent deployment evidence.",
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Wrote split index: {split_index_path}")
    logger.info(f"Wrote registry: {registry_path}")
    logger.info(f"Overall Stage 22A passed: {overall_passed}")

    print("\nStage 22A outputs:")
    print(f"1. {split_index_path}")
    print(f"2. {all_splits_path}")
    print(f"3. {registry_path}")
    print(f"4. {checks_path}")
    print(f"5. {summary_path}")
    print(f"6. {out_dir / '22A_seed_iv_subject_mixed_5fold_splits_log.txt'}")

    if not overall_passed:
        logger.error("Stage 22A failed. Do not train until split integrity is fixed.")
        return 1
    logger.info("Stage 22A passed. It is safe to run Stage 22B training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
