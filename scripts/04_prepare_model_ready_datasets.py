from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def safe_name(value: str) -> str:
    value = str(value)
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)
    value = value.strip("_")
    return value


def rel_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


def shape_text(arr: Any) -> str:
    return "x".join(str(x) for x in np.asarray(arr).shape)


def array_md5_shape_dtype(arr: np.ndarray) -> str:
    h = hashlib.md5()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(np.ascontiguousarray(arr).view(np.uint8))
    return h.hexdigest()


def clean_zip_members(zip_path: Path, suffix: str = ".mat") -> List[str]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            m for m in zf.namelist()
            if m.lower().endswith(suffix.lower())
            and "__macosx" not in m.lower()
            and not Path(m).name.startswith(".")
        ]
    return sorted(members)


def load_mat_from_zip(zip_path: Path, member: str) -> Dict[str, Any]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        payload = zf.read(member)
    return sio.loadmat(BytesIO(payload), simplify_cells=True)


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


def npz_save(path: Path, compressed: bool, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def find_npz_array_key(npz_obj: Any, preferred: List[str], ndim: int) -> str:
    keys = list(npz_obj.keys())
    for key in preferred:
        if key in keys and np.asarray(npz_obj[key]).ndim == ndim:
            return key
    candidates = [(key, np.asarray(npz_obj[key]).size) for key in keys if np.asarray(npz_obj[key]).ndim == ndim]
    if not candidates:
        raise RuntimeError(f"No {ndim}D array found. Available keys={keys}")
    return sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]


def normalize_counts(series: pd.Series) -> Dict[str, int]:
    vc = series.value_counts(dropna=False).sort_index()
    out: Dict[str, int] = {}
    for k, v in vc.items():
        if pd.isna(k):
            out["NaN"] = int(v)
        else:
            try:
                out[str(int(float(k)))] = int(v)
            except Exception:
                out[str(k)] = int(v)
    return out


def add_check(checks: List[Dict[str, Any]], dataset: str, check: str, observed: Any, expected: Any, passed: Optional[bool] = None) -> None:
    if passed is None:
        passed = observed == expected
    checks.append({
        "dataset": dataset,
        "check": check,
        "observed": json.dumps(observed, ensure_ascii=False),
        "expected": json.dumps(expected, ensure_ascii=False),
        "passed": bool(passed),
    })


def seed_iv_prepare(
    project_root: Path,
    dataset_root: Path,
    manifest: pd.DataFrame,
    cfg: Dict[str, Any],
    out_dir: Path,
    compressed: bool,
    logger: Logger,
    checks: List[Dict[str, Any]],
) -> pd.DataFrame:
    logger.info("Preparing SEED-IV model-ready trial store.")

    eeg_zip = dataset_root / cfg["eeg_feature_zip"]
    eye_zip = dataset_root / cfg["eye_feature_zip"]
    store_dir = out_dir / "trial_store" / "SEED_IV"
    rows: List[Dict[str, Any]] = []

    grouped = manifest.groupby(["source_member_eeg_feature", "source_member_eye_feature"], sort=True)

    for (eeg_member, eye_member), group in tqdm(grouped, desc="SEED-IV trial-store files", unit="file-pair"):
        eeg_mat = load_mat_from_zip(eeg_zip, str(eeg_member))
        eye_mat = load_mat_from_zip(eye_zip, str(eye_member))

        for _, row in tqdm(group.iterrows(), total=len(group), desc="SEED-IV trials", unit="trial", leave=False):
            trial_uid = str(row["trial_uid"])
            de_key = str(row["eeg_de_key"])
            psd_key = str(row["eeg_psd_key"])
            eye_key = str(row["eye_key"])

            eeg_de = np.asarray(eeg_mat[de_key], dtype=np.float32)
            eeg_psd = np.asarray(eeg_mat[psd_key], dtype=np.float32)
            eye_features = np.asarray(eye_mat[eye_key], dtype=np.float32)

            output_path = store_dir / f"{safe_name(trial_uid)}.npz"

            npz_save(
                output_path,
                compressed,
                eeg_de=eeg_de,
                eeg_psd=eeg_psd,
                eye_features=eye_features,
                label_seed_iv=np.asarray(row["seed_iv_label"], dtype=np.int64),
                subject_index=np.asarray(row["subject_index"], dtype=np.int64),
                session_index=np.asarray(row["session_index"], dtype=np.int64),
                trial_index=np.asarray(row["trial_index"], dtype=np.int64),
            )

            rows.append({
                "dataset": "SEED-IV",
                "trial_uid": trial_uid,
                "subject_id": row["subject_id"],
                "subject_index": int(row["subject_index"]),
                "session_id": row["session_id"],
                "session_index": int(row["session_index"]),
                "trial_id": row["trial_id"],
                "trial_index": int(row["trial_index"]),
                "label_seed_iv": int(row["seed_iv_label"]),
                "modalities_available": "EEG|EYE",
                "has_eeg": 1,
                "has_eye": 1,
                "has_ecg": 0,
                "has_gsr": 0,
                "trial_store_file": rel_to_project(output_path, project_root),
                "eeg_de_shape": shape_text(eeg_de),
                "eeg_psd_shape": shape_text(eeg_psd),
                "eye_features_shape": shape_text(eye_features),
                "eeg_de_md5": array_md5_shape_dtype(eeg_de),
                "eeg_psd_md5": array_md5_shape_dtype(eeg_psd),
                "eye_features_md5": array_md5_shape_dtype(eye_features),
                "source_member_eeg_feature": eeg_member,
                "source_member_eye_feature": eye_member,
            })

    df = pd.DataFrame(rows)

    add_check(checks, "SEED-IV", "prepared rows", int(len(df)), int(cfg["expected"]["rows"]))
    add_check(checks, "SEED-IV", "unique trial store files", int(df["trial_store_file"].nunique()), int(cfg["expected"]["rows"]))
    add_check(checks, "SEED-IV", "missing output files", int(sum(not (project_root / p).exists() for p in df["trial_store_file"])), 0)
    add_check(checks, "SEED-IV", "class counts", normalize_counts(df["label_seed_iv"]), {"0": 270, "1": 270, "2": 270, "3": 270})
    add_check(checks, "SEED-IV", "missing EEG DE shape", int(df["eeg_de_shape"].isna().sum()), 0)
    add_check(checks, "SEED-IV", "missing EEG PSD shape", int(df["eeg_psd_shape"].isna().sum()), 0)
    add_check(checks, "SEED-IV", "missing eye shape", int(df["eye_features_shape"].isna().sum()), 0)

    logger.info(f"SEED-IV prepared rows: {len(df)}")
    return df


def dreamer_prepare(
    project_root: Path,
    dataset_root: Path,
    manifest: pd.DataFrame,
    cfg: Dict[str, Any],
    out_dir: Path,
    compressed: bool,
    logger: Logger,
    checks: List[Dict[str, Any]],
) -> pd.DataFrame:
    logger.info("Preparing DREAMER model-ready trial store.")

    mat_path = dataset_root / cfg["mat_file"]
    mat = sio.loadmat(mat_path, simplify_cells=True)
    dreamer = mat.get("DREAMER")
    subjects = as_list(get_field(dreamer, "Data"))
    store_dir = out_dir / "trial_store" / "DREAMER"

    rows: List[Dict[str, Any]] = []

    manifest_by_uid = {str(r["trial_uid"]): r for _, r in manifest.iterrows()}

    for subj_idx, subj in enumerate(tqdm(subjects, desc="DREAMER subjects", unit="subject"), start=1):
        valence = np.asarray(get_field(subj, "ScoreValence")).reshape(-1)
        arousal = np.asarray(get_field(subj, "ScoreArousal")).reshape(-1)
        dominance = np.asarray(get_field(subj, "ScoreDominance")).reshape(-1)

        eeg = get_field(subj, "EEG")
        ecg = get_field(subj, "ECG")

        eeg_stimuli = as_list(get_field(eeg, "stimuli"))
        ecg_stimuli = as_list(get_field(ecg, "stimuli"))
        eeg_baseline = as_list(get_field(eeg, "baseline"))
        ecg_baseline = as_list(get_field(ecg, "baseline"))

        for trial_i in tqdm(range(len(valence)), desc="DREAMER trials", unit="trial", leave=False):
            trial_uid = f"DREAMER__S{subj_idx:02d}__T{trial_i + 1:02d}"
            meta = manifest_by_uid[trial_uid]

            eeg_stim = np.asarray(eeg_stimuli[trial_i], dtype=np.float32)
            ecg_stim = np.asarray(ecg_stimuli[trial_i], dtype=np.float32)

            eeg_base = np.asarray(eeg_baseline[trial_i], dtype=np.float32) if trial_i < len(eeg_baseline) else np.asarray([], dtype=np.float32)
            ecg_base = np.asarray(ecg_baseline[trial_i], dtype=np.float32) if trial_i < len(ecg_baseline) else np.asarray([], dtype=np.float32)

            output_path = store_dir / f"{safe_name(trial_uid)}.npz"

            npz_save(
                output_path,
                compressed,
                eeg_stimulus=eeg_stim,
                ecg_stimulus=ecg_stim,
                eeg_baseline=eeg_base,
                ecg_baseline=ecg_base,
                valence_score=np.asarray(valence[trial_i], dtype=np.float32),
                arousal_score=np.asarray(arousal[trial_i], dtype=np.float32),
                dominance_score=np.asarray(dominance[trial_i], dtype=np.float32),
                valence_binary=np.asarray(int(meta["valence_binary"]), dtype=np.int64),
                arousal_binary=np.asarray(int(meta["arousal_binary"]), dtype=np.int64),
                dominance_binary=np.asarray(int(meta["dominance_binary"]), dtype=np.int64),
                subject_index=np.asarray(subj_idx, dtype=np.int64),
                trial_index=np.asarray(trial_i + 1, dtype=np.int64),
            )

            rows.append({
                "dataset": "DREAMER",
                "trial_uid": trial_uid,
                "subject_id": meta["subject_id"],
                "subject_index": int(meta["subject_index"]),
                "session_id": "NA",
                "session_index": np.nan,
                "trial_id": meta["trial_id"],
                "trial_index": int(meta["trial_index"]),
                "valence_score": float(valence[trial_i]),
                "arousal_score": float(arousal[trial_i]),
                "dominance_score": float(dominance[trial_i]),
                "valence_binary": int(meta["valence_binary"]),
                "arousal_binary": int(meta["arousal_binary"]),
                "dominance_binary": int(meta["dominance_binary"]),
                "modalities_available": "EEG|ECG",
                "has_eeg": 1,
                "has_eye": 0,
                "has_ecg": 1,
                "has_gsr": 0,
                "trial_store_file": rel_to_project(output_path, project_root),
                "dreamer_eeg_stimulus_shape": shape_text(eeg_stim),
                "dreamer_ecg_stimulus_shape": shape_text(ecg_stim),
                "dreamer_eeg_baseline_shape": shape_text(eeg_base),
                "dreamer_ecg_baseline_shape": shape_text(ecg_base),
                "dreamer_eeg_stimulus_md5": array_md5_shape_dtype(eeg_stim),
                "dreamer_ecg_stimulus_md5": array_md5_shape_dtype(ecg_stim),
            })

    df = pd.DataFrame(rows)

    add_check(checks, "DREAMER", "prepared rows", int(len(df)), int(cfg["expected"]["rows"]))
    add_check(checks, "DREAMER", "unique trial store files", int(df["trial_store_file"].nunique()), int(cfg["expected"]["rows"]))
    add_check(checks, "DREAMER", "missing output files", int(sum(not (project_root / p).exists() for p in df["trial_store_file"])), 0)
    add_check(checks, "DREAMER", "valence counts", normalize_counts(df["valence_binary"]), {"0": 251, "1": 163})
    add_check(checks, "DREAMER", "arousal counts", normalize_counts(df["arousal_binary"]), {"0": 233, "1": 181})
    add_check(checks, "DREAMER", "dominance counts", normalize_counts(df["dominance_binary"]), {"0": 215, "1": 199})
    add_check(checks, "DREAMER", "missing EEG stimulus shape", int(df["dreamer_eeg_stimulus_shape"].isna().sum()), 0)
    add_check(checks, "DREAMER", "missing ECG stimulus shape", int(df["dreamer_ecg_stimulus_shape"].isna().sum()), 0)

    logger.info(f"DREAMER prepared rows: {len(df)}")
    return df


def amigos_prepare(
    project_root: Path,
    dataset_root: Path,
    manifest: pd.DataFrame,
    cfg: Dict[str, Any],
    out_dir: Path,
    compressed: bool,
    logger: Logger,
    checks: List[Dict[str, Any]],
) -> pd.DataFrame:
    logger.info("Preparing AMIGOS model-ready trial store.")

    trial_bag_path = dataset_root / cfg["trial_bag_file"]
    feature_bag_path = dataset_root / cfg["feature_bag_file"]
    feature_names_path = dataset_root / cfg["feature_names_file"]

    with np.load(trial_bag_path, allow_pickle=True) as trial_npz:
        trial_key = find_npz_array_key(trial_npz, ["X_bags", "X", "data"], ndim=4)
        trial_bags = np.asarray(trial_npz[trial_key], dtype=np.float32)

    with np.load(feature_bag_path, allow_pickle=True) as feature_npz:
        feature_key = find_npz_array_key(feature_npz, ["F_bags", "X_features", "features"], ndim=3)
        feature_bags = np.asarray(feature_npz[feature_key], dtype=np.float32)

    feature_names = pd.read_csv(feature_names_path)

    store_dir = out_dir / "trial_store" / "AMIGOS"
    rows: List[Dict[str, Any]] = []

    if list(trial_bags.shape) != cfg["expected"]["trial_bag_shape"]:
        raise RuntimeError(f"AMIGOS trial bag shape mismatch: {trial_bags.shape}")
    if list(feature_bags.shape) != cfg["expected"]["feature_bag_shape"]:
        raise RuntimeError(f"AMIGOS feature bag shape mismatch: {feature_bags.shape}")

    for idx, row in tqdm(manifest.iterrows(), total=len(manifest), desc="AMIGOS trials", unit="trial"):
        trial_uid = str(row["trial_uid"])
        trial_bag = trial_bags[idx]
        feature_bag = feature_bags[idx]

        output_path = store_dir / f"{safe_name(trial_uid)}.npz"

        npz_save(
            output_path,
            compressed,
            eeg_ecg_gsr_window_bag=trial_bag,
            feature_bag=feature_bag,
            valence_binary=np.asarray(int(row["valence_binary"]), dtype=np.int64),
            arousal_binary=np.asarray(int(row["arousal_binary"]), dtype=np.int64),
            subject_index=np.asarray(int(row["subject_index"]), dtype=np.int64),
            trial_index=np.asarray(int(row["trial_index"]), dtype=np.int64),
        )

        rows.append({
            "dataset": "AMIGOS",
            "trial_uid": trial_uid,
            "subject_id": row["subject_id"],
            "subject_index": int(row["subject_index"]),
            "session_id": "NA",
            "session_index": np.nan,
            "trial_id": row["trial_id"],
            "trial_index": int(row["trial_index"]),
            "valence_binary": int(row["valence_binary"]),
            "arousal_binary": int(row["arousal_binary"]),
            "modalities_available": "EEG|ECG|GSR",
            "has_eeg": 1,
            "has_eye": 0,
            "has_ecg": 1,
            "has_gsr": 1,
            "trial_store_file": rel_to_project(output_path, project_root),
            "amigos_trial_bag_shape": shape_text(trial_bag),
            "amigos_feature_bag_shape": shape_text(feature_bag),
            "amigos_trial_bag_md5": array_md5_shape_dtype(trial_bag),
            "amigos_feature_bag_md5": array_md5_shape_dtype(feature_bag),
            "feature_name_count": int(len(feature_names)),
            "source_trial_bag_key": trial_key,
            "source_feature_bag_key": feature_key,
        })

    df = pd.DataFrame(rows)

    add_check(checks, "AMIGOS", "prepared rows", int(len(df)), int(cfg["expected"]["rows"]))
    add_check(checks, "AMIGOS", "unique trial store files", int(df["trial_store_file"].nunique()), int(cfg["expected"]["rows"]))
    add_check(checks, "AMIGOS", "missing output files", int(sum(not (project_root / p).exists() for p in df["trial_store_file"])), 0)
    add_check(checks, "AMIGOS", "valence counts", normalize_counts(df["valence_binary"]), {"0": 97, "1": 143})
    add_check(checks, "AMIGOS", "arousal counts", normalize_counts(df["arousal_binary"]), {"0": 66, "1": 174})
    add_check(checks, "AMIGOS", "feature name count", int(len(feature_names)), int(cfg["expected"]["feature_bag_shape"][-1]))
    add_check(checks, "AMIGOS", "trial bag global shape", list(trial_bags.shape), cfg["expected"]["trial_bag_shape"])
    add_check(checks, "AMIGOS", "feature bag global shape", list(feature_bags.shape), cfg["expected"]["feature_bag_shape"])

    logger.info(f"AMIGOS prepared rows: {len(df)}")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="04_prepare_model_ready_datasets: leakage-safe trial-store preparation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-paths", required=True)
    parser.add_argument("--prepare-config", default="configs/04_prepare_model_ready_datasets.yaml")
    args = parser.parse_args()

    t0 = time.time()

    _main_cfg = load_yaml(Path(args.config))
    local_paths = load_yaml(Path(args.local_paths))
    prepare_cfg = load_yaml(Path(args.prepare_config))["prepare"]

    project_root = as_path(local_paths["PROJECT_ROOT"])
    out_dir = project_root / prepare_cfg["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(out_dir / "04_prepare_model_ready_datasets_log.txt")
    logger.info("Starting 04_prepare_model_ready_datasets.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {out_dir}")

    req = prepare_cfg["required_previous_steps"]
    if req.get("require_passed", True):
        require_passed_json(project_root, req["manifest_summary_json"], logger)
        require_passed_json(project_root, req["split_summary_json"], logger)

    compressed = bool(prepare_cfg.get("save_compressed_npz", True))
    logger.info(f"Save compressed NPZ: {compressed}")

    manifests_cfg = prepare_cfg["manifests"]
    seed_manifest = pd.read_csv(project_root / manifests_cfg["seed_iv"])
    dreamer_manifest = pd.read_csv(project_root / manifests_cfg["dreamer"])
    amigos_manifest = pd.read_csv(project_root / manifests_cfg["amigos"])

    checks: List[Dict[str, Any]] = []

    seed_root = as_path(local_paths[prepare_cfg["seed_iv"]["root_key"]])
    dreamer_root = as_path(local_paths[prepare_cfg["dreamer"]["root_key"]])
    amigos_root = as_path(local_paths[prepare_cfg["amigos"]["root_key"]])

    seed_index = seed_iv_prepare(project_root, seed_root, seed_manifest, prepare_cfg["seed_iv"], out_dir, compressed, logger, checks)
    dreamer_index = dreamer_prepare(project_root, dreamer_root, dreamer_manifest, prepare_cfg["dreamer"], out_dir, compressed, logger, checks)
    amigos_index = amigos_prepare(project_root, amigos_root, amigos_manifest, prepare_cfg["amigos"], out_dir, compressed, logger, checks)

    unified_index = pd.concat([seed_index, dreamer_index, amigos_index], ignore_index=True, sort=False)

    add_check(checks, "UNIFIED", "prepared rows", int(len(unified_index)), int(len(seed_index) + len(dreamer_index) + len(amigos_index)))
    add_check(checks, "UNIFIED", "datasets", sorted(unified_index["dataset"].unique().tolist()), ["AMIGOS", "DREAMER", "SEED-IV"])
    add_check(checks, "UNIFIED", "unique trial uids", int(unified_index["trial_uid"].nunique()), int(len(unified_index)))

    seed_path = out_dir / "04_seed_iv_model_ready_index.csv"
    dreamer_path = out_dir / "04_dreamer_model_ready_index.csv"
    amigos_path = out_dir / "04_amigos_model_ready_index.csv"
    unified_path = out_dir / "04_unified_model_ready_index.csv"
    checks_path = out_dir / "04_model_ready_checks.csv"
    summary_path = out_dir / "04_model_ready_summary.json"

    seed_index.to_csv(seed_path, index=False)
    dreamer_index.to_csv(dreamer_path, index=False)
    amigos_index.to_csv(amigos_path, index=False)
    unified_index.to_csv(unified_path, index=False)

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(checks_path, index=False)

    failed = checks_df[checks_df["passed"] == False]
    overall_passed = len(failed) == 0

    summary = {
        "name": prepare_cfg["name"],
        "created_at": now(),
        "overall_passed": bool(overall_passed),
        "elapsed_seconds": round(time.time() - t0, 3),
        "outputs": {
            "seed_iv_index": str(seed_path),
            "dreamer_index": str(dreamer_path),
            "amigos_index": str(amigos_path),
            "unified_index": str(unified_path),
            "checks": str(checks_path),
            "summary": str(summary_path),
            "log": str(out_dir / "04_prepare_model_ready_datasets_log.txt"),
            "trial_store_dir": str(out_dir / "trial_store"),
        },
        "row_counts": {
            "SEED-IV": int(len(seed_index)),
            "DREAMER": int(len(dreamer_index)),
            "AMIGOS": int(len(amigos_index)),
            "UNIFIED": int(len(unified_index)),
        },
        "failed_checks": failed.to_dict(orient="records"),
        "leakage_statement": "This stage only copies/raw-serializes per-trial modality arrays and labels into traceable NPZ files. It fits no scaler, selector, sampler, calibration, or model on any split.",
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Wrote SEED-IV model-ready index: {seed_path}")
    logger.info(f"Wrote DREAMER model-ready index: {dreamer_path}")
    logger.info(f"Wrote AMIGOS model-ready index: {amigos_path}")
    logger.info(f"Wrote unified model-ready index: {unified_path}")
    logger.info(f"Wrote checks: {checks_path}")
    logger.info(f"Wrote summary: {summary_path}")
    logger.info(f"Overall model-ready preparation passed: {overall_passed}")
    logger.info(f"Elapsed seconds: {summary['elapsed_seconds']}")

    print("\nTARGETED OUTPUTS")
    print(f"1. {seed_path}")
    print(f"2. {dreamer_path}")
    print(f"3. {amigos_path}")
    print(f"4. {unified_path}")
    print(f"5. {checks_path}")
    print(f"6. {summary_path}")
    print(f"7. {out_dir / '04_prepare_model_ready_datasets_log.txt'}")
    print(f"8. {out_dir / 'trial_store'}")

    if not overall_passed:
        logger.error("Model-ready preparation failed. Do not proceed to loaders or training.")
        return 1

    logger.info("Model-ready preparation passed. It is safe to proceed to loader verification stage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

