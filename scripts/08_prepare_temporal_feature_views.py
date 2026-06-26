from __future__ import annotations

import argparse
import hashlib
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


def safe_name(value: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", str(value)).strip("_")


def rel_to_project(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


def npz_save(path: Path, compressed: bool, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **arrays)
    else:
        np.savez(path, **arrays)


def shape_text(arr: np.ndarray) -> str:
    return "x".join(str(x) for x in arr.shape)


def array_md5_shape_dtype(arr: np.ndarray) -> str:
    h = hashlib.md5()
    arr = np.ascontiguousarray(arr)
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.view(np.uint8))
    return h.hexdigest()


def clean_float(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def seed_eeg_to_tf(arr: np.ndarray) -> np.ndarray:
    arr = clean_float(arr)
    if arr.ndim == 3:
        shape = list(arr.shape)
        if shape[0] == 62 and shape[2] == 5:
            arr = np.moveaxis(arr, 1, 0)
            return arr.reshape(arr.shape[0], -1).astype(np.float32)
        if shape[0] == 62 and shape[1] == 5:
            arr = np.moveaxis(arr, 2, 0)
            return arr.reshape(arr.shape[0], -1).astype(np.float32)
        static_like = {5, 62}
        candidates = [i for i, s in enumerate(shape) if s not in static_like]
        time_axis = candidates[0] if candidates else int(np.argmax(shape))
        arr = np.moveaxis(arr, time_axis, 0)
        return arr.reshape(arr.shape[0], -1).astype(np.float32)
    if arr.ndim == 2:
        return arr.T.astype(np.float32) if arr.shape[0] == 62 else arr.astype(np.float32)
    return arr.reshape(1, -1).astype(np.float32)


def eye_to_tf(arr: np.ndarray) -> np.ndarray:
    arr = clean_float(arr)
    if arr.ndim == 2:
        if arr.shape[0] == 31:
            return arr.T.astype(np.float32)
        if arr.shape[1] == 31:
            return arr.astype(np.float32)
        return arr.astype(np.float32) if arr.shape[0] >= arr.shape[1] else arr.T.astype(np.float32)
    return arr.reshape(1, -1).astype(np.float32)


def raw_to_tc(arr: np.ndarray, expected_channels: int) -> np.ndarray:
    arr = clean_float(arr)
    if arr.ndim != 2:
        return arr.reshape(-1, expected_channels).astype(np.float32)

    if arr.shape[0] == expected_channels and arr.shape[1] != expected_channels:
        return arr.T.astype(np.float32)
    if arr.shape[1] == expected_channels:
        return arr.astype(np.float32)

    return arr.astype(np.float32) if arr.shape[0] >= arr.shape[1] else arr.T.astype(np.float32)


def channel_window_features(window: np.ndarray, features: List[str]) -> np.ndarray:
    vals = []
    for feat in features:
        if feat == "mean":
            vals.append(np.mean(window, axis=0))
        elif feat == "std":
            vals.append(np.std(window, axis=0))
        elif feat == "rms":
            vals.append(np.sqrt(np.mean(np.square(window), axis=0)))
        elif feat == "min":
            vals.append(np.min(window, axis=0))
        elif feat == "max":
            vals.append(np.max(window, axis=0))
        elif feat == "ptp":
            vals.append(np.ptp(window, axis=0))
        else:
            raise ValueError(f"Unsupported channel feature: {feat}")
    return np.concatenate(vals, axis=0).astype(np.float32)


def windowed_channel_features(
    arr_tc: np.ndarray,
    fs: int,
    window_sec: float,
    hop_sec: float,
    features: List[str],
) -> np.ndarray:
    arr_tc = clean_float(arr_tc)
    win = max(1, int(round(float(window_sec) * int(fs))))
    hop = max(1, int(round(float(hop_sec) * int(fs))))

    if arr_tc.shape[0] < win:
        return channel_window_features(arr_tc, features).reshape(1, -1)

    rows = []
    for start in range(0, arr_tc.shape[0] - win + 1, hop):
        rows.append(channel_window_features(arr_tc[start:start + win], features))

    if not rows:
        rows = [channel_window_features(arr_tc, features)]

    return np.stack(rows, axis=0).astype(np.float32)


def align_concat_time_features(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = clean_float(a)
    b = clean_float(b)

    target_t = min(a.shape[0], b.shape[0])
    if target_t <= 0:
        raise RuntimeError("Cannot align empty temporal feature arrays.")

    return np.concatenate([a[:target_t], b[:target_t]], axis=1).astype(np.float32)


def normalize_counts(series: pd.Series) -> Dict[str, int]:
    vc = series.value_counts(dropna=False).sort_index()
    out = {}
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


def process_seed(row: pd.Series, project_root: Path, out_dir: Path, compressed: bool) -> Dict[str, Any]:
    source_path = project_root / str(row["trial_store_file"])
    with np.load(source_path, allow_pickle=False) as npz:
        eeg_de = seed_eeg_to_tf(npz["eeg_de"])
        eeg_psd = seed_eeg_to_tf(npz["eeg_psd"])
        eye = eye_to_tf(npz["eye_features"])

    eeg_combined = align_concat_time_features(eeg_de, eeg_psd)
    eeg_eye = align_concat_time_features(eeg_combined, eye)

    out_path = out_dir / "temporal_store" / "SEED_IV" / f"{safe_name(row['trial_uid'])}.npz"
    npz_save(
        out_path,
        compressed,
        eeg_de=eeg_de,
        eeg_psd=eeg_psd,
        eeg_combined=eeg_combined,
        eye_features=eye,
        eeg_eye=eeg_eye,
        label_seed_iv=np.asarray(int(row["label_seed_iv"]), dtype=np.int64),
    )

    return {
        "temporal_view_file": rel_to_project(out_path, project_root),
        "primary_view_key": "eeg_eye",
        "eeg_de_shape": shape_text(eeg_de),
        "eeg_psd_shape": shape_text(eeg_psd),
        "eeg_combined_shape": shape_text(eeg_combined),
        "eye_features_shape": shape_text(eye),
        "primary_view_shape": shape_text(eeg_eye),
        "primary_view_md5": array_md5_shape_dtype(eeg_eye),
    }


def process_dreamer(row: pd.Series, project_root: Path, out_dir: Path, cfg: Dict[str, Any], compressed: bool) -> Dict[str, Any]:
    source_path = project_root / str(row["trial_store_file"])

    with np.load(source_path, allow_pickle=False) as npz:
        eeg = raw_to_tc(npz["eeg_stimulus"], expected_channels=14)
        ecg = raw_to_tc(npz["ecg_stimulus"], expected_channels=2)
        eeg_base = raw_to_tc(npz["eeg_baseline"], expected_channels=14) if "eeg_baseline" in npz and npz["eeg_baseline"].size else None
        ecg_base = raw_to_tc(npz["ecg_baseline"], expected_channels=2) if "ecg_baseline" in npz and npz["ecg_baseline"].size else None

    if bool(cfg["baseline_subtract_channel_mean"]):
        if eeg_base is not None and eeg_base.size:
            eeg = eeg - np.mean(eeg_base, axis=0, keepdims=True)
        if ecg_base is not None and ecg_base.size:
            ecg = ecg - np.mean(ecg_base, axis=0, keepdims=True)

    features = list(cfg["channel_features"])

    eeg_features = windowed_channel_features(
        eeg,
        fs=int(cfg["eeg_sampling_rate_hz"]),
        window_sec=float(cfg["window_sec"]),
        hop_sec=float(cfg["hop_sec"]),
        features=features,
    )

    ecg_features = windowed_channel_features(
        ecg,
        fs=int(cfg["ecg_sampling_rate_hz"]),
        window_sec=float(cfg["window_sec"]),
        hop_sec=float(cfg["hop_sec"]),
        features=features,
    )

    eeg_ecg = align_concat_time_features(eeg_features, ecg_features)

    out_path = out_dir / "temporal_store" / "DREAMER" / f"{safe_name(row['trial_uid'])}.npz"
    npz_save(
        out_path,
        compressed,
        eeg_temporal_features=eeg_features,
        ecg_temporal_features=ecg_features,
        eeg_ecg_temporal_features=eeg_ecg,
        valence_binary=np.asarray(int(row["valence_binary"]), dtype=np.int64),
        arousal_binary=np.asarray(int(row["arousal_binary"]), dtype=np.int64),
        dominance_binary=np.asarray(int(row["dominance_binary"]), dtype=np.int64),
    )

    return {
        "temporal_view_file": rel_to_project(out_path, project_root),
        "primary_view_key": "eeg_ecg_temporal_features",
        "dreamer_eeg_temporal_shape": shape_text(eeg_features),
        "dreamer_ecg_temporal_shape": shape_text(ecg_features),
        "primary_view_shape": shape_text(eeg_ecg),
        "primary_view_md5": array_md5_shape_dtype(eeg_ecg),
    }


def process_amigos(row: pd.Series, project_root: Path, out_dir: Path, compressed: bool) -> Dict[str, Any]:
    source_path = project_root / str(row["trial_store_file"])

    with np.load(source_path, allow_pickle=False) as npz:
        feature_bag = clean_float(npz["feature_bag"])
        raw_bag = clean_float(npz["eeg_ecg_gsr_window_bag"])
        raw_flat = raw_bag.reshape(raw_bag.shape[0], -1).astype(np.float32)

    out_path = out_dir / "temporal_store" / "AMIGOS" / f"{safe_name(row['trial_uid'])}.npz"
    npz_save(
        out_path,
        compressed,
        feature_bag=feature_bag,
        eeg_ecg_gsr_window_flat=raw_flat,
        valence_binary=np.asarray(int(row["valence_binary"]), dtype=np.int64),
        arousal_binary=np.asarray(int(row["arousal_binary"]), dtype=np.int64),
    )

    return {
        "temporal_view_file": rel_to_project(out_path, project_root),
        "primary_view_key": "feature_bag",
        "amigos_feature_bag_shape": shape_text(feature_bag),
        "amigos_raw_flat_shape": shape_text(raw_flat),
        "primary_view_shape": shape_text(feature_bag),
        "primary_view_md5": array_md5_shape_dtype(feature_bag),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="08_prepare_temporal_feature_views: deterministic temporal-view preparation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-paths", required=True)
    parser.add_argument("--temporal-config", default="configs/08_prepare_temporal_feature_views.yaml")
    args = parser.parse_args()

    t0 = time.time()

    _main_cfg = load_yaml(Path(args.config))
    local_paths = load_yaml(Path(args.local_paths))
    temporal_cfg = load_yaml(Path(args.temporal_config))["temporal_views"]

    project_root = as_path(local_paths["PROJECT_ROOT"])
    out_dir = project_root / temporal_cfg["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(out_dir / "08_prepare_temporal_feature_views_log.txt")
    logger.info("Starting 08_prepare_temporal_feature_views.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {out_dir}")

    req = temporal_cfg["required_previous_steps"]
    if req.get("require_passed", True):
        require_passed_json(project_root, req["model_ready_summary_json"], logger)
        require_passed_json(project_root, req["architecture_summary_json"], logger)

    compressed = bool(temporal_cfg.get("save_compressed_npz", True))
    index_path = project_root / temporal_cfg["inputs"]["unified_model_ready_index"]
    index_df = pd.read_csv(index_path)

    rows = []
    checks: List[Dict[str, Any]] = []

    for _, row in tqdm(index_df.iterrows(), total=len(index_df), desc="Preparing temporal views", unit="trial"):
        dataset = str(row["dataset"])
        base = row.to_dict()

        if dataset == "SEED-IV":
            update = process_seed(row, project_root, out_dir, compressed)
        elif dataset == "DREAMER":
            update = process_dreamer(row, project_root, out_dir, temporal_cfg["dreamer"], compressed)
        elif dataset == "AMIGOS":
            update = process_amigos(row, project_root, out_dir, compressed)
        else:
            raise RuntimeError(f"Unsupported dataset: {dataset}")

        base.update(update)
        rows.append(base)

    temporal_index = pd.DataFrame(rows)

    add_check(checks, "UNIFIED", "temporal rows", int(len(temporal_index)), int(len(index_df)))
    add_check(checks, "UNIFIED", "unique temporal view files", int(temporal_index["temporal_view_file"].nunique()), int(len(index_df)))
    add_check(checks, "UNIFIED", "missing temporal files", int(sum(not (project_root / p).exists() for p in temporal_index["temporal_view_file"])), 0)
    add_check(checks, "UNIFIED", "datasets", sorted(temporal_index["dataset"].unique().tolist()), ["AMIGOS", "DREAMER", "SEED-IV"])

    for dataset in ["SEED-IV", "DREAMER", "AMIGOS"]:
        d = temporal_index[temporal_index["dataset"] == dataset]
        add_check(checks, dataset, "rows", int(len(d)), int((index_df["dataset"] == dataset).sum()))
        add_check(checks, dataset, "missing primary view shape", int(d["primary_view_shape"].isna().sum()), 0)
        add_check(checks, dataset, "missing primary view md5", int(d["primary_view_md5"].isna().sum()), 0)

    checks_df = pd.DataFrame(checks)

    temporal_index_path = out_dir / "08_temporal_view_index.csv"
    checks_path = out_dir / "08_temporal_view_checks.csv"
    summary_path = out_dir / "08_temporal_view_summary.json"

    temporal_index.to_csv(temporal_index_path, index=False)
    checks_df.to_csv(checks_path, index=False)

    failed = checks_df[checks_df["passed"] == False]
    overall_passed = len(failed) == 0

    summary = {
        "name": temporal_cfg["name"],
        "created_at": now(),
        "overall_passed": bool(overall_passed),
        "elapsed_seconds": round(time.time() - t0, 3),
        "row_counts": {
            "temporal_rows": int(len(temporal_index)),
            "SEED-IV": int((temporal_index["dataset"] == "SEED-IV").sum()),
            "DREAMER": int((temporal_index["dataset"] == "DREAMER").sum()),
            "AMIGOS": int((temporal_index["dataset"] == "AMIGOS").sum()),
        },
        "outputs": {
            "temporal_view_index": str(temporal_index_path),
            "checks": str(checks_path),
            "summary": str(summary_path),
            "log": str(out_dir / "08_prepare_temporal_feature_views_log.txt"),
            "temporal_store": str(out_dir / "temporal_store"),
        },
        "failed_checks": failed.to_dict(orient="records"),
        "leakage_statement": "This stage creates deterministic per-trial temporal feature views only. It fits no scaler, selector, sampler, calibration model, classifier, or validation/test-dependent transformation.",
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Wrote temporal view index: {temporal_index_path}")
    logger.info(f"Wrote checks: {checks_path}")
    logger.info(f"Wrote summary: {summary_path}")
    logger.info(f"Overall temporal-view stage passed: {overall_passed}")
    logger.info(f"Elapsed seconds: {summary['elapsed_seconds']}")

    print("\nTARGETED OUTPUTS")
    print(f"1. {temporal_index_path}")
    print(f"2. {checks_path}")
    print(f"3. {summary_path}")
    print(f"4. {out_dir / '08_prepare_temporal_feature_views_log.txt'}")
    print(f"5. {out_dir / 'temporal_store'}")

    if not overall_passed:
        logger.error("Temporal-view stage failed. Do not proceed to training-loop design.")
        return 1

    logger.info("Temporal-view stage passed. It is safe to proceed to temporal data-module verification.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
