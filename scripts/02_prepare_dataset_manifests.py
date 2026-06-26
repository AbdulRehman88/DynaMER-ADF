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


def public_keys(mat: Dict[str, Any]) -> List[str]:
    return sorted([k for k in mat.keys() if not k.startswith("__")])


def shape_or_none(x: Any) -> Optional[str]:
    if x is None:
        return None
    try:
        return "x".join(str(v) for v in np.asarray(x).shape)
    except Exception:
        return None


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


def binary_from_score(score: Any, threshold: float = 3.0) -> Optional[int]:
    if pd.isna(score):
        return None
    return int(float(score) > threshold)


def normalize_counts(counts: Dict[Any, Any]) -> Dict[int, int]:
    out = {}
    for k, v in counts.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            pass
    return dict(sorted(out.items()))


def add_check(checks: List[Dict[str, Any]], dataset: str, check: str, observed: Any, expected: Any) -> None:
    passed = observed == expected
    checks.append({
        "dataset": dataset,
        "check": check,
        "observed": json.dumps(observed, ensure_ascii=False),
        "expected": json.dumps(expected, ensure_ascii=False),
        "passed": bool(passed),
    })


def require_previous_audit(project_root: Path, cfg: Dict[str, Any], logger: Logger) -> None:
    prev = cfg["required_previous_audit"]
    audit_path = project_root / prev["summary_json"]
    if not audit_path.exists():
        raise FileNotFoundError(f"Required previous audit not found: {audit_path}")
    data = json.loads(audit_path.read_text(encoding="utf-8"))
    passed = bool(data.get("overall_passed", False))
    logger.info(f"Previous audit found: {audit_path}")
    logger.info(f"Previous audit passed: {passed}")
    if prev.get("require_passed", True) and not passed:
        raise RuntimeError("Previous audit did not pass. Refusing to prepare manifests.")


def infer_seed_file_mapping(members: List[str], expected_subjects: int, expected_sessions: int) -> Dict[str, Tuple[str, int, int]]:
    parsed = []
    for member in members:
        stem = Path(member).stem
        nums = re.findall(r"\d+", stem)
        subj_num = int(nums[0]) if nums else None
        parsed.append({"member": member, "stem": stem, "subject_num": subj_num})

    groups: Dict[int, List[str]] = {}
    for item in parsed:
        if item["subject_num"] is not None:
            groups.setdefault(item["subject_num"], []).append(item["member"])

    mapping: Dict[str, Tuple[str, int, int]] = {}

    valid_grouping = len(groups) == expected_subjects and all(len(v) == expected_sessions for v in groups.values())

    if valid_grouping:
        for subj_num in sorted(groups):
            for sess_index, member in enumerate(sorted(groups[subj_num]), start=1):
                subject_id = f"SEED_S{subj_num:02d}"
                mapping[member] = (subject_id, subj_num, sess_index)
        return mapping

    sorted_members = sorted(members)
    expected_total = expected_subjects * expected_sessions
    if len(sorted_members) != expected_total:
        raise RuntimeError(
            f"Cannot infer SEED-IV subject/session mapping. Found {len(sorted_members)} files; expected {expected_total}."
        )

    idx = 0
    for subj_num in range(1, expected_subjects + 1):
        for sess_index in range(1, expected_sessions + 1):
            member = sorted_members[idx]
            mapping[member] = (f"SEED_S{subj_num:02d}", subj_num, sess_index)
            idx += 1
    return mapping


def extract_seed_session_labels_from_xlsx(xlsx_path: Path, expected_trials: int) -> Tuple[Dict[int, List[int]], Dict[int, str]]:
    try:
        sheets = pd.read_excel(xlsx_path, sheet_name=None)
    except Exception as exc:
        raise RuntimeError(f"Could not read SEED-IV stimulation XLSX: {exc}")

    session_labels: Dict[int, List[int]] = {}
    label_sources: Dict[int, str] = {}

    for sheet_name, df in sheets.items():
        sheet_lower = str(sheet_name).lower()
        session_match = re.search(r"experiment\s*(\d+)", sheet_lower)
        if not session_match:
            continue

        session_idx = int(session_match.group(1))
        if session_idx not in {1, 2, 3}:
            continue

        if "Label" not in df.columns:
            raise RuntimeError(f"SEED-IV sheet '{sheet_name}' does not contain a Label column.")

        labels = pd.to_numeric(df["Label"], errors="coerce").dropna().astype(int).tolist()
        labels = labels[:expected_trials]

        if len(labels) != expected_trials:
            raise RuntimeError(
                f"SEED-IV sheet '{sheet_name}' has {len(labels)} valid labels, expected {expected_trials}."
            )

        if not set(labels).issubset({0, 1, 2, 3}):
            raise RuntimeError(f"SEED-IV sheet '{sheet_name}' contains labels outside 0,1,2,3: {sorted(set(labels))}")

        counts = pd.Series(labels).value_counts().to_dict()
        if any(counts.get(k, 0) != 6 for k in [0, 1, 2, 3]):
            raise RuntimeError(
                f"SEED-IV sheet '{sheet_name}' is not balanced 6/class. Counts={counts}"
            )

        session_labels[session_idx] = labels
        label_sources[session_idx] = f"xlsx:{sheet_name}:Label"

    if sorted(session_labels.keys()) != [1, 2, 3]:
        raise RuntimeError(
            f"Could not extract session-specific SEED-IV labels for experiments 1,2,3. Found sessions={sorted(session_labels.keys())}"
        )

    return session_labels, label_sources

def build_seed_manifest(root: Path, cfg: Dict[str, Any], logger: Logger, checks: List[Dict[str, Any]]) -> pd.DataFrame:
    logger.info("Building SEED-IV trial manifest.")
    eeg_zip = root / cfg["eeg_feature_zip"]
    eye_zip = root / cfg["eye_feature_zip"]
    stimulation = root / cfg["stimulation_file"]
    expected = cfg["expected"]

    session_labels, session_label_sources = extract_seed_session_labels_from_xlsx(
        stimulation,
        expected["trials_per_subject_session"],
    )
    logger.info("Using session-specific SEED-IV labels from stimulation XLSX experiments 1, 2, and 3.")

    eeg_members = clean_zip_members(eeg_zip)
    eye_members = clean_zip_members(eye_zip)

    eeg_mapping = infer_seed_file_mapping(eeg_members, expected["subjects"], expected["sessions"])
    eye_mapping = infer_seed_file_mapping(eye_members, expected["subjects"], expected["sessions"])

    eye_by_subject_session = {
        (subj_num, sess_num): member
        for member, (_, subj_num, sess_num) in eye_mapping.items()
    }

    rows: List[Dict[str, Any]] = []
    global_idx = 0

    for eeg_member in tqdm(sorted(eeg_members), desc="SEED-IV manifest", unit="file"):
        subject_id, subj_num, sess_num = eeg_mapping[eeg_member]
        eye_member = eye_by_subject_session.get((subj_num, sess_num))
        if eye_member is None:
            raise RuntimeError(f"Missing paired eye feature file for subject={subj_num}, session={sess_num}")

        eeg_mat = load_mat_from_zip(eeg_zip, eeg_member)
        eye_mat = load_mat_from_zip(eye_zip, eye_member)

        for trial_id in range(1, expected["trials_per_subject_session"] + 1):
            global_idx += 1
            de_key = f"de_LDS{trial_id}"
            psd_key = f"psd_LDS{trial_id}"
            eye_key = f"eye_{trial_id}"

            de_shape = shape_or_none(eeg_mat.get(de_key))
            psd_shape = shape_or_none(eeg_mat.get(psd_key))
            eye_shape = shape_or_none(eye_mat.get(eye_key))

            rows.append({
                "dataset": "SEED-IV",
                "trial_uid": f"SEED-IV__S{subj_num:02d}__SES{sess_num:02d}__T{trial_id:02d}",
                "subject_id": subject_id,
                "subject_index": subj_num,
                "session_id": f"SES{sess_num:02d}",
                "session_index": sess_num,
                "trial_id": f"T{trial_id:02d}",
                "trial_index": trial_id,
                "global_index_within_dataset": global_idx,
                "task_family": "four_class_emotion",
                "seed_iv_label": int(session_labels[sess_num][trial_id - 1]),
                "valence_score": np.nan,
                "arousal_score": np.nan,
                "dominance_score": np.nan,
                "valence_binary": np.nan,
                "arousal_binary": np.nan,
                "dominance_binary": np.nan,
                "modalities_available": "EEG|EYE",
                "has_eeg": 1,
                "has_eye": 1,
                "has_ecg": 0,
                "has_gsr": 0,
                "source_label": session_label_sources[sess_num],
                "source_file_primary": str(eeg_zip),
                "source_member_eeg_feature": eeg_member,
                "source_member_eye_feature": eye_member,
                "eeg_de_key": de_key,
                "eeg_psd_key": psd_key,
                "eye_key": eye_key,
                "eeg_de_shape": de_shape,
                "eeg_psd_shape": psd_shape,
                "eye_shape": eye_shape,
            })

    df = pd.DataFrame(rows)
    class_counts = normalize_counts(df["seed_iv_label"].value_counts().to_dict())

    add_check(checks, "SEED-IV", "manifest rows", int(len(df)), expected["rows"])
    add_check(checks, "SEED-IV", "subjects", int(df["subject_id"].nunique()), expected["subjects"])
    add_check(checks, "SEED-IV", "sessions", int(df["session_index"].nunique()), expected["sessions"])
    add_check(checks, "SEED-IV", "class counts", class_counts, normalize_counts(expected["class_counts"]))
    add_check(checks, "SEED-IV", "missing EEG DE shapes", int(df["eeg_de_shape"].isna().sum()), 0)
    add_check(checks, "SEED-IV", "missing EEG PSD shapes", int(df["eeg_psd_shape"].isna().sum()), 0)
    add_check(checks, "SEED-IV", "missing eye shapes", int(df["eye_shape"].isna().sum()), 0)

    logger.info(f"SEED-IV manifest rows: {len(df)}")
    return df


def build_dreamer_manifest(root: Path, cfg: Dict[str, Any], logger: Logger, checks: List[Dict[str, Any]]) -> pd.DataFrame:
    logger.info("Building DREAMER trial manifest.")
    expected = cfg["expected"]
    mat_path = root / cfg["mat_file"]
    mat = sio.loadmat(mat_path, simplify_cells=True)
    dreamer = mat.get("DREAMER")
    subjects = as_list(get_field(dreamer, "Data"))

    rows: List[Dict[str, Any]] = []
    global_idx = 0

    for subj_idx, subj in enumerate(tqdm(subjects, desc="DREAMER manifest", unit="subject"), start=1):
        valence = np.asarray(get_field(subj, "ScoreValence")).reshape(-1)
        arousal = np.asarray(get_field(subj, "ScoreArousal")).reshape(-1)
        dominance = np.asarray(get_field(subj, "ScoreDominance")).reshape(-1)

        eeg = get_field(subj, "EEG")
        ecg = get_field(subj, "ECG")
        eeg_stimuli = as_list(get_field(eeg, "stimuli"))
        ecg_stimuli = as_list(get_field(ecg, "stimuli"))

        n_trials = len(valence)
        if n_trials != expected["trials_per_subject"]:
            raise RuntimeError(f"DREAMER subject {subj_idx} has {n_trials} trials, expected {expected['trials_per_subject']}.")

        for trial_i in range(n_trials):
            global_idx += 1

            eeg_arr = np.asarray(eeg_stimuli[trial_i]) if trial_i < len(eeg_stimuli) else None
            ecg_arr = np.asarray(ecg_stimuli[trial_i]) if trial_i < len(ecg_stimuli) else None

            eeg_shape = shape_or_none(eeg_arr)
            ecg_shape = shape_or_none(ecg_arr)

            rows.append({
                "dataset": "DREAMER",
                "trial_uid": f"DREAMER__S{subj_idx:02d}__T{trial_i + 1:02d}",
                "subject_id": f"DREAMER_S{subj_idx:02d}",
                "subject_index": subj_idx,
                "session_id": "NA",
                "session_index": np.nan,
                "trial_id": f"T{trial_i + 1:02d}",
                "trial_index": trial_i + 1,
                "global_index_within_dataset": global_idx,
                "task_family": "binary_valence_arousal_dominance",
                "seed_iv_label": np.nan,
                "valence_score": float(valence[trial_i]),
                "arousal_score": float(arousal[trial_i]),
                "dominance_score": float(dominance[trial_i]),
                "valence_binary": binary_from_score(valence[trial_i]),
                "arousal_binary": binary_from_score(arousal[trial_i]),
                "dominance_binary": binary_from_score(dominance[trial_i]),
                "modalities_available": "EEG|ECG",
                "has_eeg": 1,
                "has_eye": 0,
                "has_ecg": 1,
                "has_gsr": 0,
                "source_label": "DREAMER.mat:ScoreValence/ScoreArousal/ScoreDominance:binary_gt_3",
                "source_file_primary": str(mat_path),
                "source_member_eeg_feature": "",
                "source_member_eye_feature": "",
                "eeg_de_key": "",
                "eeg_psd_key": "",
                "eye_key": "",
                "eeg_de_shape": "",
                "eeg_psd_shape": "",
                "eye_shape": "",
                "dreamer_eeg_shape": eeg_shape,
                "dreamer_ecg_shape": ecg_shape,
            })

    df = pd.DataFrame(rows)

    add_check(checks, "DREAMER", "manifest rows", int(len(df)), expected["rows"])
    add_check(checks, "DREAMER", "subjects", int(df["subject_id"].nunique()), expected["subjects"])
    for col, exp_counts in expected["label_counts"].items():
        obs = normalize_counts(df[col].astype(int).value_counts().to_dict())
        add_check(checks, "DREAMER", f"{col} counts", obs, normalize_counts(exp_counts))
    add_check(checks, "DREAMER", "missing EEG shapes", int((df["dreamer_eeg_shape"].isna() | (df["dreamer_eeg_shape"] == "")).sum()), 0)
    add_check(checks, "DREAMER", "missing ECG shapes", int((df["dreamer_ecg_shape"].isna() | (df["dreamer_ecg_shape"] == "")).sum()), 0)

    logger.info(f"DREAMER manifest rows: {len(df)}")
    return df


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        for lower, original in lower_map.items():
            if cand.lower() == lower or cand.lower() in lower:
                return str(original)
    return None


def find_binary_col(df: pd.DataFrame, keyword: str) -> Optional[str]:
    keyword = keyword.lower()
    candidate_cols = []
    for col in df.columns:
        col_lower = str(col).lower()
        if keyword not in col_lower:
            continue
        vals = pd.to_numeric(df[col], errors="coerce").dropna().unique()
        int_vals = sorted([int(v) for v in vals if float(v).is_integer()])
        if len(int_vals) > 0 and set(int_vals).issubset({0, 1}):
            priority = 0 if "binary" in col_lower else 1
            candidate_cols.append((priority, len(col_lower), str(col)))
    if not candidate_cols:
        return None
    candidate_cols.sort()
    return candidate_cols[0][2]


def npz_shape(path: Path, preferred_keys: List[str], ndim: int) -> Tuple[str, List[int], List[str]]:
    with np.load(path, allow_pickle=True) as z:
        keys = list(z.keys())
        selected = None
        for key in preferred_keys:
            if key in keys and np.asarray(z[key]).ndim == ndim:
                selected = key
                break
        if selected is None:
            candidates = [(k, np.asarray(z[k]).size) for k in keys if np.asarray(z[k]).ndim == ndim]
            if not candidates:
                raise RuntimeError(f"No {ndim}D array found in {path}")
            selected = sorted(candidates, key=lambda x: x[1], reverse=True)[0][0]
        return selected, list(np.asarray(z[selected]).shape), keys


def build_amigos_manifest(root: Path, cfg: Dict[str, Any], logger: Logger, checks: List[Dict[str, Any]]) -> pd.DataFrame:
    logger.info("Building AMIGOS trial manifest.")
    expected = cfg["expected"]

    trial_bag_path = root / cfg["trial_bag_file"]
    feature_bag_path = root / cfg["feature_bag_file"]
    trial_meta_path = root / cfg["trial_metadata_file"]
    feature_meta_path = root / cfg["feature_metadata_file"]
    feature_names_path = root / cfg["feature_names_file"]

    trial_key, trial_shape, trial_npz_keys = npz_shape(trial_bag_path, ["X_bags", "X", "data"], 4)
    feature_key, feature_shape, feature_npz_keys = npz_shape(feature_bag_path, ["F_bags", "X_features", "features"], 3)

    trial_meta = pd.read_csv(trial_meta_path)
    feature_meta = pd.read_csv(feature_meta_path)
    feature_names = pd.read_csv(feature_names_path)

    val_col = find_binary_col(trial_meta, "valence")
    aro_col = find_binary_col(trial_meta, "arousal")

    if val_col is None or aro_col is None:
        raise RuntimeError(f"Could not detect AMIGOS binary label columns. Columns={list(trial_meta.columns)}")

    subj_col = find_column(trial_meta, ["subject_id", "subject", "participant", "pid"])
    trial_col = find_column(trial_meta, ["trial_id", "trial", "video"])

    rows: List[Dict[str, Any]] = []

    for idx, row in tqdm(trial_meta.iterrows(), total=len(trial_meta), desc="AMIGOS manifest", unit="trial"):
        if subj_col:
            raw_subj = row[subj_col]
            subject_text = str(raw_subj)
            nums = re.findall(r"\d+", subject_text)
            subject_index = int(nums[0]) if nums else int(idx // expected["trials_per_subject"] + 1)
        else:
            subject_index = int(idx // expected["trials_per_subject"] + 1)

        if trial_col:
            raw_trial = row[trial_col]
            trial_text = str(raw_trial)
            nums = re.findall(r"\d+", trial_text)
            trial_index = int(nums[-1]) if nums else int(idx % expected["trials_per_subject"] + 1)
        else:
            trial_index = int(idx % expected["trials_per_subject"] + 1)

        rows.append({
            "dataset": "AMIGOS",
            "trial_uid": f"AMIGOS__S{subject_index:02d}__T{trial_index:02d}",
            "subject_id": f"AMIGOS_S{subject_index:02d}",
            "subject_index": subject_index,
            "session_id": "NA",
            "session_index": np.nan,
            "trial_id": f"T{trial_index:02d}",
            "trial_index": trial_index,
            "global_index_within_dataset": int(idx + 1),
            "task_family": "binary_valence_arousal",
            "seed_iv_label": np.nan,
            "valence_score": row.get("valence", np.nan),
            "arousal_score": row.get("arousal", np.nan),
            "dominance_score": np.nan,
            "valence_binary": int(row[val_col]),
            "arousal_binary": int(row[aro_col]),
            "dominance_binary": np.nan,
            "modalities_available": "EEG|ECG|GSR",
            "has_eeg": 1,
            "has_eye": 0,
            "has_ecg": 1,
            "has_gsr": 1,
            "source_label": f"{trial_meta_path.name}:{val_col}/{aro_col}",
            "source_file_primary": str(trial_bag_path),
            "source_feature_bag_file": str(feature_bag_path),
            "source_trial_metadata_file": str(trial_meta_path),
            "source_feature_metadata_file": str(feature_meta_path),
            "trial_bag_key": trial_key,
            "feature_bag_key": feature_key,
            "trial_bag_shape_global": "x".join(map(str, trial_shape)),
            "feature_bag_shape_global": "x".join(map(str, feature_shape)),
            "windows_per_trial": int(trial_shape[1]),
            "samples_per_window": int(trial_shape[2]),
            "channels": int(trial_shape[3]),
            "feature_dim": int(feature_shape[2]),
        })

    df = pd.DataFrame(rows)

    add_check(checks, "AMIGOS", "manifest rows", int(len(df)), expected["rows"])
    add_check(checks, "AMIGOS", "subjects", int(df["subject_id"].nunique()), expected["subjects"])
    add_check(checks, "AMIGOS", "trial bag shape", trial_shape, expected["trial_bag_shape"])
    add_check(checks, "AMIGOS", "feature bag shape", feature_shape, expected["feature_bag_shape"])
    add_check(checks, "AMIGOS", "feature metadata rows", int(len(feature_meta)), expected["rows"])
    add_check(checks, "AMIGOS", "feature names count", int(len(feature_names)), expected["feature_bag_shape"][-1])

    for col, exp_counts in expected["label_counts"].items():
        obs = normalize_counts(df[col].astype(int).value_counts().to_dict())
        add_check(checks, "AMIGOS", f"{col} counts", obs, normalize_counts(exp_counts))

    logger.info(f"AMIGOS manifest rows: {len(df)}")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="02_prepare_dataset_manifests: trial-level dataset manifest generation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-paths", required=True)
    parser.add_argument("--manifest-config", default="configs/02_prepare_dataset_manifests.yaml")
    args = parser.parse_args()

    t0 = time.time()

    _main_cfg = load_yaml(Path(args.config))
    local_paths = load_yaml(Path(args.local_paths))
    manifest_cfg = load_yaml(Path(args.manifest_config))["manifest"]

    project_root = as_path(local_paths["PROJECT_ROOT"])
    out_dir = project_root / manifest_cfg["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(out_dir / "02_prepare_dataset_manifests_log.txt")
    logger.info("Starting 02_prepare_dataset_manifests.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {out_dir}")

    require_previous_audit(project_root, manifest_cfg, logger)

    checks: List[Dict[str, Any]] = []

    seed_root = as_path(local_paths[manifest_cfg["seed_iv"]["root_key"]])
    dreamer_root = as_path(local_paths[manifest_cfg["dreamer"]["root_key"]])
    amigos_root = as_path(local_paths[manifest_cfg["amigos"]["root_key"]])

    seed_df = build_seed_manifest(seed_root, manifest_cfg["seed_iv"], logger, checks)
    dreamer_df = build_dreamer_manifest(dreamer_root, manifest_cfg["dreamer"], logger, checks)
    amigos_df = build_amigos_manifest(amigos_root, manifest_cfg["amigos"], logger, checks)

    seed_path = out_dir / "02_seed_iv_trial_manifest.csv"
    dreamer_path = out_dir / "02_dreamer_trial_manifest.csv"
    amigos_path = out_dir / "02_amigos_trial_manifest.csv"
    unified_path = out_dir / "02_unified_trial_manifest.csv"
    checks_path = out_dir / "02_manifest_checks.csv"
    summary_path = out_dir / "02_manifest_summary.json"

    seed_df.to_csv(seed_path, index=False)
    dreamer_df.to_csv(dreamer_path, index=False)
    amigos_df.to_csv(amigos_path, index=False)

    unified = pd.concat([seed_df, dreamer_df, amigos_df], ignore_index=True, sort=False)
    unified.to_csv(unified_path, index=False)

    add_check(checks, "UNIFIED", "manifest rows", int(len(unified)), int(len(seed_df) + len(dreamer_df) + len(amigos_df)))
    add_check(checks, "UNIFIED", "datasets", sorted(unified["dataset"].unique().tolist()), ["AMIGOS", "DREAMER", "SEED-IV"])

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(checks_path, index=False)

    failed = checks_df[checks_df["passed"] == False]
    overall_passed = len(failed) == 0

    summary = {
        "name": manifest_cfg["name"],
        "created_at": now(),
        "overall_passed": bool(overall_passed),
        "elapsed_seconds": round(time.time() - t0, 3),
        "outputs": {
            "seed_iv_manifest": str(seed_path),
            "dreamer_manifest": str(dreamer_path),
            "amigos_manifest": str(amigos_path),
            "unified_manifest": str(unified_path),
            "checks": str(checks_path),
            "log": str(out_dir / "02_prepare_dataset_manifests_log.txt"),
        },
        "row_counts": {
            "SEED-IV": int(len(seed_df)),
            "DREAMER": int(len(dreamer_df)),
            "AMIGOS": int(len(amigos_df)),
            "UNIFIED": int(len(unified)),
        },
        "failed_checks": failed.to_dict(orient="records"),
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Wrote SEED-IV manifest: {seed_path}")
    logger.info(f"Wrote DREAMER manifest: {dreamer_path}")
    logger.info(f"Wrote AMIGOS manifest: {amigos_path}")
    logger.info(f"Wrote unified manifest: {unified_path}")
    logger.info(f"Wrote checks: {checks_path}")
    logger.info(f"Wrote summary: {summary_path}")
    logger.info(f"Overall manifest stage passed: {overall_passed}")
    logger.info(f"Elapsed seconds: {summary['elapsed_seconds']}")

    print("\nTARGETED OUTPUTS")
    print(f"1. {seed_path}")
    print(f"2. {dreamer_path}")
    print(f"3. {amigos_path}")
    print(f"4. {unified_path}")
    print(f"5. {checks_path}")
    print(f"6. {summary_path}")
    print(f"7. {out_dir / '02_prepare_dataset_manifests_log.txt'}")

    if not overall_passed:
        logger.error("Manifest stage failed. Do not proceed to split generation or preprocessing.")
        return 1

    logger.info("Manifest stage passed. It is safe to proceed to split-design stage.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

