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


def assign_phase(dataset: str, protocol: str, phases: Dict[str, Any]) -> tuple[str, int, str]:
    for phase_name, phase_cfg in phases.items():
        if dataset in phase_cfg["datasets"] and protocol in phase_cfg["protocols"]:
            return phase_name, int(phase_cfg["priority"]), str(phase_cfg["purpose"])
    raise RuntimeError(f"No phase assigned for dataset={dataset}, protocol={protocol}")


def main() -> int:
    parser = argparse.ArgumentParser(description="11_full_experiment_registry: create the official full experiment run registry.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--local-paths", required=True)
    parser.add_argument("--registry-config", default="configs/11_full_experiment_registry.yaml")
    args = parser.parse_args()

    t0 = time.time()

    _main_cfg = load_yaml(Path(args.config))
    local_paths = load_yaml(Path(args.local_paths))
    reg_cfg = load_yaml(Path(args.registry_config))["experiment_registry"]

    project_root = as_path(local_paths["PROJECT_ROOT"])
    out_dir = project_root / reg_cfg["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = Logger(out_dir / "11_full_experiment_registry_log.txt")
    logger.info("Starting 11_full_experiment_registry.")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {out_dir}")

    req = reg_cfg["required_previous_steps"]
    if req.get("require_passed", True):
        require_passed_json(project_root, req["training_smoke_summary_json"], logger)
        require_passed_json(project_root, req["temporal_data_module_summary_json"], logger)

    split_index = pd.read_csv(project_root / reg_cfg["inputs"]["split_index"])
    temporal_index = pd.read_csv(project_root / reg_cfg["inputs"]["temporal_view_index"])

    rows: List[Dict[str, Any]] = []

    for _, split_row in split_index.iterrows():
        dataset = str(split_row["dataset"])
        task = str(split_row["task"])
        protocol = str(split_row["protocol"])

        if dataset not in reg_cfg["tasks"]:
            continue
        if task not in reg_cfg["tasks"][dataset]:
            continue

        task_cfg = reg_cfg["tasks"][dataset][task]
        if protocol not in task_cfg["allowed_protocols"]:
            continue

        phase_name, phase_priority, phase_purpose = assign_phase(dataset, protocol, reg_cfg["phases"])
        scientific_role = str(reg_cfg["scientific_roles"][protocol])

        run_id = (
            f"11__{phase_name}__{dataset.replace('-', '_')}__{task}"
            f"__{protocol}__fold_{int(split_row['fold_index']):03d}"
        )

        rows.append(
            {
                "run_id": run_id,
                "phase": phase_name,
                "phase_priority": phase_priority,
                "phase_purpose": phase_purpose,
                "dataset": dataset,
                "task": task,
                "protocol": protocol,
                "scientific_role": scientific_role,
                "is_primary_claim_allowed": int("upper_bound" not in scientific_role),
                "fold_index": int(split_row["fold_index"]),
                "split_id": str(split_row["split_id"]),
                "split_file": str(split_row["split_file"]),
                "model_name": reg_cfg["model_family"]["proposed_model_name"],
                "hidden_dim": int(reg_cfg["model_family"]["hidden_dim"]),
                "temporal_backbone": str(reg_cfg["model_family"]["temporal_backbone"]),
                "fusion": str(reg_cfg["model_family"]["fusion"]),
                "head": str(reg_cfg["model_family"]["head"]),
                "label_column": str(task_cfg["label_column"]),
                "num_classes": int(task_cfg["num_classes"]),
                "modality_keys": "|".join(task_cfg["modality_keys"]),
                "status": "queued",
                "notes": "Generated by Stage 11 registry. Training has not been run yet.",
            }
        )

    registry = pd.DataFrame(rows).sort_values(
        ["phase_priority", "dataset", "task", "protocol", "fold_index"]
    ).reset_index(drop=True)

    checks: List[Dict[str, Any]] = []

    add_check(checks, "registry rows", int(len(registry)), int(len(split_index)))
    add_check(checks, "unique run ids", int(registry["run_id"].nunique()), int(len(registry)))
    add_check(checks, "all split files exist", int(sum(Path(p).exists() for p in registry["split_file"])), int(len(registry)))
    add_check(checks, "all statuses queued", sorted(registry["status"].unique().tolist()), ["queued"])
    add_check(checks, "datasets included", sorted(registry["dataset"].unique().tolist()), ["AMIGOS", "DREAMER", "SEED-IV"])
    add_check(checks, "subject mixed marked non-primary", int(registry.loc[registry["protocol"] == "subject_mixed_upper_bound", "is_primary_claim_allowed"].sum()), 0)
    add_check(checks, "primary protocols marked primary", int((registry.loc[registry["protocol"] != "subject_mixed_upper_bound", "is_primary_claim_allowed"] == 1).sum()), int((registry["protocol"] != "subject_mixed_upper_bound").sum()))
    add_check(checks, "temporal index rows available", int(len(temporal_index)), 1734)

    expected_tasks = sorted({task for dataset_cfg in reg_cfg["tasks"].values() for task in dataset_cfg.keys()})
    add_check(checks, "tasks included", sorted(registry["task"].unique().tolist()), expected_tasks)

    phase_summary = (
        registry.groupby(["phase", "phase_priority", "dataset", "task", "protocol", "scientific_role"])
        .size()
        .reset_index(name="n_runs")
        .sort_values(["phase_priority", "dataset", "task", "protocol"])
    )

    checks_df = pd.DataFrame(checks)

    registry_path = out_dir / "11_full_experiment_registry.csv"
    phase_summary_path = out_dir / "11_experiment_phase_summary.csv"
    checks_path = out_dir / "11_experiment_registry_checks.csv"
    summary_path = out_dir / "11_experiment_registry_summary.json"

    registry.to_csv(registry_path, index=False)
    phase_summary.to_csv(phase_summary_path, index=False)
    checks_df.to_csv(checks_path, index=False)

    failed = checks_df[checks_df["passed"] == False]
    overall_passed = len(failed) == 0

    summary = {
        "name": reg_cfg["name"],
        "created_at": now(),
        "overall_passed": bool(overall_passed),
        "elapsed_seconds": round(time.time() - t0, 3),
        "row_counts": {
            "total_runs": int(len(registry)),
            "primary_claim_runs": int(registry["is_primary_claim_allowed"].sum()),
            "diagnostic_runs": int((registry["is_primary_claim_allowed"] == 0).sum()),
        },
        "outputs": {
            "registry": str(registry_path),
            "phase_summary": str(phase_summary_path),
            "checks": str(checks_path),
            "summary": str(summary_path),
            "log": str(out_dir / "11_full_experiment_registry_log.txt"),
        },
        "failed_checks": failed.to_dict(orient="records"),
        "scientific_statement": "This registry separates primary subject/session generalization evidence from diagnostic subject-mixed upper-bound runs. Subject-mixed runs are explicitly marked as non-primary and must not be used for main generalization claims.",
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(f"Wrote registry: {registry_path}")
    logger.info(f"Wrote phase summary: {phase_summary_path}")
    logger.info(f"Wrote checks: {checks_path}")
    logger.info(f"Wrote summary: {summary_path}")
    logger.info(f"Overall registry stage passed: {overall_passed}")
    logger.info(f"Elapsed seconds: {summary['elapsed_seconds']}")

    print("\nTARGETED OUTPUTS")
    print(f"1. {registry_path}")
    print(f"2. {phase_summary_path}")
    print(f"3. {checks_path}")
    print(f"4. {summary_path}")
    print(f"5. {out_dir / '11_full_experiment_registry_log.txt'}")

    if not overall_passed:
        logger.error("Experiment registry failed. Do not proceed to full training.")
        return 1

    logger.info("Experiment registry passed. It is safe to proceed to controlled full training implementation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
