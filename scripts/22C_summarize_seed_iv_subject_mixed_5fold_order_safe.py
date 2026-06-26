from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def as_path(x: str) -> Path:
    return Path(str(x).replace("\\", "/")).expanduser().resolve()


def add_check(checks: List[Dict[str, Any]], name: str, observed: Any, expected: Any, passed: Optional[bool] = None) -> None:
    if passed is None:
        passed = observed == expected
    checks.append({"check": name, "observed": observed, "expected": expected, "passed": bool(passed)})


def mean_std(series: pd.Series) -> str:
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float).to_numpy()
    if len(vals) == 0:
        return "NA"
    if len(vals) == 1:
        return f"{vals[0]:.3f}"
    return f"{np.mean(vals):.3f} ± {np.std(vals, ddof=1):.3f}"


def safe_mean(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna().astype(float).to_numpy()
    return float(np.mean(vals)) if len(vals) else float("nan")


def display_name(variant: str) -> str:
    mapping = {
        "temporal_mlp": "Temporal MLP",
        "lstm": "LSTM",
        "gru": "GRU",
        "bilstm": "BiLSTM",
        "tcn": "TCN",
        "cnn_lstm": "CNN-LSTM",
        "dynamer_v2": "DynaMER-BiTCN",
        "dynamer_v3": "DynaMER-ADF",
        "dynamer_v5": "DynaMER-Anchor",
    }
    return mapping.get(str(variant).lower(), str(variant))


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 22C order-safe: summarize SEED-IV subject-mixed 5-fold diagnostic results.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--local-paths", default="configs/local_paths.yaml")
    parser.add_argument("--training-dir", default=None)
    parser.add_argument("--loso-reference-ba", type=float, default=0.600)
    parser.add_argument("--loso-reference-macro-f1", type=float, default=0.584)
    parser.add_argument("--loso-reference-roc-auc", type=float, default=0.863)
    parser.add_argument("--loso-reference-pr-auc", type=float, default=0.729)
    args = parser.parse_args()

    _main_cfg = load_yaml(Path(args.config))
    local_paths = load_yaml(Path(args.local_paths))
    project_root = as_path(local_paths["PROJECT_ROOT"])

    base_dir = project_root / "outputs" / "protocol_extension" / "22_seed_iv_subject_mixed_5fold"
    training_dir = Path(args.training_dir) if args.training_dir else base_dir / "training_order_safe"
    epoch_path = training_dir / "22B_subject_mixed_5fold_all_epoch_metrics.csv"
    run_report_path = training_dir / "22B_subject_mixed_5fold_training_run_report.csv"

    if not epoch_path.exists():
        raise FileNotFoundError(f"Missing Stage 22B epoch metrics: {epoch_path}")
    if not run_report_path.exists():
        raise FileNotFoundError(f"Missing Stage 22B run report: {run_report_path}")

    out_dir = base_dir / "summary_order_safe"
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = pd.read_csv(epoch_path)
    run_report = pd.read_csv(run_report_path)

    required = ["model_variant", "fold_index", "epoch", "monitor_value", "test_balanced_accuracy", "test_macro_f1", "test_roc_auc", "test_pr_auc"]
    missing = [c for c in required if c not in epochs.columns]
    if missing:
        raise RuntimeError(f"Epoch metric file missing required columns: {missing}")

    # Select one row per model/fold using the validation-selected best epoch.
    best_rows = []
    for (variant, fold), group in epochs.groupby(["model_variant", "fold_index"], dropna=False):
        group = group.copy()
        # Use monitor_value, then earliest epoch tie-break. This mirrors validation-based checkpointing.
        group["_monitor"] = pd.to_numeric(group["monitor_value"], errors="coerce")
        if group["_monitor"].notna().any():
            idx = group.sort_values(["_monitor", "epoch"], ascending=[False, True]).index[-1]
            # Correction: sort ascending false then true, top is first.
            idx = group.sort_values(["_monitor", "epoch"], ascending=[False, True]).index[0]
        else:
            idx = group.sort_values("epoch").index[-1]
        best_rows.append(group.loc[idx].drop(labels=["_monitor"], errors="ignore"))

    best_df = pd.DataFrame(best_rows).sort_values(["model_variant", "fold_index"]).reset_index(drop=True)
    best_df["model_display"] = best_df["model_variant"].map(display_name)

    summary_rows = []
    for variant, group in best_df.groupby("model_variant"):
        group = group.sort_values("fold_index")
        summary_rows.append({
            "protocol": "Subject-mixed 5-fold",
            "model_variant": variant,
            "model_display": display_name(variant),
            "folds": int(group["fold_index"].nunique()),
            "BA": mean_std(group["test_balanced_accuracy"]),
            "Macro-F1": mean_std(group["test_macro_f1"]),
            "ROC-AUC": mean_std(group["test_roc_auc"]),
            "PR-AUC": mean_std(group["test_pr_auc"]),
            "BA_mean": safe_mean(group["test_balanced_accuracy"]),
            "MacroF1_mean": safe_mean(group["test_macro_f1"]),
            "ROCAUC_mean": safe_mean(group["test_roc_auc"]),
            "PRAUC_mean": safe_mean(group["test_pr_auc"]),
            "mean_best_epoch": safe_mean(group["epoch"]),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values(["BA_mean", "MacroF1_mean"], ascending=[False, False]).reset_index(drop=True)

    # DynaMER-ADF protocol-gap row for manuscript reporting.
    adf = summary_df[summary_df["model_variant"] == "dynamer_v3"].copy()
    protocol_gap_rows = []
    if len(adf) == 1:
        adf_row = adf.iloc[0]
        protocol_gap_rows.append({
            "protocol": "Subject-mixed 5-fold",
            "role": "Diagnostic upper-bound capacity estimate",
            "model": "DynaMER-ADF",
            "folds": int(adf_row["folds"]),
            "BA": adf_row["BA"],
            "Macro-F1": adf_row["Macro-F1"],
            "ROC-AUC": adf_row["ROC-AUC"],
            "PR-AUC": adf_row["PR-AUC"],
            "BA_mean": float(adf_row["BA_mean"]),
            "MacroF1_mean": float(adf_row["MacroF1_mean"]),
            "delta_BA_vs_subject_LOSO": float(adf_row["BA_mean"] - args.loso_reference_ba),
            "delta_MacroF1_vs_subject_LOSO": float(adf_row["MacroF1_mean"] - args.loso_reference_macro_f1),
            "interpretation": "Measures within-dataset discriminative capacity when subject-specific patterns may be shared across train and test.",
        })
    protocol_gap_rows.append({
        "protocol": "Conventional subject-LOSO",
        "role": "Primary subject-independent deployment-like evaluation",
        "model": "DynaMER-ADF",
        "folds": 15,
        "BA": "0.600 ± 0.087",
        "Macro-F1": "0.584 ± 0.098",
        "ROC-AUC": "0.863 ± 0.039",
        "PR-AUC": "0.729 ± 0.070",
        "BA_mean": float(args.loso_reference_ba),
        "MacroF1_mean": float(args.loso_reference_macro_f1),
        "delta_BA_vs_subject_LOSO": 0.0,
        "delta_MacroF1_vs_subject_LOSO": 0.0,
        "interpretation": "Measures held-out-subject generalization and remains the primary claim.",
    })
    gap_df = pd.DataFrame(protocol_gap_rows)

    best_rows_path = out_dir / "22C_subject_mixed_5fold_best_epoch_rows.csv"
    model_summary_path = out_dir / "22C_subject_mixed_5fold_model_summary.csv"
    protocol_gap_path = out_dir / "22C_subject_mixed_vs_loso_protocol_gap.csv"
    checks_path = out_dir / "22C_subject_mixed_5fold_summary_checks.csv"
    summary_json_path = out_dir / "22C_subject_mixed_5fold_summary.json"
    latex_path = out_dir / "22C_subject_mixed_vs_loso_table.tex"

    best_df.to_csv(best_rows_path, index=False)
    summary_df.to_csv(model_summary_path, index=False)
    gap_df.to_csv(protocol_gap_path, index=False)

    # Minimal LaTeX table for later manuscript/supplement insertion.
    latex_lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Diagnostic SEED-IV subject-mixed 5-fold evaluation compared with the primary subject-LOSO result. Subject-mixed 5-fold is reported only as an upper-bound capacity estimate and is not used as the main subject-independent claim.}",
        r"\label{tab:subject_mixed_vs_loso}",
        r"\begin{tabular}{l l c c c c}",
        r"\hline",
        r"Protocol & Role & Folds & BA & Macro-F1 & PR-AUC \\",
        r"\hline",
    ]
    for _, row in gap_df.iterrows():
        latex_lines.append(
            f"{row['protocol']} & {row['role']} & {int(row['folds'])} & {row['BA']} & {row['Macro-F1']} & {row['PR-AUC']} \\\\"
        )
    latex_lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    latex_path.write_text("\n".join(latex_lines), encoding="utf-8")

    checks: List[Dict[str, Any]] = []
    add_check(checks, "best rows per completed model-fold", int(len(best_df)), ">0", int(len(best_df)) > 0)
    add_check(checks, "model summary rows", int(len(summary_df)), ">0", int(len(summary_df)) > 0)
    add_check(checks, "DynaMER-ADF present", int((summary_df["model_variant"] == "dynamer_v3").sum()), 1)
    add_check(checks, "all summarized models have 5 folds", int((summary_df["folds"] == 5).sum()), int(len(summary_df)))
    checks_df = pd.DataFrame(checks)
    failed = checks_df[checks_df["passed"] == False]
    checks_df.to_csv(checks_path, index=False)

    summary = {
        "name": "22C_summarize_seed_iv_subject_mixed_5fold_order_safe",
        "created_at": now(),
        "overall_passed": bool(len(failed) == 0),
        "outputs": {
            "best_epoch_rows": str(best_rows_path),
            "model_summary": str(model_summary_path),
            "protocol_gap": str(protocol_gap_path),
            "latex_table": str(latex_path),
            "checks": str(checks_path),
            "summary": str(summary_json_path),
        },
        "failed_checks": failed.to_dict(orient="records"),
        "scientific_statement": "Subject-mixed 5-fold results quantify diagnostic within-dataset capacity. The primary subject-independent claim remains the conventional subject-LOSO result unless nested LOSO later confirms or updates it.",
    }
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nStage 22C outputs:")
    print(f"1. {best_rows_path}")
    print(f"2. {model_summary_path}")
    print(f"3. {protocol_gap_path}")
    print(f"4. {latex_path}")
    print(f"5. {checks_path}")
    print(f"6. {summary_json_path}")

    if len(failed) > 0:
        print("\n[ERROR] Stage 22C failed checks. Inspect the checks CSV before manuscript use.")
        return 1
    print("\n[DONE] Stage 22C passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
