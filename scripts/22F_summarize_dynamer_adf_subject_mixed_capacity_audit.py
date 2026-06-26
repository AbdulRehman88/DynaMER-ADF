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


def safe_float(x: Any) -> float:
    try:
        if pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


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


def maybe_read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def pick_best_epoch_rows(epochs: pd.DataFrame) -> pd.DataFrame:
    required = [
        "capacity_variant", "fold_index", "epoch", "monitor_value",
        "test_balanced_accuracy", "test_macro_f1", "test_roc_auc", "test_pr_auc",
    ]
    missing = [c for c in required if c not in epochs.columns]
    if missing:
        raise RuntimeError(f"Epoch metric file missing required columns: {missing}")

    rows = []
    for (variant, fold), group in epochs.groupby(["capacity_variant", "fold_index"], dropna=False):
        g = group.copy()
        g["_monitor"] = pd.to_numeric(g["monitor_value"], errors="coerce")
        if g["_monitor"].notna().any():
            idx = g.sort_values(["_monitor", "epoch"], ascending=[False, True]).index[0]
        else:
            idx = g.sort_values("epoch").index[-1]
        rows.append(g.loc[idx].drop(labels=["_monitor"], errors="ignore"))
    return pd.DataFrame(rows).sort_values(["capacity_variant", "fold_index"]).reset_index(drop=True)


def summarize_capacity(best_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in best_df.groupby("capacity_variant"):
        group = group.sort_values("fold_index")
        display = str(group["capacity_display"].iloc[0]) if "capacity_display" in group.columns else str(variant)
        rows.append({
            "protocol": "Subject-mixed 5-fold",
            "capacity_variant": variant,
            "capacity_display": display,
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
            "mean_params_m": safe_mean(group["parameter_count"]) / 1_000_000.0 if "parameter_count" in group.columns else float("nan"),
            "dropout": safe_float(group["dropout"].iloc[0]) if "dropout" in group.columns else float("nan"),
            "tcn_layers": safe_float(group["tcn_layers"].iloc[0]) if "tcn_layers" in group.columns else float("nan"),
            "modality_dropout": safe_float(group["modality_dropout"].iloc[0]) if "modality_dropout" in group.columns else float("nan"),
            "spike_mix": safe_float(group["spike_mix"].iloc[0]) if "spike_mix" in group.columns else float("nan"),
        })
    return pd.DataFrame(rows).sort_values(["BA_mean", "MacroF1_mean"], ascending=[False, False]).reset_index(drop=True)


def normalize_reference_summary(ref: pd.DataFrame) -> pd.DataFrame:
    if ref.empty:
        return pd.DataFrame()
    df = ref.copy()
    # Expected Stage 22C order-safe columns: model_display, BA_mean, MacroF1_mean, ROCAUC_mean, PRAUC_mean.
    if "model_display" not in df.columns and "capacity_display" in df.columns:
        df["model_display"] = df["capacity_display"]
    keep = [c for c in ["model_variant", "model_display", "folds", "BA", "Macro-F1", "ROC-AUC", "PR-AUC", "BA_mean", "MacroF1_mean", "ROCAUC_mean", "PRAUC_mean"] if c in df.columns]
    df = df[keep].copy()
    if "BA_mean" in df.columns:
        df = df.sort_values(["BA_mean", "MacroF1_mean"], ascending=[False, False]).reset_index(drop=True)
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 22F: summarize DynaMER-ADF subject-mixed capacity audit.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--local-paths", default="configs/local_paths.yaml")
    parser.add_argument("--training-dir", default=None)
    parser.add_argument("--reference-summary", default=None, help="Optional Stage 22C order-safe model summary CSV.")
    parser.add_argument("--subject-loso-ba", type=float, default=0.600)
    parser.add_argument("--subject-loso-macro-f1", type=float, default=0.584)
    parser.add_argument("--subject-loso-roc-auc", type=float, default=0.863)
    parser.add_argument("--subject-loso-pr-auc", type=float, default=0.729)
    args = parser.parse_args()

    _main_cfg = load_yaml(Path(args.config))
    local_paths = load_yaml(Path(args.local_paths))
    project_root = as_path(local_paths["PROJECT_ROOT"])

    base_dir = project_root / "outputs" / "protocol_extension" / "22_seed_iv_subject_mixed_5fold"
    training_dir = Path(args.training_dir) if args.training_dir else base_dir / "capacity_audit_22E"
    epoch_path = training_dir / "22E_capacity_audit_all_epoch_metrics.csv"
    run_report_path = training_dir / "22E_capacity_audit_run_report.csv"

    if not epoch_path.exists():
        raise FileNotFoundError(f"Missing Stage 22E epoch metrics: {epoch_path}")
    if not run_report_path.exists():
        raise FileNotFoundError(f"Missing Stage 22E run report: {run_report_path}")

    out_dir = base_dir / "capacity_audit_22F_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    epochs = pd.read_csv(epoch_path)
    run_report = pd.read_csv(run_report_path)
    best_df = pick_best_epoch_rows(epochs)
    capacity_summary = summarize_capacity(best_df)

    ref_path = Path(args.reference_summary) if args.reference_summary else base_dir / "summary_order_safe" / "22C_subject_mixed_5fold_model_summary.csv"
    ref_summary = normalize_reference_summary(maybe_read_csv(ref_path))

    # Comparison against current subject-mixed best reference models.
    reference_best = None
    reference_best_ba = float("nan")
    reference_best_macro = float("nan")
    if not ref_summary.empty and "BA_mean" in ref_summary.columns:
        ref_sorted = ref_summary.sort_values(["BA_mean", "MacroF1_mean"], ascending=[False, False]).reset_index(drop=True)
        reference_best = str(ref_sorted.loc[0, "model_display"])
        reference_best_ba = safe_float(ref_sorted.loc[0, "BA_mean"])
        reference_best_macro = safe_float(ref_sorted.loc[0, "MacroF1_mean"])

    best_capacity = capacity_summary.iloc[0] if len(capacity_summary) else None
    best_capacity_name = str(best_capacity["capacity_display"]) if best_capacity is not None else "NA"
    best_capacity_ba = safe_float(best_capacity["BA_mean"]) if best_capacity is not None else float("nan")
    best_capacity_macro = safe_float(best_capacity["MacroF1_mean"]) if best_capacity is not None else float("nan")

    comparison_rows = []
    if best_capacity is not None:
        comparison_rows.append({
            "group": "Best ADF-family subject-mixed capacity candidate",
            "model_or_variant": best_capacity_name,
            "folds": int(best_capacity["folds"]),
            "BA": best_capacity["BA"],
            "Macro-F1": best_capacity["Macro-F1"],
            "ROC-AUC": best_capacity["ROC-AUC"],
            "PR-AUC": best_capacity["PR-AUC"],
            "BA_mean": best_capacity_ba,
            "MacroF1_mean": best_capacity_macro,
            "delta_BA_vs_LOSO_ADF": best_capacity_ba - float(args.subject_loso_ba),
            "delta_MacroF1_vs_LOSO_ADF": best_capacity_macro - float(args.subject_loso_macro_f1),
            "delta_BA_vs_subject_mixed_reference_best": best_capacity_ba - reference_best_ba if not math.isnan(reference_best_ba) else float("nan"),
            "delta_MacroF1_vs_subject_mixed_reference_best": best_capacity_macro - reference_best_macro if not math.isnan(reference_best_macro) else float("nan"),
        })
    if reference_best is not None:
        ref_row = ref_summary.sort_values(["BA_mean", "MacroF1_mean"], ascending=[False, False]).iloc[0]
        comparison_rows.append({
            "group": "Best existing subject-mixed reference from Stage 22C",
            "model_or_variant": reference_best,
            "folds": int(ref_row["folds"]) if "folds" in ref_row else 5,
            "BA": ref_row.get("BA", "NA"),
            "Macro-F1": ref_row.get("Macro-F1", "NA"),
            "ROC-AUC": ref_row.get("ROC-AUC", "NA"),
            "PR-AUC": ref_row.get("PR-AUC", "NA"),
            "BA_mean": reference_best_ba,
            "MacroF1_mean": reference_best_macro,
            "delta_BA_vs_LOSO_ADF": reference_best_ba - float(args.subject_loso_ba),
            "delta_MacroF1_vs_LOSO_ADF": reference_best_macro - float(args.subject_loso_macro_f1),
            "delta_BA_vs_subject_mixed_reference_best": 0.0,
            "delta_MacroF1_vs_subject_mixed_reference_best": 0.0,
        })
    comparison_df = pd.DataFrame(comparison_rows)

    # Write outputs.
    best_rows_path = out_dir / "22F_capacity_audit_best_epoch_rows.csv"
    capacity_summary_path = out_dir / "22F_capacity_audit_model_summary.csv"
    comparison_path = out_dir / "22F_capacity_audit_vs_reference_best.csv"
    checks_path = out_dir / "22F_capacity_audit_summary_checks.csv"
    summary_json_path = out_dir / "22F_capacity_audit_summary.json"
    latex_path = out_dir / "22F_capacity_audit_table.tex"

    best_df.to_csv(best_rows_path, index=False)
    capacity_summary.to_csv(capacity_summary_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)

    # Compact LaTeX table for later supplement/report use.
    latex_lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{SEED-IV subject-mixed 5-fold DynaMER-ADF-family capacity audit. This diagnostic protocol estimates within-dataset discriminative capacity and is not used as the primary subject-independent claim.}",
        r"\label{tab:adf_capacity_audit_subject_mixed}",
        r"\begin{tabular}{l c c c c c}",
        r"\hline",
        r"ADF-family candidate & Folds & BA & Macro-F1 & ROC-AUC & PR-AUC \\",
        r"\hline",
    ]
    for _, row in capacity_summary.iterrows():
        latex_lines.append(
            f"{row['capacity_display']} & {int(row['folds'])} & {row['BA']} & {row['Macro-F1']} & {row['ROC-AUC']} & {row['PR-AUC']} \\\\"  # noqa: E501
        )
    latex_lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    latex_path.write_text("\n".join(latex_lines), encoding="utf-8")

    checks: List[Dict[str, Any]] = []
    add_check(checks, "run report has no failed runs", int((run_report["status"] == "failed").sum()), 0)
    add_check(checks, "best epoch rows exist", int(len(best_df)), ">0", int(len(best_df)) > 0)
    add_check(checks, "capacity summary rows exist", int(len(capacity_summary)), ">0", int(len(capacity_summary)) > 0)
    add_check(checks, "all capacity candidates have 5 folds", int((capacity_summary["folds"] == 5).sum()) if len(capacity_summary) else 0, int(len(capacity_summary)))
    add_check(checks, "reference summary found", str(ref_path.exists()), "True", bool(ref_path.exists()))
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(checks_path, index=False)
    failed = checks_df[checks_df["passed"] == False]

    overall_passed = len(failed) == 0
    best_beats_reference = bool(best_capacity_ba >= reference_best_ba) if not math.isnan(reference_best_ba) and not math.isnan(best_capacity_ba) else None

    summary = {
        "name": "22F_summarize_dynamer_adf_subject_mixed_capacity_audit",
        "created_at": now(),
        "overall_passed": bool(overall_passed),
        "best_capacity_candidate": best_capacity_name,
        "best_capacity_BA_mean": best_capacity_ba,
        "best_capacity_MacroF1_mean": best_capacity_macro,
        "reference_best_model": reference_best,
        "reference_best_BA_mean": reference_best_ba,
        "reference_best_MacroF1_mean": reference_best_macro,
        "best_capacity_beats_reference_by_BA": best_beats_reference,
        "outputs": {
            "best_epoch_rows": str(best_rows_path),
            "capacity_summary": str(capacity_summary_path),
            "comparison": str(comparison_path),
            "latex_table": str(latex_path),
            "checks": str(checks_path),
            "summary": str(summary_json_path),
        },
        "failed_checks": failed.to_dict(orient="records"),
        "scientific_statement": "Use this audit to determine whether an ADF-family setting can match or exceed the current subject-mixed reference best. Regardless of the outcome, the primary manuscript claim remains subject-LOSO unless the paper is explicitly reframed.",
    }
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nStage 22F outputs:")
    print(f"1. {best_rows_path}")
    print(f"2. {capacity_summary_path}")
    print(f"3. {comparison_path}")
    print(f"4. {latex_path}")
    print(f"5. {checks_path}")
    print(f"6. {summary_json_path}")

    if len(failed) > 0:
        print("\n[ERROR] Stage 22F failed checks. Inspect checks CSV before manuscript use.")
        return 1
    print("\n[DONE] Stage 22F passed.")
    if best_beats_reference is not None:
        print(f"Best capacity candidate: {best_capacity_name} | BA={best_capacity_ba:.3f} | Macro-F1={best_capacity_macro:.3f}")
        print(f"Reference best: {reference_best} | BA={reference_best_ba:.3f} | Macro-F1={reference_best_macro:.3f}")
        print(f"Best ADF-family candidate beats/equal reference by BA: {best_beats_reference}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
