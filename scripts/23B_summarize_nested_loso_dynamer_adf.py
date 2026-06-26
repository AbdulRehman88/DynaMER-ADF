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
    x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    if len(x) == 0:
        return "NA"
    if len(x) == 1:
        return f"{x.mean():.3f}"
    return f"{x.mean():.3f} $\\pm$ {x.std(ddof=1):.3f}"


def mean_value(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    return float(x.mean()) if len(x) else float("nan")


def std_value(series: pd.Series) -> float:
    x = pd.to_numeric(series, errors="coerce").dropna().astype(float)
    return float(x.std(ddof=1)) if len(x) > 1 else 0.0 if len(x) == 1 else float("nan")


def write_latex_table(protocol_df: pd.DataFrame, path: Path) -> None:
    lines = []
    lines.append(r"\begin{table*}[!t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Protocol comparison for SEED-IV DynaMER-ADF. Subject-mixed 5-fold estimates diagnostic within-dataset capacity, conventional subject-LOSO is the primary held-out-subject benchmark, and nested LOSO reports the bias-controlled ADF-family result selected only through inner subject-level validation.}")
    lines.append(r"\label{tab:seediv_protocol_comparison_nested_loso}")
    lines.append(r"\scriptsize")
    lines.append(r"\begin{tabular}{l c c c c l}")
    lines.append(r"\hline")
    lines.append(r"Protocol & Folds & BA & Macro-F1 & ROC-AUC & Role \\")
    lines.append(r"\hline")
    for _, row in protocol_df.iterrows():
        lines.append(
            f"{row['Protocol']} & {row['Folds']} & {row['BA']} & {row['Macro-F1']} & {row['ROC-AUC']} & {row['Role']} \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 23B: summarize nested LOSO DynaMER-ADF results.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--local-paths", default="configs/local_paths.yaml")
    parser.add_argument("--stage23-dir", default=None)
    parser.add_argument("--subject-mixed-stage22-dir", default=None)
    args = parser.parse_args()

    _main_cfg = load_yaml(Path(args.config))
    local_paths = load_yaml(Path(args.local_paths))
    project_root = as_path(local_paths["PROJECT_ROOT"])

    stage23_dir = Path(args.stage23_dir) if args.stage23_dir else project_root / "outputs" / "protocol_extension" / "23_nested_loso_dynamer_adf"
    out_dir = stage23_dir / "23B_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    outer_metrics_path = stage23_dir / "23A_nested_loso_outer_final_metrics.csv"
    selection_path = stage23_dir / "23A_nested_loso_outer_selection.csv"
    inner_best_path = stage23_dir / "23A_nested_loso_inner_best_epoch_rows.csv"
    stage23_summary_path = stage23_dir / "23A_nested_loso_summary.json"

    if not outer_metrics_path.exists():
        raise FileNotFoundError(f"Missing nested outer metrics: {outer_metrics_path}")
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing nested outer selection: {selection_path}")
    if not inner_best_path.exists():
        raise FileNotFoundError(f"Missing nested inner best rows: {inner_best_path}")

    if stage23_summary_path.exists():
        obj = json.loads(stage23_summary_path.read_text(encoding="utf-8"))
        if not bool(obj.get("overall_passed", False)):
            raise RuntimeError(f"Stage 23A summary exists but did not pass: {stage23_summary_path}")

    outer = pd.read_csv(outer_metrics_path)
    selection = pd.read_csv(selection_path)
    inner_best = pd.read_csv(inner_best_path)

    # Fold-level clean table.
    fold_cols = [
        "outer_fold", "test_subject", "selected_variant", "selected_variant_display", "final_epochs",
        "test_balanced_accuracy", "test_macro_f1", "test_roc_auc", "test_pr_auc", "parameter_count",
    ]
    fold_cols = [c for c in fold_cols if c in outer.columns]
    fold_df = outer[fold_cols].copy().sort_values("outer_fold") if not outer.empty else pd.DataFrame(columns=fold_cols)
    fold_df.to_csv(out_dir / "23B_nested_loso_fold_metrics.csv", index=False)

    nested_summary = pd.DataFrame([{
        "protocol": "nested_loso_adf_family",
        "folds": int(fold_df["outer_fold"].nunique()) if "outer_fold" in fold_df.columns else int(len(fold_df)),
        "BA_mean": mean_value(fold_df["test_balanced_accuracy"]) if "test_balanced_accuracy" in fold_df.columns else float("nan"),
        "BA_std": std_value(fold_df["test_balanced_accuracy"]) if "test_balanced_accuracy" in fold_df.columns else float("nan"),
        "MacroF1_mean": mean_value(fold_df["test_macro_f1"]) if "test_macro_f1" in fold_df.columns else float("nan"),
        "MacroF1_std": std_value(fold_df["test_macro_f1"]) if "test_macro_f1" in fold_df.columns else float("nan"),
        "ROCAUC_mean": mean_value(fold_df["test_roc_auc"]) if "test_roc_auc" in fold_df.columns else float("nan"),
        "ROCAUC_std": std_value(fold_df["test_roc_auc"]) if "test_roc_auc" in fold_df.columns else float("nan"),
        "PRAUC_mean": mean_value(fold_df["test_pr_auc"]) if "test_pr_auc" in fold_df.columns else float("nan"),
        "PRAUC_std": std_value(fold_df["test_pr_auc"]) if "test_pr_auc" in fold_df.columns else float("nan"),
        "BA": mean_std(fold_df["test_balanced_accuracy"]) if "test_balanced_accuracy" in fold_df.columns else "NA",
        "Macro-F1": mean_std(fold_df["test_macro_f1"]) if "test_macro_f1" in fold_df.columns else "NA",
        "ROC-AUC": mean_std(fold_df["test_roc_auc"]) if "test_roc_auc" in fold_df.columns else "NA",
        "PR-AUC": mean_std(fold_df["test_pr_auc"]) if "test_pr_auc" in fold_df.columns else "NA",
    }])
    nested_summary.to_csv(out_dir / "23B_nested_loso_summary.csv", index=False)

    # Variant selection counts and inner validation summary.
    if "selected_variant_display" in selection.columns:
        variant_counts = selection["selected_variant_display"].value_counts().rename_axis("selected_variant_display").reset_index(name="outer_fold_count")
    else:
        variant_counts = pd.DataFrame()
    variant_counts.to_csv(out_dir / "23B_nested_loso_selected_variant_counts.csv", index=False)

    inner_summary_rows = []
    for (outer_fold, variant), part in inner_best.groupby(["outer_fold", "variant"]):
        inner_summary_rows.append({
            "outer_fold": int(outer_fold),
            "variant": str(variant),
            "variant_display": part["variant_display"].iloc[0] if "variant_display" in part.columns else str(variant),
            "n_inner_folds": int(part["inner_fold"].nunique()),
            "mean_inner_val_macro_f1": mean_value(part["val_macro_f1"]),
            "std_inner_val_macro_f1": std_value(part["val_macro_f1"]),
            "mean_inner_val_ba": mean_value(part["val_balanced_accuracy"]),
            "median_best_epoch": float(pd.to_numeric(part["epoch"], errors="coerce").median()),
        })
    inner_summary = pd.DataFrame(inner_summary_rows).sort_values(["outer_fold", "mean_inner_val_macro_f1"], ascending=[True, False]) if inner_summary_rows else pd.DataFrame()
    inner_summary.to_csv(out_dir / "23B_nested_loso_inner_candidate_summary.csv", index=False)

    # Protocol comparison with already locked manuscript results and optional Stage 22 diagnostic evidence.
    # Conventional LOSO values are the locked DynaMER-ADF manuscript values.
    protocol_rows = [
        {
            "Protocol": "Conventional subject-LOSO",
            "Folds": 15,
            "BA": "$0.600 \\pm 0.087$",
            "Macro-F1": "$0.584 \\pm 0.098$",
            "ROC-AUC": "$0.863 \\pm 0.039$",
            "PR-AUC": "$0.729 \\pm 0.070$",
            "Role": "Primary held-out-subject benchmark",
        }
    ]

    # Prefer Stage 22G validation-selected ADF-family if available, otherwise Stage 22 order-safe default summary if available.
    stage22_base = Path(args.subject_mixed_stage22_dir) if args.subject_mixed_stage22_dir else project_root / "outputs" / "protocol_extension" / "22_seed_iv_subject_mixed_5fold"
    stage22g = stage22_base / "validation_selected_22G_summary" / "22G_validation_selected_adf_family_summary.csv"
    stage22f = stage22_base / "capacity_audit_22F_summary" / "22F_capacity_audit_model_summary.csv"
    stage22_order = stage22_base / "summary_order_safe" / "22C_subject_mixed_5fold_model_summary.csv"
    if stage22g.exists():
        sm = pd.read_csv(stage22g)
        row = sm.iloc[0]
        protocol_rows.insert(0, {
            "Protocol": "Subject-mixed 5-fold ADF-family",
            "Folds": int(row.get("folds", row.get("n_folds", 5))) if not pd.isna(row.get("folds", row.get("n_folds", 5))) else 5,
            "BA": f"${float(row.get('BA_mean', row.get('test_balanced_accuracy_mean'))):.3f} \\pm {float(row.get('BA_std', row.get('test_balanced_accuracy_std'))):.3f}$" if "BA_mean" in row.index or "test_balanced_accuracy_mean" in row.index else "NA",
            "Macro-F1": f"${float(row.get('MacroF1_mean', row.get('test_macro_f1_mean'))):.3f} \\pm {float(row.get('MacroF1_std', row.get('test_macro_f1_std'))):.3f}$" if "MacroF1_mean" in row.index or "test_macro_f1_mean" in row.index else "NA",
            "ROC-AUC": f"${float(row.get('ROCAUC_mean', row.get('test_roc_auc_mean'))):.3f} \\pm {float(row.get('ROCAUC_std', row.get('test_roc_auc_std'))):.3f}$" if "ROCAUC_mean" in row.index or "test_roc_auc_mean" in row.index else "NA",
            "PR-AUC": f"${float(row.get('PRAUC_mean', row.get('test_pr_auc_mean'))):.3f} \\pm {float(row.get('PRAUC_std', row.get('test_pr_auc_std'))):.3f}$" if "PRAUC_mean" in row.index or "test_pr_auc_mean" in row.index else "NA",
            "Role": "Diagnostic capacity upper bound",
        })
    elif stage22_order.exists():
        sm = pd.read_csv(stage22_order)
        hit = sm[sm.get("model", sm.get("display_name", "")).astype(str).str.contains("DynaMER-ADF", na=False)] if not sm.empty else pd.DataFrame()
        if not hit.empty:
            row = hit.iloc[0]
            protocol_rows.insert(0, {
                "Protocol": "Subject-mixed 5-fold DynaMER-ADF",
                "Folds": int(row.get("folds", row.get("n_folds", 5))) if not pd.isna(row.get("folds", row.get("n_folds", 5))) else 5,
                "BA": row.get("BA", row.get("balanced_accuracy", "NA")),
                "Macro-F1": row.get("Macro-F1", row.get("macro_f1", "NA")),
                "ROC-AUC": row.get("ROC-AUC", row.get("roc_auc", "NA")),
                "PR-AUC": row.get("PR-AUC", row.get("pr_auc", "NA")),
                "Role": "Diagnostic capacity upper bound",
            })

    ns = nested_summary.iloc[0]
    protocol_rows.append({
        "Protocol": "Nested LOSO ADF-family",
        "Folds": int(ns["folds"]),
        "BA": f"${float(ns['BA_mean']):.3f} \\pm {float(ns['BA_std']):.3f}$",
        "Macro-F1": f"${float(ns['MacroF1_mean']):.3f} \\pm {float(ns['MacroF1_std']):.3f}$",
        "ROC-AUC": f"${float(ns['ROCAUC_mean']):.3f} \\pm {float(ns['ROCAUC_std']):.3f}$" if not math.isnan(float(ns['ROCAUC_mean'])) else "NA",
        "PR-AUC": f"${float(ns['PRAUC_mean']):.3f} \\pm {float(ns['PRAUC_std']):.3f}$" if not math.isnan(float(ns['PRAUC_mean'])) else "NA",
        "Role": "Bias-controlled outer held-out-subject test",
    })
    protocol_df = pd.DataFrame(protocol_rows)
    protocol_df.to_csv(out_dir / "23B_protocol_comparison_table.csv", index=False)
    write_latex_table(protocol_df, out_dir / "23B_protocol_comparison_table.tex")

    checks: List[Dict[str, Any]] = []
    add_check(checks, "outer fold metric rows", int(len(fold_df)), ">0", int(len(fold_df)) > 0)
    add_check(checks, "nested summary rows", int(len(nested_summary)), 1)
    add_check(checks, "protocol comparison rows", int(len(protocol_df)), ">=2", int(len(protocol_df)) >= 2)
    add_check(checks, "variant selection counts rows", int(len(variant_counts)), ">0", int(len(variant_counts)) > 0)
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(out_dir / "23B_nested_loso_summary_checks.csv", index=False)
    failed = checks_df[checks_df["passed"] == False]

    summary = {
        "name": "23B_summarize_nested_loso_dynamer_adf",
        "created_at": now(),
        "overall_passed": bool(len(failed) == 0),
        "nested_loso": nested_summary.to_dict(orient="records")[0],
        "selected_variant_counts": variant_counts.to_dict(orient="records"),
        "outputs": {
            "fold_metrics": str(out_dir / "23B_nested_loso_fold_metrics.csv"),
            "nested_summary": str(out_dir / "23B_nested_loso_summary.csv"),
            "selected_variant_counts": str(out_dir / "23B_nested_loso_selected_variant_counts.csv"),
            "inner_candidate_summary": str(out_dir / "23B_nested_loso_inner_candidate_summary.csv"),
            "protocol_comparison_csv": str(out_dir / "23B_protocol_comparison_table.csv"),
            "protocol_comparison_tex": str(out_dir / "23B_protocol_comparison_table.tex"),
            "checks": str(out_dir / "23B_nested_loso_summary_checks.csv"),
            "summary": str(out_dir / "23B_nested_loso_summary.json"),
        },
        "failed_checks": failed.to_dict(orient="records"),
        "scientific_statement": (
            "Use nested LOSO as robustness evidence for model-selection bias control. It should not replace the locked "
            "conventional LOSO primary result unless explicitly stated, because the nested protocol performs inner-loop "
            "candidate selection within each outer development set."
        ),
    }
    (out_dir / "23B_nested_loso_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nStage 23B outputs:")
    for i, p in enumerate(summary["outputs"].values(), start=1):
        print(f"{i}. {p}")
    print(f"\n[DONE] Stage 23B passed: {summary['overall_passed']}")
    print(f"Nested LOSO ADF-family | BA={float(ns['BA_mean']):.3f} | Macro-F1={float(ns['MacroF1_mean']):.3f}")
    return 0 if summary["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
