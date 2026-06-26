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


# ============================================================
# Stage 22G
# Validation-selected DynaMER-ADF-family aggregation for
# SEED-IV subject-mixed 5-fold capacity audit.
#
# This script DOES NOT retrain models. It uses the already
# completed Stage 22E/22F capacity-audit outputs.
#
# Scientific purpose:
#   For each subject-mixed fold, select the ADF-family candidate
#   using validation metrics only, then report that selected
#   candidate's test metrics for the fold. This estimates the
#   protocol-specific capacity of the ADF design family under
#   validation-based model selection without looking at the test
#   fold during candidate selection.
# ============================================================


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


def safe_mean(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
    return float(np.mean(arr)) if len(arr) else float("nan")


def safe_std(values: pd.Series) -> float:
    arr = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
    if len(arr) <= 1:
        return float("nan")
    return float(np.std(arr, ddof=1))


def mean_std(values: pd.Series) -> str:
    arr = pd.to_numeric(values, errors="coerce").dropna().astype(float).to_numpy()
    if len(arr) == 0:
        return "NA"
    if len(arr) == 1:
        return f"{arr[0]:.3f}"
    return f"{np.mean(arr):.3f} ± {np.std(arr, ddof=1):.3f}"


def maybe_read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def pick_best_epoch_rows_from_epochs(epochs: pd.DataFrame, monitor_metric: str) -> pd.DataFrame:
    """Fallback if Stage 22F best rows are unavailable."""
    required = [
        "capacity_variant", "fold_index", "epoch", monitor_metric,
        "test_balanced_accuracy", "test_macro_f1", "test_roc_auc", "test_pr_auc",
    ]
    missing = [c for c in required if c not in epochs.columns]
    if missing:
        raise RuntimeError(f"Cannot reconstruct best rows. Missing columns in Stage 22E epoch metrics: {missing}")

    rows = []
    for (variant, fold), group in epochs.groupby(["capacity_variant", "fold_index"], dropna=False):
        g = group.copy()
        g["_monitor"] = pd.to_numeric(g[monitor_metric], errors="coerce")
        if g["_monitor"].notna().any():
            # validation selected within candidate, earliest epoch tie-break
            idx = g.sort_values(["_monitor", "epoch"], ascending=[False, True]).index[0]
        else:
            idx = g.sort_values("epoch").index[-1]
        rows.append(g.loc[idx].drop(labels=["_monitor"], errors="ignore"))
    return pd.DataFrame(rows).sort_values(["capacity_variant", "fold_index"]).reset_index(drop=True)


def ensure_best_rows(base_dir: Path, best_rows_arg: Optional[str], epochs_arg: Optional[str], monitor_metric: str) -> pd.DataFrame:
    if best_rows_arg:
        best_rows_path = Path(best_rows_arg)
    else:
        best_rows_path = base_dir / "capacity_audit_22F_summary" / "22F_capacity_audit_best_epoch_rows.csv"

    if best_rows_path.exists():
        return pd.read_csv(best_rows_path)

    if epochs_arg:
        epochs_path = Path(epochs_arg)
    else:
        epochs_path = base_dir / "capacity_audit_22E" / "22E_capacity_audit_all_epoch_metrics.csv"

    if not epochs_path.exists():
        raise FileNotFoundError(
            "Neither Stage 22F best rows nor Stage 22E epoch metrics were found.\n"
            f"Expected best rows: {best_rows_path}\n"
            f"Expected epoch metrics: {epochs_path}"
        )
    epochs = pd.read_csv(epochs_path)
    return pick_best_epoch_rows_from_epochs(epochs, monitor_metric=monitor_metric)


def normalize_reference_summary(ref: pd.DataFrame) -> pd.DataFrame:
    if ref.empty:
        return pd.DataFrame()
    df = ref.copy()
    if "model_display" not in df.columns:
        if "capacity_display" in df.columns:
            df["model_display"] = df["capacity_display"]
        elif "model" in df.columns:
            df["model_display"] = df["model"]
        elif "model_variant" in df.columns:
            df["model_display"] = df["model_variant"]
    required_numeric = ["BA_mean", "MacroF1_mean", "ROCAUC_mean", "PRAUC_mean"]
    for col in required_numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "BA_mean" in df.columns:
        df = df.sort_values(["BA_mean", "MacroF1_mean"], ascending=[False, False]).reset_index(drop=True)
    return df


def select_variant_per_fold(best_rows: pd.DataFrame, selection_metric: str, tie_metrics: List[str]) -> pd.DataFrame:
    required = [
        "capacity_variant", "capacity_display", "fold_index", selection_metric,
        "test_balanced_accuracy", "test_macro_f1", "test_roc_auc", "test_pr_auc",
    ]
    missing = [c for c in required if c not in best_rows.columns]
    if missing:
        raise RuntimeError(f"Stage 22G input missing required columns: {missing}")

    df = best_rows.copy()
    sort_cols = []
    ascending = []
    for c in [selection_metric] + tie_metrics:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            sort_cols.append(c)
            ascending.append(False)
    if "epoch" in df.columns:
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        sort_cols.append("epoch")
        ascending.append(True)
    # Final deterministic tie-breaker, never uses test metrics.
    sort_cols.append("capacity_variant")
    ascending.append(True)

    selected = []
    for fold, group in df.groupby("fold_index", dropna=False):
        g = group.copy()
        g = g.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)
        row = g.iloc[0].copy()
        row["selection_rank_rule"] = " > ".join(sort_cols)
        selected.append(row)
    return pd.DataFrame(selected).sort_values("fold_index").reset_index(drop=True)


def summarize_selected(selected: pd.DataFrame, display_name: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "protocol": "Subject-mixed 5-fold",
        "model_or_variant": display_name,
        "folds": int(selected["fold_index"].nunique()),
        "BA": mean_std(selected["test_balanced_accuracy"]),
        "Macro-F1": mean_std(selected["test_macro_f1"]),
        "ROC-AUC": mean_std(selected["test_roc_auc"]),
        "PR-AUC": mean_std(selected["test_pr_auc"]),
        "BA_mean": safe_mean(selected["test_balanced_accuracy"]),
        "BA_std": safe_std(selected["test_balanced_accuracy"]),
        "MacroF1_mean": safe_mean(selected["test_macro_f1"]),
        "MacroF1_std": safe_std(selected["test_macro_f1"]),
        "ROCAUC_mean": safe_mean(selected["test_roc_auc"]),
        "ROCAUC_std": safe_std(selected["test_roc_auc"]),
        "PRAUC_mean": safe_mean(selected["test_pr_auc"]),
        "PRAUC_std": safe_std(selected["test_pr_auc"]),
    }])


def get_capacity_fixed_summary(base_dir: Path) -> pd.DataFrame:
    path = base_dir / "capacity_audit_22F_summary" / "22F_capacity_audit_model_summary.csv"
    return maybe_read_csv(path)


def row_from_fixed_capacity_summary(capacity_summary: pd.DataFrame, selector: str) -> Optional[Dict[str, Any]]:
    if capacity_summary.empty:
        return None
    df = capacity_summary.copy()
    if "capacity_display" not in df.columns:
        return None
    if selector == "best_fixed":
        df["BA_mean"] = pd.to_numeric(df["BA_mean"], errors="coerce")
        df["MacroF1_mean"] = pd.to_numeric(df["MacroF1_mean"], errors="coerce")
        r = df.sort_values(["BA_mean", "MacroF1_mean"], ascending=[False, False]).iloc[0]
    else:
        hit = df[df["capacity_variant"].astype(str).str.lower() == selector.lower()]
        if hit.empty:
            hit = df[df["capacity_display"].astype(str).str.lower() == selector.lower()]
        if hit.empty:
            return None
        r = hit.iloc[0]
    return {
        "protocol": "Subject-mixed 5-fold",
        "model_or_variant": str(r["capacity_display"]),
        "folds": int(r["folds"]),
        "BA": r.get("BA", "NA"),
        "Macro-F1": r.get("Macro-F1", "NA"),
        "ROC-AUC": r.get("ROC-AUC", "NA"),
        "PR-AUC": r.get("PR-AUC", "NA"),
        "BA_mean": safe_float(r.get("BA_mean", float("nan"))),
        "MacroF1_mean": safe_float(r.get("MacroF1_mean", float("nan"))),
        "ROCAUC_mean": safe_float(r.get("ROCAUC_mean", float("nan"))),
        "PRAUC_mean": safe_float(r.get("PRAUC_mean", float("nan"))),
    }


def reference_best_row(ref_summary: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if ref_summary.empty or "BA_mean" not in ref_summary.columns:
        return None
    r = ref_summary.sort_values(["BA_mean", "MacroF1_mean"], ascending=[False, False]).iloc[0]
    return {
        "protocol": "Subject-mixed 5-fold",
        "model_or_variant": str(r.get("model_display", "Reference best")),
        "folds": int(r.get("folds", 5)) if not pd.isna(r.get("folds", 5)) else 5,
        "BA": r.get("BA", "NA"),
        "Macro-F1": r.get("Macro-F1", "NA"),
        "ROC-AUC": r.get("ROC-AUC", "NA"),
        "PR-AUC": r.get("PR-AUC", "NA"),
        "BA_mean": safe_float(r.get("BA_mean", float("nan"))),
        "MacroF1_mean": safe_float(r.get("MacroF1_mean", float("nan"))),
        "ROCAUC_mean": safe_float(r.get("ROCAUC_mean", float("nan"))),
        "PRAUC_mean": safe_float(r.get("PRAUC_mean", float("nan"))),
    }


def make_comparison_table(selected_summary: pd.DataFrame, capacity_summary: pd.DataFrame, ref_summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    sel = selected_summary.iloc[0].to_dict()
    sel["group"] = "Validation-selected ADF-family"
    rows.append(sel)

    default_row = row_from_fixed_capacity_summary(capacity_summary, "v3_adf_default")
    if default_row:
        default_row["group"] = "Fixed ADF default"
        rows.append(default_row)

    best_fixed_row = row_from_fixed_capacity_summary(capacity_summary, "best_fixed")
    if best_fixed_row:
        best_fixed_row["group"] = "Best fixed ADF-family candidate"
        rows.append(best_fixed_row)

    ref_row = reference_best_row(ref_summary)
    if ref_row:
        ref_row["group"] = "Best non-ADF/reference subject-mixed model"
        rows.append(ref_row)

    loso_row = {
        "group": "Primary subject-LOSO DynaMER-ADF",
        "protocol": "Subject-LOSO",
        "model_or_variant": "DynaMER-ADF",
        "folds": 15,
        "BA": f"{args.subject_loso_ba:.3f} ± {args.subject_loso_ba_std:.3f}",
        "Macro-F1": f"{args.subject_loso_macro_f1:.3f} ± {args.subject_loso_macro_f1_std:.3f}",
        "ROC-AUC": f"{args.subject_loso_roc_auc:.3f} ± {args.subject_loso_roc_auc_std:.3f}",
        "PR-AUC": f"{args.subject_loso_pr_auc:.3f} ± {args.subject_loso_pr_auc_std:.3f}",
        "BA_mean": float(args.subject_loso_ba),
        "MacroF1_mean": float(args.subject_loso_macro_f1),
        "ROCAUC_mean": float(args.subject_loso_roc_auc),
        "PRAUC_mean": float(args.subject_loso_pr_auc),
    }
    rows.append(loso_row)

    df = pd.DataFrame(rows)
    selected_ba = safe_float(selected_summary.iloc[0]["BA_mean"])
    selected_macro = safe_float(selected_summary.iloc[0]["MacroF1_mean"])
    ref_best = reference_best_row(ref_summary)
    ref_ba = safe_float(ref_best["BA_mean"]) if ref_best else float("nan")
    ref_macro = safe_float(ref_best["MacroF1_mean"]) if ref_best else float("nan")

    df["delta_BA_vs_primary_LOSO_ADF"] = pd.to_numeric(df["BA_mean"], errors="coerce") - float(args.subject_loso_ba)
    df["delta_MacroF1_vs_primary_LOSO_ADF"] = pd.to_numeric(df["MacroF1_mean"], errors="coerce") - float(args.subject_loso_macro_f1)
    df["delta_BA_vs_subject_mixed_reference_best"] = pd.to_numeric(df["BA_mean"], errors="coerce") - ref_ba
    df["delta_MacroF1_vs_subject_mixed_reference_best"] = pd.to_numeric(df["MacroF1_mean"], errors="coerce") - ref_macro
    df["selected_adf_family_beats_or_equals_reference_BA"] = bool(selected_ba >= ref_ba) if not math.isnan(ref_ba) else None
    df["selected_adf_family_beats_or_equals_reference_MacroF1"] = bool(selected_macro >= ref_macro) if not math.isnan(ref_macro) else None
    return df


def write_latex_table(path: Path, selected_summary: pd.DataFrame, comparison_df: pd.DataFrame) -> None:
    rows = []
    # Keep report table compact: selected ADF-family, fixed default, best reference, primary LOSO.
    for _, r in comparison_df.iterrows():
        if r["group"] in [
            "Validation-selected ADF-family",
            "Fixed ADF default",
            "Best non-ADF/reference subject-mixed model",
            "Primary subject-LOSO DynaMER-ADF",
        ]:
            rows.append(r)

    lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{SEED-IV validation-selected ADF-family subject-mixed 5-fold diagnostic evaluation. Candidate selection is performed using validation metrics within each fold only, and the held-out test fold is used only for final metric computation.}",
        r"\label{tab:subject_mixed_validation_selected_adf_family}",
        r"\begin{tabular}{l l c c c c}",
        r"\hline",
        r"Protocol / group & Model or variant & Folds & BA & Macro-F1 & PR-AUC \\",
        r"\hline",
    ]
    for r in rows:
        lines.append(
            f"{r['group']} & {r['model_or_variant']} & {int(r['folds'])} & {r['BA']} & {r['Macro-F1']} & {r['PR-AUC']} \\\\"  # noqa: E501
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 22G: validation-selected ADF-family subject-mixed 5-fold aggregation.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--local-paths", default="configs/local_paths.yaml")
    parser.add_argument("--best-rows", default=None, help="Optional Stage 22F best epoch rows CSV.")
    parser.add_argument("--epochs", default=None, help="Optional Stage 22E all epoch metrics CSV fallback.")
    parser.add_argument("--reference-summary", default=None, help="Optional Stage 22C order-safe model summary CSV.")
    parser.add_argument("--selection-metric", default="val_macro_f1", help="Validation metric used to select ADF-family candidate per fold.")
    parser.add_argument("--tie-metrics", default="val_balanced_accuracy,val_roc_auc,val_pr_auc", help="Comma-separated validation-only tie-break metrics.")
    parser.add_argument("--subject-loso-ba", type=float, default=0.600)
    parser.add_argument("--subject-loso-ba-std", type=float, default=0.087)
    parser.add_argument("--subject-loso-macro-f1", type=float, default=0.584)
    parser.add_argument("--subject-loso-macro-f1-std", type=float, default=0.098)
    parser.add_argument("--subject-loso-roc-auc", type=float, default=0.863)
    parser.add_argument("--subject-loso-roc-auc-std", type=float, default=0.039)
    parser.add_argument("--subject-loso-pr-auc", type=float, default=0.729)
    parser.add_argument("--subject-loso-pr-auc-std", type=float, default=0.070)
    args = parser.parse_args()

    _main_cfg = load_yaml(Path(args.config))
    local_paths = load_yaml(Path(args.local_paths))
    project_root = as_path(local_paths["PROJECT_ROOT"])

    base_dir = project_root / "outputs" / "protocol_extension" / "22_seed_iv_subject_mixed_5fold"
    out_dir = base_dir / "validation_selected_22G_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_rows = ensure_best_rows(
        base_dir=base_dir,
        best_rows_arg=args.best_rows,
        epochs_arg=args.epochs,
        monitor_metric=args.selection_metric,
    )

    tie_metrics = [x.strip() for x in str(args.tie_metrics).split(",") if x.strip()]
    selected = select_variant_per_fold(best_rows, selection_metric=args.selection_metric, tie_metrics=tie_metrics)
    selected_summary = summarize_selected(selected, display_name="Validation-selected DynaMER-ADF family")

    capacity_summary = get_capacity_fixed_summary(base_dir)

    ref_path = Path(args.reference_summary) if args.reference_summary else base_dir / "summary_order_safe" / "22C_subject_mixed_5fold_model_summary.csv"
    ref_summary = normalize_reference_summary(maybe_read_csv(ref_path))

    comparison_df = make_comparison_table(
        selected_summary=selected_summary,
        capacity_summary=capacity_summary,
        ref_summary=ref_summary,
        args=args,
    )

    variant_counts = (
        selected.groupby(["capacity_variant", "capacity_display"], dropna=False)
        .size()
        .reset_index(name="selected_folds")
        .sort_values(["selected_folds", "capacity_display"], ascending=[False, True])
        .reset_index(drop=True)
    )

    selected_path = out_dir / "22G_validation_selected_adf_family_fold_selection.csv"
    selected_summary_path = out_dir / "22G_validation_selected_adf_family_summary.csv"
    comparison_path = out_dir / "22G_validation_selected_adf_family_vs_references.csv"
    counts_path = out_dir / "22G_validation_selected_adf_family_variant_counts.csv"
    checks_path = out_dir / "22G_validation_selected_adf_family_checks.csv"
    latex_path = out_dir / "22G_validation_selected_adf_family_table.tex"
    summary_json_path = out_dir / "22G_validation_selected_adf_family_summary.json"

    selected.to_csv(selected_path, index=False)
    selected_summary.to_csv(selected_summary_path, index=False)
    comparison_df.to_csv(comparison_path, index=False)
    variant_counts.to_csv(counts_path, index=False)
    write_latex_table(latex_path, selected_summary, comparison_df)

    selected_ba = safe_float(selected_summary.iloc[0]["BA_mean"])
    selected_macro = safe_float(selected_summary.iloc[0]["MacroF1_mean"])
    ref_row = reference_best_row(ref_summary)
    ref_name = ref_row["model_or_variant"] if ref_row else None
    ref_ba = safe_float(ref_row["BA_mean"]) if ref_row else float("nan")
    ref_macro = safe_float(ref_row["MacroF1_mean"]) if ref_row else float("nan")

    checks: List[Dict[str, Any]] = []
    add_check(checks, "best rows loaded", int(len(best_rows)), ">0", int(len(best_rows)) > 0)
    add_check(checks, "selected folds", int(selected["fold_index"].nunique()), 5)
    add_check(checks, "one selected row per fold", int(len(selected)), 5)
    add_check(checks, "selection metric exists", args.selection_metric in best_rows.columns, True)
    add_check(checks, "reference summary found", bool(ref_path.exists()), True)
    add_check(checks, "no test metrics used for selection", "test" not in args.selection_metric.lower() and all("test" not in m.lower() for m in tie_metrics), True)
    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(checks_path, index=False)
    failed = checks_df[checks_df["passed"] == False]
    overall_passed = len(failed) == 0

    summary = {
        "name": "22G_validation_selected_adf_family_subject_mixed",
        "created_at": now(),
        "overall_passed": bool(overall_passed),
        "selection_metric": args.selection_metric,
        "tie_metrics": tie_metrics,
        "selected_adf_family": {
            "BA_mean": selected_ba,
            "MacroF1_mean": selected_macro,
            "ROC_AUC_mean": safe_float(selected_summary.iloc[0]["ROCAUC_mean"]),
            "PR_AUC_mean": safe_float(selected_summary.iloc[0]["PRAUC_mean"]),
            "delta_BA_vs_primary_LOSO_ADF": selected_ba - float(args.subject_loso_ba),
            "delta_MacroF1_vs_primary_LOSO_ADF": selected_macro - float(args.subject_loso_macro_f1),
        },
        "reference_best": {
            "model": ref_name,
            "BA_mean": ref_ba,
            "MacroF1_mean": ref_macro,
        },
        "selected_adf_family_beats_or_equals_reference_by_BA": bool(selected_ba >= ref_ba) if not math.isnan(ref_ba) else None,
        "selected_adf_family_beats_or_equals_reference_by_MacroF1": bool(selected_macro >= ref_macro) if not math.isnan(ref_macro) else None,
        "selected_variant_counts": variant_counts.to_dict(orient="records"),
        "outputs": {
            "fold_selection": str(selected_path),
            "selected_summary": str(selected_summary_path),
            "comparison": str(comparison_path),
            "variant_counts": str(counts_path),
            "latex_table": str(latex_path),
            "checks": str(checks_path),
            "summary": str(summary_json_path),
        },
        "failed_checks": failed.to_dict(orient="records"),
        "scientific_statement": "Stage 22G uses validation-only candidate selection within each subject-mixed fold and reports test metrics only after the candidate is selected. This is a diagnostic capacity estimate for the ADF design family, not the primary subject-independent claim.",
    }
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nStage 22G outputs:")
    print(f"1. {selected_path}")
    print(f"2. {selected_summary_path}")
    print(f"3. {comparison_path}")
    print(f"4. {counts_path}")
    print(f"5. {latex_path}")
    print(f"6. {checks_path}")
    print(f"7. {summary_json_path}")

    if not overall_passed:
        print("\n[ERROR] Stage 22G failed checks. Inspect checks CSV before manuscript use.")
        return 1

    print("\n[DONE] Stage 22G passed.")
    print(f"Validation-selected ADF-family | BA={selected_ba:.3f} | Macro-F1={selected_macro:.3f}")
    if ref_name is not None:
        print(f"Reference best: {ref_name} | BA={ref_ba:.3f} | Macro-F1={ref_macro:.3f}")
        print(f"Selected ADF-family beats/equal reference by BA: {bool(selected_ba >= ref_ba)}")
    print(f"Gap vs primary LOSO ADF: BA={selected_ba - float(args.subject_loso_ba):+.3f}, Macro-F1={selected_macro - float(args.subject_loso_macro_f1):+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
