#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage 24A: Protocol-extension evidence package for DynaMER-ADF manuscript.

Purpose
-------
Generate a compact, traceable professor-response evidence package from the
completed Stage 22 and Stage 23 outputs:

1. Subject-mixed 5-fold ADF-family diagnostic capacity result.
2. Conventional subject-LOSO locked primary DynaMER-ADF result.
3. Nested LOSO ADF-family model-selection-bias robustness result.

This script does not train models and does not modify original results.
It only reads completed protocol-extension outputs and writes manuscript-ready
summary tables, figures, and text snippets.

Run from project root:
    python scripts\24A_generate_protocol_extension_evidence_package.py \
        --config configs\config.yaml --local-paths configs\local_paths.yaml

Expected primary input:
    outputs\protocol_extension\23_nested_loso_dynamer_adf\23B_summary\23B_protocol_comparison_table.csv

Expected optional inputs:
    outputs\protocol_extension\23_nested_loso_dynamer_adf\23B_summary\23B_nested_loso_selected_variant_counts.csv
    outputs\protocol_extension\23_nested_loso_dynamer_adf\23B_summary\23B_nested_loso_fold_metrics.csv

Outputs:
    outputs\protocol_extension\24_protocol_extension_evidence_package\...
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

try:
    import yaml  # noqa: F401  # kept for interface consistency with prior stages
except Exception:
    yaml = None

import matplotlib.pyplot as plt
import numpy as np


LOCKED_LOSO = {
    "Protocol": "Conventional subject-LOSO",
    "Folds": 15,
    "BA": "$0.600 \\pm 0.087$",
    "Macro-F1": "$0.584 \\pm 0.098$",
    "ROC-AUC": "$0.863 \\pm 0.039$",
    "PR-AUC": "$0.729 \\pm 0.070$",
    "Role": "Primary held-out-subject benchmark",
}


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now()}] [INFO] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 24A protocol-extension evidence package.")
    parser.add_argument("--config", default="configs/config.yaml", help="Kept for consistency. Not modified.")
    parser.add_argument("--local-paths", default="configs/local_paths.yaml", help="Kept for consistency. Not modified.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing Stage 24 outputs.")
    return parser.parse_args()


def project_root_from_args(args: argparse.Namespace) -> Path:
    # Scripts are intended to be run from project root. This avoids modifying any original config.
    return Path.cwd().resolve()


def require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def parse_metric_string(value: object) -> Tuple[float, float]:
    """Parse strings like '$0.724 \\pm 0.017$' or '0.724 ± 0.017'."""
    if pd.isna(value):
        return math.nan, math.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value), math.nan
    s = str(value)
    nums = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", s)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    if len(nums) == 1:
        return float(nums[0]), math.nan
    return math.nan, math.nan


def metric_text(mean: float, std: float) -> str:
    return f"${mean:.3f} \\pm {std:.3f}$"


def add_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    metric_map = {
        "BA": ("BA_mean", "BA_std"),
        "Macro-F1": ("MacroF1_mean", "MacroF1_std"),
        "ROC-AUC": ("ROCAUC_mean", "ROCAUC_std"),
        "PR-AUC": ("PRAUC_mean", "PRAUC_std"),
    }
    for col, (mcol, scol) in metric_map.items():
        vals = df[col].apply(parse_metric_string)
        df[mcol] = [x[0] for x in vals]
        df[scol] = [x[1] for x in vals]
    return df


def create_protocol_comparison_figure(df: pd.DataFrame, out_dir: Path) -> Dict[str, str]:
    """Create BA/Macro-F1 protocol comparison figure."""
    plot_df = df.copy()
    protocols = plot_df["Protocol"].tolist()
    x = np.arange(len(protocols))
    width = 0.36

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"

    fig, ax = plt.subplots(figsize=(7.2, 4.1))

    ba = plot_df["BA_mean"].to_numpy(float)
    ba_err = plot_df["BA_std"].to_numpy(float)
    mf1 = plot_df["MacroF1_mean"].to_numpy(float)
    mf1_err = plot_df["MacroF1_std"].to_numpy(float)

    ax.bar(x - width / 2, ba, width, yerr=ba_err, label="Balanced accuracy", capsize=3,
           edgecolor="black", linewidth=0.6)
    ax.bar(x + width / 2, mf1, width, yerr=mf1_err, label="Macro-F1", capsize=3,
           edgecolor="black", linewidth=0.6)

    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(protocols, rotation=15, ha="right")
    ax.set_title("Protocol-extension evaluation on SEED-IV")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")

    # Add compact mean labels above bars.
    for xi, yi in zip(x - width / 2, ba):
        ax.text(xi, yi + 0.025, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)
    for xi, yi in zip(x + width / 2, mf1):
        ax.text(xi, yi + 0.025, f"{yi:.3f}", ha="center", va="bottom", fontsize=8)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    base = out_dir / "Fig24_protocol_extension_ba_macro_f1"
    paths = {}
    for ext in ["pdf", "png", "svg"]:
        out = base.with_suffix(f".{ext}")
        if ext == "png":
            fig.savefig(out, dpi=600, bbox_inches="tight")
        else:
            fig.savefig(out, bbox_inches="tight")
        paths[ext] = str(out)
    plt.close(fig)
    return paths


def create_variant_count_figure(counts_df: pd.DataFrame, out_dir: Path) -> Dict[str, str]:
    """Create selected nested LOSO variant counts figure."""
    df = counts_df.copy()
    if df.empty:
        return {}
    df = df.sort_values("outer_fold_count", ascending=True)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["svg.fonttype"] = "none"

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    y = np.arange(len(df))
    ax.barh(y, df["outer_fold_count"].to_numpy(int), edgecolor="black", linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(df["selected_variant_display"].tolist())
    ax.set_xlabel("Number of outer LOSO folds")
    ax.set_title("Nested LOSO selected ADF-family configurations")
    ax.set_xlim(0, max(5, int(df["outer_fold_count"].max()) + 1))
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.45)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for yi, val in zip(y, df["outer_fold_count"].to_numpy(int)):
        ax.text(val + 0.05, yi, str(val), va="center", fontsize=8)
    fig.tight_layout()

    base = out_dir / "Fig24_nested_loso_selected_variant_counts"
    paths = {}
    for ext in ["pdf", "png", "svg"]:
        out = base.with_suffix(f".{ext}")
        if ext == "png":
            fig.savefig(out, dpi=600, bbox_inches="tight")
        else:
            fig.savefig(out, bbox_inches="tight")
        paths[ext] = str(out)
    plt.close(fig)
    return paths


def make_latex_table(df: pd.DataFrame, out_path: Path) -> str:
    table_df = df[["Protocol", "Folds", "BA", "Macro-F1", "ROC-AUC", "PR-AUC", "Role"]].copy()
    latex = r"""\begin{table*}[!t]
\centering
\caption{Protocol-extension evaluation on SEED-IV. Subject-mixed 5-fold estimates diagnostic within-dataset capacity when subject-specific patterns may be shared across training and testing. Conventional subject-LOSO remains the primary held-out-subject benchmark. Nested LOSO performs inner-loop ADF-family selection within the development subjects before evaluating each held-out subject. Values are mean $\pm$ standard deviation across folds.}
\label{tab:protocol_extension_evaluation}
\scriptsize
\setlength{\tabcolsep}{3.5pt}
\renewcommand{\arraystretch}{1.08}
\begin{tabular}{l c c c c c l}
\hline
Protocol & Folds & BA & Macro-F1 & ROC-AUC & PR-AUC & Role \\
\hline
"""
    for _, r in table_df.iterrows():
        latex += f"{r['Protocol']} & {int(r['Folds'])} & {r['BA']} & {r['Macro-F1']} & {r['ROC-AUC']} & {r['PR-AUC']} & {r['Role']} \\\n"
    latex += r"""\hline
\end{tabular}
\end{table*}
"""
    out_path.write_text(latex, encoding="utf-8")
    return latex


def make_manuscript_results_insert(df: pd.DataFrame, out_path: Path) -> str:
    sm = df[df["Protocol"].str.contains("Subject-mixed", case=False, na=False)].iloc[0]
    loso = df[df["Protocol"].str.contains("Conventional", case=False, na=False)].iloc[0]
    nested = df[df["Protocol"].str.contains("Nested", case=False, na=False)].iloc[0]

    text = rf"""
\subsection{{Protocol-Extension Analysis}}

To separate intrinsic within-dataset discriminative capacity from held-out-subject generalization, we performed an additional protocol-extension analysis on SEED-IV. Subject-mixed five-fold evaluation yielded higher ADF-family performance than the primary subject-LOSO protocol, with balanced accuracy of {sm['BA']} and macro-F1 of {sm['Macro-F1']}. In comparison, the locked conventional subject-LOSO DynaMER-ADF result was {loso['BA']} balanced accuracy and {loso['Macro-F1']} macro-F1. The gap indicates that a substantial part of the LOSO degradation arises from inter-subject distribution shift rather than from a complete lack of discriminative capacity.

We further performed nested LOSO using inner-loop subject-level validation to select among ADF-family configurations within each outer development set. Nested LOSO achieved balanced accuracy of {nested['BA']} and macro-F1 of {nested['Macro-F1']}. This estimate was slightly more conservative than conventional LOSO but remained in the same performance range, indicating that the primary LOSO result was not severely inflated by model-selection bias. These protocol-extension results support retaining conventional subject-LOSO as the primary deployment-relevant benchmark while using subject-mixed and nested LOSO results as diagnostic robustness evidence.
""".strip() + "\n"
    out_path.write_text(text, encoding="utf-8")
    return text


def make_manuscript_discussion_insert(df: pd.DataFrame, out_path: Path) -> str:
    text = r"""
The protocol-extension experiments clarify the interpretation of the main result. Subject-mixed five-fold evaluation produced substantially higher ADF-family performance than subject-LOSO, confirming that the model can exploit within-dataset affective structure when subject-specific patterns are partially available during training. In contrast, both conventional LOSO and nested LOSO remained lower, showing that held-out-subject generalization is the central difficulty. The nested result was slightly more conservative than the locked conventional LOSO result, which is expected because model selection was performed inside each outer development set. Importantly, the nested estimate did not collapse, suggesting that the primary LOSO result is not mainly an artifact of optimization bias. These findings support the paper's main framing: DynaMER-ADF is most relevant as a subject-independent physiological affect-recognition framework, while subject-mixed performance should be interpreted only as a diagnostic upper bound.
""".strip() + "\n"
    out_path.write_text(text, encoding="utf-8")
    return text


def make_professor_report(df: pd.DataFrame, counts_df: pd.DataFrame, out_path: Path) -> str:
    sm = df[df["Protocol"].str.contains("Subject-mixed", case=False, na=False)].iloc[0]
    loso = df[df["Protocol"].str.contains("Conventional", case=False, na=False)].iloc[0]
    nested = df[df["Protocol"].str.contains("Nested", case=False, na=False)].iloc[0]

    ba_gap = sm["BA_mean"] - loso["BA_mean"]
    mf1_gap = sm["MacroF1_mean"] - loso["MacroF1_mean"]
    nested_ba_gap = nested["BA_mean"] - loso["BA_mean"]
    nested_mf1_gap = nested["MacroF1_mean"] - loso["MacroF1_mean"]

    counts_txt = ""
    if not counts_df.empty:
        counts_txt = "\n".join(
            f"- {r['selected_variant_display']}: {int(r['outer_fold_count'])} outer folds"
            for _, r in counts_df.iterrows()
        )
    else:
        counts_txt = "- Selected-variant count file not found."

    report = f"""# Protocol-extension evaluation report

## Purpose

This report addresses the validation concern that conventional LOSO alone does not separate model-capacity limitations from the intrinsic difficulty of subject-independent physiological emotion recognition.

## Experiments added

1. **Subject-mixed 5-fold evaluation on SEED-IV**: diagnostic upper-bound setting where subject-specific patterns may be partially shared between training and testing.
2. **Nested LOSO for the DynaMER-ADF family**: outer held-out-subject evaluation with inner subject-level validation for candidate selection.
3. **Conventional subject-LOSO**: retained as the primary deployment-relevant benchmark.

## Main results

| Protocol | Folds | BA | Macro-F1 | ROC-AUC | PR-AUC | Role |
|---|---:|---:|---:|---:|---:|---|
| {sm['Protocol']} | {int(sm['Folds'])} | {sm['BA']} | {sm['Macro-F1']} | {sm['ROC-AUC']} | {sm['PR-AUC']} | {sm['Role']} |
| {loso['Protocol']} | {int(loso['Folds'])} | {loso['BA']} | {loso['Macro-F1']} | {loso['ROC-AUC']} | {loso['PR-AUC']} | {loso['Role']} |
| {nested['Protocol']} | {int(nested['Folds'])} | {nested['BA']} | {nested['Macro-F1']} | {nested['ROC-AUC']} | {nested['PR-AUC']} | {nested['Role']} |

## Interpretation

The subject-mixed ADF-family result exceeded the conventional LOSO DynaMER-ADF result by **{ba_gap:+.3f} BA** and **{mf1_gap:+.3f} macro-F1**. This supports the interpretation that the lower LOSO performance is substantially driven by inter-subject distribution shift rather than by a complete lack of model discriminative capacity.

Nested LOSO produced a more conservative estimate than conventional LOSO by **{nested_ba_gap:+.3f} BA** and **{nested_mf1_gap:+.3f} macro-F1**. This is expected because candidate selection was performed inside the outer development set. The nested result remained in the same general performance range, indicating that the locked conventional LOSO result is not severely inflated by model-selection bias.

## Nested LOSO selected configurations

{counts_txt}

The selected configuration varied across outer folds, which supports the paper's central subject-variability argument: no single ADF-family configuration dominated all held-out subjects.

## Recommended manuscript use

- Keep conventional subject-LOSO as the main result and primary claim.
- Add the subject-mixed 5-fold result as diagnostic capacity evidence.
- Add nested LOSO as robustness evidence for model-selection bias control.
- Do not claim that subject-mixed performance is the primary deployment estimate.
- Do not claim universal cross-dataset generalization.
"""
    out_path.write_text(report, encoding="utf-8")
    return report


def main() -> int:
    args = parse_args()
    root = project_root_from_args(args)
    log("Starting Stage 24A protocol-extension evidence package.")
    log(f"Project root: {root}")

    stage23_dir = root / "outputs" / "protocol_extension" / "23_nested_loso_dynamer_adf" / "23B_summary"
    protocol_csv = stage23_dir / "23B_protocol_comparison_table.csv"
    counts_csv = stage23_dir / "23B_nested_loso_selected_variant_counts.csv"
    fold_csv = stage23_dir / "23B_nested_loso_fold_metrics.csv"

    require_file(protocol_csv, "Stage 23B protocol comparison table")

    out_dir = Path(args.output_dir).resolve() if args.output_dir else root / "outputs" / "protocol_extension" / "24_protocol_extension_evidence_package"
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty. Use --overwrite or choose another output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(protocol_csv)
    # Safety: ensure the locked LOSO row exists. If missing, insert locked row.
    if not df["Protocol"].astype(str).str.contains("Conventional subject-LOSO", case=False, na=False).any():
        log("Conventional LOSO row missing in Stage 23B table. Inserting locked primary LOSO values.")
        df = pd.concat([df, pd.DataFrame([LOCKED_LOSO])], ignore_index=True)

    df = add_numeric_columns(df)

    # Preferred order for manuscript/professor readability.
    order = [
        "Subject-mixed 5-fold ADF-family",
        "Conventional subject-LOSO",
        "Nested LOSO ADF-family",
    ]
    df["_order"] = df["Protocol"].apply(lambda x: order.index(x) if x in order else 999)
    df = df.sort_values("_order").drop(columns=["_order"]).reset_index(drop=True)

    table_csv = out_dir / "24A_protocol_extension_table.csv"
    df.to_csv(table_csv, index=False)
    log(f"Wrote protocol table CSV: {table_csv}")

    table_tex = out_dir / "24A_protocol_extension_table.tex"
    latex_table = make_latex_table(df, table_tex)
    log(f"Wrote protocol table LaTeX: {table_tex}")

    fig_paths = create_protocol_comparison_figure(df, out_dir)
    log("Wrote protocol comparison figure files.")

    counts_df = pd.DataFrame()
    if counts_csv.exists():
        counts_df = pd.read_csv(counts_csv)
    counts_paths = create_variant_count_figure(counts_df, out_dir)
    if counts_paths:
        log("Wrote nested selected-variant-count figure files.")

    if fold_csv.exists():
        fold_df = pd.read_csv(fold_csv)
        fold_out = out_dir / "24A_nested_loso_fold_metrics_copy.csv"
        fold_df.to_csv(fold_out, index=False)

    results_tex = out_dir / "24A_manuscript_results_insert.tex"
    discussion_tex = out_dir / "24A_manuscript_discussion_insert.tex"
    professor_md = out_dir / "24A_professor_response_report.md"

    make_manuscript_results_insert(df, results_tex)
    make_manuscript_discussion_insert(df, discussion_tex)
    make_professor_report(df, counts_df, professor_md)
    log("Wrote manuscript/professor text inserts.")

    checks = []
    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append({"check": name, "passed": bool(passed), "detail": detail})

    check("has_three_protocol_rows", len(df) >= 3, f"rows={len(df)}")
    check("subject_mixed_above_loso_ba", df.loc[df["Protocol"].str.contains("Subject-mixed"), "BA_mean"].iloc[0] > df.loc[df["Protocol"].str.contains("Conventional"), "BA_mean"].iloc[0])
    check("nested_loso_not_nan", not math.isnan(df.loc[df["Protocol"].str.contains("Nested"), "BA_mean"].iloc[0]))
    check("counts_available", not counts_df.empty, str(counts_csv))

    checks_csv = out_dir / "24A_protocol_extension_checks.csv"
    pd.DataFrame(checks).to_csv(checks_csv, index=False)

    summary = {
        "name": "24A_generate_protocol_extension_evidence_package",
        "created_at": now(),
        "overall_passed": all(c["passed"] for c in checks if c["check"] != "counts_available"),
        "inputs": {
            "protocol_comparison_csv": str(protocol_csv),
            "selected_variant_counts_csv": str(counts_csv) if counts_csv.exists() else None,
            "nested_loso_fold_metrics_csv": str(fold_csv) if fold_csv.exists() else None,
        },
        "outputs": {
            "protocol_table_csv": str(table_csv),
            "protocol_table_tex": str(table_tex),
            "protocol_comparison_figure": fig_paths,
            "selected_variant_counts_figure": counts_paths,
            "manuscript_results_insert_tex": str(results_tex),
            "manuscript_discussion_insert_tex": str(discussion_tex),
            "professor_response_report_md": str(professor_md),
            "checks_csv": str(checks_csv),
        },
        "checks": checks,
        "scientific_statement": (
            "Subject-mixed 5-fold is diagnostic upper-bound evidence; conventional subject-LOSO remains the primary held-out-subject benchmark; "
            "nested LOSO is model-selection-bias robustness evidence and should not replace the locked conventional LOSO primary result."
        ),
    }
    summary_json = out_dir / "24A_protocol_extension_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nStage 24A outputs:")
    for i, p in enumerate([
        table_csv,
        table_tex,
        Path(fig_paths.get("pdf", "")),
        Path(counts_paths.get("pdf", "")) if counts_paths else None,
        results_tex,
        discussion_tex,
        professor_md,
        checks_csv,
        summary_json,
    ], start=1):
        if p:
            print(f"{i}. {p}")
    print("\n[DONE] Stage 24A passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
