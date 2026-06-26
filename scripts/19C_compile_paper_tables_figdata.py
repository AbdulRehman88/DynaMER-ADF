
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def parse_mean(x):
    s = str(x)
    m = re.search(r"[-+]?\d*\.?\d+", s)
    return float(m.group(0)) if m else np.nan


def fmt3(x):
    if pd.isna(x):
        return "NA"
    return f"{float(x):.3f}"


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def normalize_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    rename = {
        "test_balanced_accuracy_mean_std": "BA",
        "test_macro_f1_mean_std": "MacroF1",
        "test_roc_auc_mean_std": "ROCAUC",
        "test_pr_auc_mean_std": "PRAUC",
        "baseline_variant": "variant",
    }
    df = df.rename(columns=rename)

    for col in ["BA", "MacroF1", "ROCAUC", "PRAUC"]:
        if col in df.columns:
            df[col + "_mean"] = df[col].map(parse_mean)

    return df


def add_clean_metric_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["BA_mean", "MacroF1_mean", "ROCAUC_mean", "PRAUC_mean"]:
        if col in df.columns:
            df[col.replace("_mean", "_value")] = df[col].map(fmt3)
    return df


def main() -> int:
    project_root = Path(".").resolve()

    out_dir = project_root / "outputs" / "ablations" / "19C_paper_ready_tables_figdata"
    tables_dir = out_dir / "tables"
    figdata_dir = out_dir / "figure_data"
    evidence_dir = out_dir / "evidence_index"
    lock_dir = project_root / "outputs" / "final_locked_results" / "stage19C_paper_ready_tables_figdata"

    for d in [out_dir, tables_dir, figdata_dir, evidence_dir, lock_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Required input files
    # ------------------------------------------------------------
    paths = {
        "stage19A_all_models": project_root / "outputs" / "ablations" / "19_reviewer_proof_ablation_suite" / "19A_all_seed_iv_model_and_architecture_ablation.csv",
        "stage19A_validation": project_root / "outputs" / "ablations" / "19_reviewer_proof_ablation_suite" / "19B_dynamer_v3_cross_dataset_validation_ablation.csv",
        "stage19A_winners": project_root / "outputs" / "ablations" / "19_reviewer_proof_ablation_suite" / "19C_seed_iv_winners_by_protocol_and_metric.csv",
        "stage19A_claim_matrix": project_root / "outputs" / "ablations" / "19_reviewer_proof_ablation_suite" / "19I_reviewer_attack_defense_claim_matrix.csv",
        "stage19B_ablation_compact": project_root / "outputs" / "ablations" / "19B_dynamer_v3_component_ablation_summary" / "19B_ablation_results_paper_compact.csv",
        "stage19B_importance": project_root / "outputs" / "ablations" / "19B_dynamer_v3_component_ablation_summary" / "19B_component_importance_delta_vs_final_v3.csv",
        "stage19B_rankings": project_root / "outputs" / "ablations" / "19B_dynamer_v3_component_ablation_summary" / "19B_ablation_rankings_by_protocol_metric.csv",
        "stage19B_best_epochs": project_root / "outputs" / "ablations" / "19B_dynamer_v3_component_ablation_summary" / "19B_ablation_best_epoch_metrics.csv",
        "stage19B_checks": project_root / "outputs" / "ablations" / "19B_dynamer_v3_component_ablation_training" / "19B_dynamer_v3_ablation_training_checks.csv",
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required input {name}: {path}")

    checks = read_csv_required(paths["stage19B_checks"])
    if not bool(checks["passed"].all()):
        raise RuntimeError("Stage 19B checks did not all pass. Do not compile Stage 19C.")

    all_models = normalize_metric_columns(read_csv_required(paths["stage19A_all_models"]))
    validation = normalize_metric_columns(read_csv_required(paths["stage19A_validation"]))
    winners = read_csv_required(paths["stage19A_winners"])
    claim_matrix = read_csv_required(paths["stage19A_claim_matrix"])
    ablation = normalize_metric_columns(read_csv_required(paths["stage19B_ablation_compact"]))
    importance = read_csv_required(paths["stage19B_importance"])
    rankings = read_csv_required(paths["stage19B_rankings"])
    best_epochs = read_csv_required(paths["stage19B_best_epochs"])

    all_models = add_clean_metric_labels(all_models)
    validation = add_clean_metric_labels(validation)
    ablation = add_clean_metric_labels(ablation)

    # ------------------------------------------------------------
    # TABLE 1: Final DynaMER-v3 validation across datasets
    # ------------------------------------------------------------
    table1_cols = [
        "model", "dataset", "task", "protocol", "folds",
        "BA", "MacroF1", "ROCAUC", "PRAUC",
    ]
    table1 = validation[[c for c in table1_cols if c in validation.columns]].copy()
    table1 = table1.sort_values(["dataset", "task", "protocol"]).reset_index(drop=True)
    table1.to_csv(tables_dir / "Table1_final_dynamer_v3_validation.csv", index=False)

    # ------------------------------------------------------------
    # TABLE 2: SEED-IV model/baseline/architecture comparison
    # ------------------------------------------------------------
    table2 = all_models.copy()
    table2 = table2[[
        "model", "variant", "dataset", "task", "protocol", "folds",
        "BA", "MacroF1", "ROCAUC", "PRAUC",
        "BA_mean", "MacroF1_mean", "ROCAUC_mean", "PRAUC_mean",
    ]]
    table2 = table2.sort_values(["protocol", "BA_mean"], ascending=[True, False]).reset_index(drop=True)
    table2.to_csv(tables_dir / "Table2_seed_iv_model_baseline_architecture_comparison.csv", index=False)

    # ------------------------------------------------------------
    # TABLE 3: Core component ablation compact table
    # ------------------------------------------------------------
    table3 = ablation.copy()
    table3_cols = [
        "ablation_category", "variant", "dataset", "task", "protocol", "folds",
        "BA", "MacroF1", "ROCAUC", "PRAUC",
        "parameter_count_mean",
    ]
    table3 = table3[[c for c in table3_cols if c in table3.columns]]
    table3 = table3.sort_values(["protocol", "ablation_category", "variant"]).reset_index(drop=True)
    table3.to_csv(tables_dir / "Table3_component_ablation_full_compact.csv", index=False)

    # ------------------------------------------------------------
    # TABLE 4: Subject-LOSO component ablation only
    # ------------------------------------------------------------
    table4 = ablation[ablation["protocol"].astype(str).eq("subject_loso")].copy()
    table4 = table4.sort_values(["BA_mean"], ascending=False).reset_index(drop=True)
    table4.to_csv(tables_dir / "Table4_subject_loso_component_ablation_ranked.csv", index=False)

    # ------------------------------------------------------------
    # TABLE 5: Modality ablation
    # ------------------------------------------------------------
    table5 = ablation[ablation["ablation_category"].astype(str).eq("modality")].copy()
    table5 = table5.sort_values(["protocol", "BA_mean"], ascending=[True, False]).reset_index(drop=True)
    table5.to_csv(tables_dir / "Table5_modality_ablation.csv", index=False)

    # ------------------------------------------------------------
    # TABLE 6: Temporal / fusion / head / regularization ablation
    # ------------------------------------------------------------
    table6 = ablation[~ablation["ablation_category"].astype(str).eq("modality")].copy()
    table6 = table6.sort_values(["protocol", "ablation_category", "BA_mean"], ascending=[True, True, False]).reset_index(drop=True)
    table6.to_csv(tables_dir / "Table6_architecture_component_ablation.csv", index=False)

    # ------------------------------------------------------------
    # TABLE 7: Component importance delta vs final v3
    # ------------------------------------------------------------
    table7 = importance.copy()
    for col in [
        "delta_BA_vs_final_v3",
        "delta_MacroF1_vs_final_v3",
        "delta_ROCAUC_vs_final_v3",
        "delta_PRAUC_vs_final_v3",
    ]:
        table7[col] = pd.to_numeric(table7[col], errors="coerce")
    table7 = table7.sort_values(["protocol", "delta_BA_vs_final_v3"], ascending=[True, False]).reset_index(drop=True)
    table7.to_csv(tables_dir / "Table7_component_importance_delta_vs_final_v3.csv", index=False)

    # ------------------------------------------------------------
    # TABLE 8: Claim-defense evidence matrix
    # ------------------------------------------------------------
    claim_matrix.to_csv(tables_dir / "Table8_reviewer_claim_defense_matrix.csv", index=False)

    # ------------------------------------------------------------
    # FIGURE DATA 1: DynaMER-v3 validation bar plot
    # ------------------------------------------------------------
    fig1 = validation.copy()
    fig1["label"] = fig1["dataset"].astype(str) + " | " + fig1["task"].astype(str) + " | " + fig1["protocol"].astype(str)
    fig1 = fig1[[
        "label", "dataset", "task", "protocol", "folds",
        "BA_mean", "MacroF1_mean", "ROCAUC_mean", "PRAUC_mean",
    ]]
    fig1.to_csv(figdata_dir / "Fig1_dynamer_v3_validation_barplot_data.csv", index=False)

    # ------------------------------------------------------------
    # FIGURE DATA 2: SEED-IV model comparison, BA/Macro-F1 grouped bars
    # ------------------------------------------------------------
    fig2 = all_models.copy()
    fig2["model_label"] = fig2["model"].astype(str) + " / " + fig2["variant"].astype(str)
    fig2 = fig2[[
        "model_label", "model", "variant", "protocol",
        "BA_mean", "MacroF1_mean", "ROCAUC_mean", "PRAUC_mean",
    ]]
    fig2 = fig2.sort_values(["protocol", "BA_mean"], ascending=[True, False])
    fig2.to_csv(figdata_dir / "Fig2_seed_iv_model_comparison_grouped_bar_data.csv", index=False)

    # ------------------------------------------------------------
    # FIGURE DATA 3: Component delta heatmap
    # ------------------------------------------------------------
    fig3 = importance.copy()
    fig3["variant_label"] = fig3["ablation_category"].astype(str) + " | " + fig3["variant"].astype(str)
    fig3 = fig3[[
        "protocol", "ablation_category", "variant", "variant_label",
        "delta_BA_vs_final_v3",
        "delta_MacroF1_vs_final_v3",
        "delta_ROCAUC_vs_final_v3",
        "delta_PRAUC_vs_final_v3",
    ]]
    fig3.to_csv(figdata_dir / "Fig3_component_delta_heatmap_data.csv", index=False)

    # ------------------------------------------------------------
    # FIGURE DATA 4: Modality ablation
    # ------------------------------------------------------------
    fig4 = ablation[ablation["ablation_category"].astype(str).eq("modality")].copy()
    fig4["modality_variant"] = fig4["variant"].astype(str)
    fig4 = fig4[[
        "protocol", "modality_variant", "BA_mean", "MacroF1_mean", "ROCAUC_mean", "PRAUC_mean",
    ]]
    fig4.to_csv(figdata_dir / "Fig4_modality_ablation_data.csv", index=False)

    # ------------------------------------------------------------
    # FIGURE DATA 5: Subject robustness vs session robustness scatter
    # ------------------------------------------------------------
    tmp = all_models.copy()
    scatter_rows = []
    for variant, g in tmp.groupby(["model", "variant"], dropna=False):
        model, var = variant
        cross = g[g["protocol"].astype(str).eq("cross_session")]
        loso = g[g["protocol"].astype(str).eq("subject_loso")]
        if len(cross) and len(loso):
            scatter_rows.append({
                "model": model,
                "variant": var,
                "cross_session_BA": float(cross["BA_mean"].iloc[0]),
                "subject_loso_BA": float(loso["BA_mean"].iloc[0]),
                "cross_session_MacroF1": float(cross["MacroF1_mean"].iloc[0]),
                "subject_loso_MacroF1": float(loso["MacroF1_mean"].iloc[0]),
            })
    fig5 = pd.DataFrame(scatter_rows)
    fig5.to_csv(figdata_dir / "Fig5_session_vs_subject_robustness_scatter_data.csv", index=False)

    # ------------------------------------------------------------
    # FIGURE DATA 6: Fold distribution data for boxplots
    # ------------------------------------------------------------
    fig6 = best_epochs.copy()
    fig6_cols = [
        "baseline_variant", "ablation_category", "dataset", "task", "protocol", "fold_index",
        "test_balanced_accuracy", "test_macro_f1", "test_roc_auc", "test_pr_auc",
    ]
    fig6 = fig6[[c for c in fig6_cols if c in fig6.columns]]
    fig6.to_csv(figdata_dir / "Fig6_fold_distribution_boxplot_data.csv", index=False)

    # ------------------------------------------------------------
    # FIGURE DATA 7: Parameter count vs performance
    # ------------------------------------------------------------
    fig7 = ablation.copy()
    if "parameter_count_mean" in fig7.columns:
        fig7["parameter_count_million"] = pd.to_numeric(fig7["parameter_count_mean"], errors="coerce") / 1_000_000.0
    else:
        fig7["parameter_count_million"] = np.nan
    fig7 = fig7[[
        "ablation_category", "variant", "protocol",
        "parameter_count_million", "BA_mean", "MacroF1_mean", "ROCAUC_mean", "PRAUC_mean",
    ]]
    fig7.to_csv(figdata_dir / "Fig7_parameter_count_vs_performance_data.csv", index=False)

    # ------------------------------------------------------------
    # Evidence index
    # ------------------------------------------------------------
    evidence_rows = [
        {
            "artifact": "Table1_final_dynamer_v3_validation.csv",
            "purpose": "Main final validation table across SEED-IV, DREAMER, AMIGOS.",
            "recommended_manuscript_location": "Results: Overall validation performance.",
        },
        {
            "artifact": "Table2_seed_iv_model_baseline_architecture_comparison.csv",
            "purpose": "Main baseline and DynaMER variant comparison.",
            "recommended_manuscript_location": "Results: Comparison with temporal baselines.",
        },
        {
            "artifact": "Table3_component_ablation_full_compact.csv",
            "purpose": "Complete component ablation table.",
            "recommended_manuscript_location": "Supplementary or main ablation table.",
        },
        {
            "artifact": "Table4_subject_loso_component_ablation_ranked.csv",
            "purpose": "Strict subject-independent ablation ranking.",
            "recommended_manuscript_location": "Main ablation table.",
        },
        {
            "artifact": "Table5_modality_ablation.csv",
            "purpose": "Modality contribution analysis.",
            "recommended_manuscript_location": "Ablation subsection.",
        },
        {
            "artifact": "Table6_architecture_component_ablation.csv",
            "purpose": "Temporal/fusion/head/regularization component analysis.",
            "recommended_manuscript_location": "Ablation subsection.",
        },
        {
            "artifact": "Table7_component_importance_delta_vs_final_v3.csv",
            "purpose": "Quantifies performance drop/gain relative to final DynaMER-v3.",
            "recommended_manuscript_location": "Ablation interpretation.",
        },
        {
            "artifact": "Table8_reviewer_claim_defense_matrix.csv",
            "purpose": "Maps manuscript claims to evidence files.",
            "recommended_manuscript_location": "Internal defense guide / supplementary planning.",
        },
        {
            "artifact": "Fig1_dynamer_v3_validation_barplot_data.csv",
            "purpose": "Figure data for final validation grouped bar plot.",
            "recommended_manuscript_location": "Main Figure: validation results.",
        },
        {
            "artifact": "Fig2_seed_iv_model_comparison_grouped_bar_data.csv",
            "purpose": "Figure data for SEED-IV model comparison.",
            "recommended_manuscript_location": "Main Figure: baseline comparison.",
        },
        {
            "artifact": "Fig3_component_delta_heatmap_data.csv",
            "purpose": "Figure data for ablation delta heatmap.",
            "recommended_manuscript_location": "Main or supplementary ablation figure.",
        },
        {
            "artifact": "Fig4_modality_ablation_data.csv",
            "purpose": "Figure data for modality ablation.",
            "recommended_manuscript_location": "Ablation figure.",
        },
        {
            "artifact": "Fig5_session_vs_subject_robustness_scatter_data.csv",
            "purpose": "Figure data for session-vs-subject trade-off.",
            "recommended_manuscript_location": "Discussion/results trade-off figure.",
        },
        {
            "artifact": "Fig6_fold_distribution_boxplot_data.csv",
            "purpose": "Figure data for per-fold distribution/boxplots.",
            "recommended_manuscript_location": "Supplementary robustness figure.",
        },
        {
            "artifact": "Fig7_parameter_count_vs_performance_data.csv",
            "purpose": "Figure data for efficiency/performance trade-off.",
            "recommended_manuscript_location": "Efficiency/complexity analysis.",
        },
    ]

    evidence_index = pd.DataFrame(evidence_rows)
    evidence_index.to_csv(evidence_dir / "stage19C_evidence_index.csv", index=False)

    # ------------------------------------------------------------
    # Summary JSON + README
    # ------------------------------------------------------------
    summary = {
        "stage": "19C_paper_ready_tables_figdata",
        "overall_passed": True,
        "inputs_checked": {k: str(v) for k, v in paths.items()},
        "outputs": {
            "root": str(out_dir),
            "tables": str(tables_dir),
            "figure_data": str(figdata_dir),
            "evidence_index": str(evidence_dir),
            "locked": str(lock_dir),
        },
        "table_count": len(list(tables_dir.glob("*.csv"))),
        "figure_data_count": len(list(figdata_dir.glob("*.csv"))),
        "note": "This stage generates paper-ready tables and figure-ready data only. It does not generate final publication figures.",
    }

    (out_dir / "19C_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    readme = """Stage 19C: Paper-Ready Tables and Figure-Ready Data
===================================================

This stage compiles clean manuscript tables and figure-ready CSV files from Stage 19A and Stage 19B.

It does NOT generate final plots yet.

Recommended usage:
1. Inspect tables/*.csv for manuscript tables.
2. Inspect figure_data/*.csv to decide which figures should be generated.
3. Use the evidence index to map each output to a manuscript section.

Key outputs:
- tables/Table1_final_dynamer_v3_validation.csv
- tables/Table2_seed_iv_model_baseline_architecture_comparison.csv
- tables/Table4_subject_loso_component_ablation_ranked.csv
- figure_data/Fig1_dynamer_v3_validation_barplot_data.csv
- figure_data/Fig2_seed_iv_model_comparison_grouped_bar_data.csv
- figure_data/Fig3_component_delta_heatmap_data.csv
- figure_data/Fig5_session_vs_subject_robustness_scatter_data.csv
"""

    (out_dir / "README_STAGE19C.txt").write_text(readme, encoding="utf-8")

    # ------------------------------------------------------------
    # Lock package
    # ------------------------------------------------------------
    if lock_dir.exists():
        for child in lock_dir.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rmtree(child)

    shutil.copytree(out_dir, lock_dir, dirs_exist_ok=True)

    # ------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------
    print("\n=== Stage 19C completed ===")
    print(f"Tables written: {len(list(tables_dir.glob('*.csv')))}")
    print(f"Figure-data files written: {len(list(figdata_dir.glob('*.csv')))}")

    print("\n=== Paper-ready tables ===")
    for p in sorted(tables_dir.glob("*.csv")):
        print(p)

    print("\n=== Figure-ready data files ===")
    for p in sorted(figdata_dir.glob("*.csv")):
        print(p)

    print("\n=== Evidence index ===")
    print(evidence_index.to_string(index=False))

    print("\nSaved:")
    print(out_dir)

    print("\nLocked:")
    print(lock_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
