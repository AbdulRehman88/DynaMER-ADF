
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".").resolve()

# Inputs
STAGE19C_REVISED = ROOT / "outputs" / "ablations" / "19C_paper_ready_tables_figdata_revised"
FIG_REVISED = ROOT / "outputs" / "figures" / "20_publication_quality_figures_revised"
FIG_POLISHED = ROOT / "outputs" / "figures" / "20_publication_quality_figures_polished"

# Outputs
OUT = ROOT / "outputs" / "qa" / "21_final_manuscript_evidence_audit"
MANUSCRIPT_ASSETS = ROOT / "outputs" / "manuscript_ready_assets"
FINAL_TABLES = MANUSCRIPT_ASSETS / "tables"
FINAL_FIGURES = MANUSCRIPT_ASSETS / "figures"
LOCK = ROOT / "outputs" / "final_locked_results" / "stage21_final_manuscript_evidence_audit"

for d in [OUT, MANUSCRIPT_ASSETS, FINAL_TABLES, FINAL_FIGURES, LOCK]:
    d.mkdir(parents=True, exist_ok=True)


NAME_MAP = {
    # Main architecture variants
    "DynaMER-v1": "DynaMER-Base",
    "DynaMER-v2b": "DynaMER-BiTCN-R",
    "DynaMER-v2": "DynaMER-BiTCN",
    "DynaMER-v3": "DynaMER-ADF",
    "DynaMER-v4": "DynaMER-ADF-LS",
    "DynaMER-v5": "DynaMER-Anchor",

    "dynamer_v1": "DynaMER-Base",
    "dynamer_v2b": "DynaMER-BiTCN-R",
    "dynamer_v2": "DynaMER-BiTCN",
    "dynamer_v3": "DynaMER-ADF",
    "dynamer_v4": "DynaMER-ADF-LS",
    "dynamer_v5": "DynaMER-Anchor",

    # Important internal table-column terms
    "final_v3": "final_ADF",
    "vs_final_v3": "vs_DynaMER_ADF",
    "Final DynaMER-v3": "DynaMER-ADF",
    "final DynaMER-v3": "DynaMER-ADF",

    # Component ablations
    "v3_eeg_combined_only": "EEG-only",
    "v3_eye_only": "Eye-only",
    "v3_de_only": "EEG-DE only",
    "v3_psd_only": "EEG-PSD only",
    "v3_de_eye": "EEG-DE + Eye",
    "v3_psd_eye": "EEG-PSD + Eye",
    "v3_path_v1_only": "Base path only",
    "v3_path_v2_only": "BiTCN path only",
    "v3_path_fixed_mean": "Fixed dual-path mean",
    "v3_fusion_mean": "Mean fusion",
    "v3_no_spike": "No spike head",
    "v3_low_spike": "Low spike head",
    "v3_high_spike": "High spike head",
    "v3_moddrop_0p10": "Modality dropout",
    "v3_tcn_depth_1": "TCN depth 1",
    "v3_tcn_depth_3": "TCN depth 3",
    "v3_dropout_0p10": "Dropout 0.10",
    "v3_dropout_0p30": "Dropout 0.30",

    # Baselines
    "temporal_mlp": "Temporal MLP",
    "cnn_lstm": "CNN-LSTM",
    "bilstm": "BiLSTM",
    "lstm": "LSTM",
    "gru": "GRU",
    "tcn": "TCN",
}

OLD_NAME_PATTERNS = [
    r"DynaMER-v[1-5]\b",
    r"dynamer_v[1-5]\b",
    r"\bv3_[A-Za-z0-9_]+\b",
    r"\bfinal_v3\b",
    r"\bvs_final_v3\b",
    r"Final DynaMER-v3",
    r"final DynaMER-v3",
]


EXPECTED_FINAL_METRICS = [
    {
        "source_hint": "Table1",
        "dataset": "SEED-IV",
        "task": "4-Class Emotion",
        "protocol": "Subject-LOSO",
        "metric": "BA",
        "expected": 0.600,
        "tolerance": 0.002,
    },
    {
        "source_hint": "Table1",
        "dataset": "SEED-IV",
        "task": "4-Class Emotion",
        "protocol": "Cross-Session",
        "metric": "BA",
        "expected": 0.549,
        "tolerance": 0.002,
    },
    {
        "source_hint": "Table1",
        "dataset": "SEED-IV",
        "task": "4-Class Emotion",
        "protocol": "Subject-LOSO",
        "metric": "MacroF1",
        "expected": 0.584,
        "tolerance": 0.002,
    },
    {
        "source_hint": "Table1",
        "dataset": "AMIGOS",
        "task": "Arousal",
        "protocol": "Subject-LOSO",
        "metric": "BA",
        "expected": 0.549,
        "tolerance": 0.002,
    },
    {
        "source_hint": "Table1",
        "dataset": "AMIGOS",
        "task": "Valence",
        "protocol": "Subject-LOSO",
        "metric": "BA",
        "expected": 0.553,
        "tolerance": 0.002,
    },
]


def replace_locked_names_text(text: str) -> str:
    s = str(text)

    # Aggressive paper-facing sanitization:
    # use direct substring replacement because old names often appear inside
    # CSV column names, SVG metadata, or file-derived labels where word-boundary
    # regex cannot catch underscores.
    for old in sorted(NAME_MAP.keys(), key=len, reverse=True):
        s = s.replace(old, NAME_MAP[old])

    # Extra safety for any remaining known internal patterns.
    s = s.replace("DynaMER-v1", "DynaMER-Base")
    s = s.replace("DynaMER-v2b", "DynaMER-BiTCN-R")
    s = s.replace("DynaMER-v2", "DynaMER-BiTCN")
    s = s.replace("DynaMER-v3", "DynaMER-ADF")
    s = s.replace("DynaMER-v4", "DynaMER-ADF-LS")
    s = s.replace("DynaMER-v5", "DynaMER-Anchor")

    s = s.replace("dynamer_v1", "DynaMER-Base")
    s = s.replace("dynamer_v2b", "DynaMER-BiTCN-R")
    s = s.replace("dynamer_v2", "DynaMER-BiTCN")
    s = s.replace("dynamer_v3", "DynaMER-ADF")
    s = s.replace("dynamer_v4", "DynaMER-ADF-LS")
    s = s.replace("dynamer_v5", "DynaMER-Anchor")

    s = s.replace("final_v3", "final_ADF")
    s = s.replace("vs_final_v3", "vs_DynaMER_ADF")
    s = s.replace("Final DynaMER-v3", "DynaMER-ADF")
    s = s.replace("final DynaMER-v3", "DynaMER-ADF")

    return s


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [replace_locked_names_text(c) for c in out.columns]
    for c in out.columns:
        if pd.api.types.is_object_dtype(out[c]) or str(out[c].dtype).startswith("string"):
            out[c] = out[c].astype(str).map(replace_locked_names_text)
    return out


def parse_mean(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(x))
    return float(nums[0]) if nums else np.nan


def copy_clean_tables() -> pd.DataFrame:
    src_tables = STAGE19C_REVISED / "tables"
    if not src_tables.exists():
        raise FileNotFoundError(f"Missing revised tables folder: {src_tables}")

    # Clear old final tables
    if FINAL_TABLES.exists():
        shutil.rmtree(FINAL_TABLES)
    FINAL_TABLES.mkdir(parents=True, exist_ok=True)

    rows = []
    for src in sorted(src_tables.glob("*.csv")):
        df = pd.read_csv(src)
        df = sanitize_dataframe(df)
        dst = FINAL_TABLES / src.name
        df.to_csv(dst, index=False)
        rows.append({
            "table": src.name,
            "source": str(src),
            "final_path": str(dst),
            "rows": len(df),
            "columns": len(df.columns),
            "exists": dst.exists(),
            "size_bytes": dst.stat().st_size if dst.exists() else 0,
        })

    return pd.DataFrame(rows)


def copy_final_figures() -> pd.DataFrame:
    if not FIG_REVISED.exists():
        raise FileNotFoundError(f"Missing revised figures folder: {FIG_REVISED}")
    if not FIG_POLISHED.exists():
        raise FileNotFoundError(f"Missing polished figures folder: {FIG_POLISHED}")

    if FINAL_FIGURES.exists():
        shutil.rmtree(FINAL_FIGURES)
    FINAL_FIGURES.mkdir(parents=True, exist_ok=True)

    selections = []

    # Keep accepted revised individual bar figures only.
    # Exclude problem figure families replaced by polished figures.
    exclude_prefixes = (
        "Fig04_",
        "Fig05_",
        "Fig06_",
        "Fig07_",
        "Fig3_",
        "Fig4_",
        "Fig5_",
        "Fig6_",
        "Fig7_",
    )

    for src in sorted(FIG_REVISED.glob("*")):
        if src.suffix.lower() not in [".pdf", ".svg", ".png"]:
            continue
        if src.name.startswith(exclude_prefixes):
            continue
        selections.append((src, "revised_clean_bar_or_metric_figure"))

    # Add polished final figures
    polished_stems = [
        "Fig04_session_vs_subject_tradeoff_polished",
        "Fig05a_modality_delta_heatmap_polished",
        "Fig05b_architecture_delta_heatmap_polished",
        "Fig06_fold_stability_boxplot_polished",
        "Fig07_efficiency_vs_performance_polished",
    ]

    for stem in polished_stems:
        for ext in [".pdf", ".svg", ".png"]:
            src = FIG_POLISHED / f"{stem}{ext}"
            if src.exists():
                selections.append((src, "polished_final_problem_figure"))

    rows = []
    for src, role in selections:
        dst = FINAL_FIGURES / replace_locked_names_text(src.name)
        shutil.copy2(src, dst)

        # Sanitize text-based vector figure content, especially SVG text/metadata.
        if dst.suffix.lower() in [".svg", ".txt", ".json", ".csv", ".md"]:
            txt = dst.read_text(encoding="utf-8", errors="ignore")
            txt = replace_locked_names_text(txt)
            dst.write_text(txt, encoding="utf-8")

        rows.append({
            "figure_file": dst.name,
            "source": replace_locked_names_text(str(src)),
            "final_path": str(dst),
            "role": role,
            "extension": dst.suffix.lower(),
            "exists": dst.exists(),
            "size_bytes": dst.stat().st_size if dst.exists() else 0,
        })

    inv = pd.DataFrame(rows)
    return inv.sort_values(["figure_file", "extension"]).reset_index(drop=True)


def scan_old_names(paths) -> pd.DataFrame:
    hits = []

    for path in paths:
        if not path.exists() or not path.is_file():
            continue

        if path.suffix.lower() not in [".csv", ".txt", ".md", ".json", ".svg", ".tex"]:
            continue

        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            hits.append({
                "file": str(path),
                "pattern": "READ_ERROR",
                "match": str(e),
                "line": -1,
                "context": "",
            })
            continue

        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            for pat in OLD_NAME_PATTERNS:
                for m in re.finditer(pat, line):
                    hits.append({
                        "file": str(path),
                        "pattern": pat,
                        "match": m.group(0),
                        "line": i,
                        "context": line[max(0, m.start() - 80):m.end() + 80],
                    })

    return pd.DataFrame(hits)


def figure_triplet_check(fig_inv: pd.DataFrame) -> pd.DataFrame:
    if fig_inv.empty:
        return pd.DataFrame(columns=["stem", "has_pdf", "has_svg", "has_png", "complete_triplet", "min_size_bytes"])

    fig_inv = fig_inv.copy()
    fig_inv["stem"] = fig_inv["figure_file"].map(lambda x: Path(str(x)).stem)

    rows = []
    for stem, g in fig_inv.groupby("stem"):
        exts = set(g["extension"].astype(str))
        sizes = pd.to_numeric(g["size_bytes"], errors="coerce").dropna()
        rows.append({
            "stem": stem,
            "has_pdf": ".pdf" in exts,
            "has_svg": ".svg" in exts,
            "has_png": ".png" in exts,
            "complete_triplet": all(e in exts for e in [".pdf", ".svg", ".png"]),
            "min_size_bytes": int(sizes.min()) if len(sizes) else 0,
            "files": len(g),
        })
    return pd.DataFrame(rows).sort_values("stem")


def metric_consistency_check() -> pd.DataFrame:
    table1_path = FINAL_TABLES / "Table1_final_dynamer_v3_validation.csv"
    if not table1_path.exists():
        return pd.DataFrame([{
            "check": "Table1 exists",
            "passed": False,
            "details": f"Missing {table1_path}",
        }])

    df = pd.read_csv(table1_path)
    rows = []

    # Normalize likely columns
    lower_cols = {c.lower(): c for c in df.columns}
    dataset_col = lower_cols.get("dataset")
    task_col = lower_cols.get("task")
    protocol_col = lower_cols.get("protocol")

    if dataset_col is None or task_col is None or protocol_col is None:
        return pd.DataFrame([{
            "check": "Table1 required columns",
            "passed": False,
            "details": f"Columns found: {list(df.columns)}",
        }])

    metric_candidates = {
        "BA": ["BA", "test_balanced_accuracy_mean_std", "Balanced Accuracy"],
        "MacroF1": ["MacroF1", "Macro-F1", "test_macro_f1_mean_std"],
        "ROCAUC": ["ROCAUC", "ROC-AUC", "test_roc_auc_mean_std"],
        "PRAUC": ["PRAUC", "PR-AUC", "test_pr_auc_mean_std"],
    }

    for item in EXPECTED_FINAL_METRICS:
        metric_col = None
        for cand in metric_candidates[item["metric"]]:
            for c in df.columns:
                if c == cand:
                    metric_col = c
                    break
            if metric_col:
                break

        if metric_col is None:
            rows.append({
                "dataset": item["dataset"],
                "task": item["task"],
                "protocol": item["protocol"],
                "metric": item["metric"],
                "observed": np.nan,
                "expected": item["expected"],
                "tolerance": item["tolerance"],
                "passed": False,
                "details": f"Metric column not found for {item['metric']}",
            })
            continue

        mask = (
            df[dataset_col].astype(str).eq(item["dataset"])
            & df[task_col].astype(str).eq(item["task"])
            & df[protocol_col].astype(str).eq(item["protocol"])
        )

        if mask.sum() != 1:
            rows.append({
                "dataset": item["dataset"],
                "task": item["task"],
                "protocol": item["protocol"],
                "metric": item["metric"],
                "observed": np.nan,
                "expected": item["expected"],
                "tolerance": item["tolerance"],
                "passed": False,
                "details": f"Expected 1 matching row, found {int(mask.sum())}",
            })
            continue

        observed = parse_mean(df.loc[mask, metric_col].iloc[0])
        passed = bool(abs(observed - item["expected"]) <= item["tolerance"])
        rows.append({
            "dataset": item["dataset"],
            "task": item["task"],
            "protocol": item["protocol"],
            "metric": item["metric"],
            "observed": observed,
            "expected": item["expected"],
            "tolerance": item["tolerance"],
            "passed": passed,
            "details": "",
        })

    return pd.DataFrame(rows)


def primary_secondary_diagnostic_check() -> pd.DataFrame:
    rows = []

    table1 = FINAL_TABLES / "Table1_final_dynamer_v3_validation.csv"
    table2 = FINAL_TABLES / "Table2_seed_iv_model_baseline_architecture_comparison.csv"
    table4 = FINAL_TABLES / "Table4_subject_loso_component_ablation_ranked.csv"

    for p in [table1, table2, table4]:
        if not p.exists():
            rows.append({
                "check": f"{p.name} exists",
                "observed": "missing",
                "expected": "exists",
                "passed": False,
            })
            continue

        txt = p.read_text(encoding="utf-8", errors="ignore")
        has_subject_mixed = "Subject-Mixed" in txt or "subject_mixed" in txt
        rows.append({
            "check": f"{p.name} does not mix diagnostic subject-mixed results into main primary table",
            "observed": "contains subject-mixed" if has_subject_mixed else "no subject-mixed",
            "expected": "no subject-mixed",
            "passed": not has_subject_mixed,
        })

    return pd.DataFrame(rows)


def locked_folder_check() -> pd.DataFrame:
    required = [
        ROOT / "outputs" / "final_locked_results" / "stage19B_dynamer_v3_component_ablation",
        ROOT / "outputs" / "final_locked_results" / "stage19C_paper_ready_tables_figdata",
        ROOT / "outputs" / "final_locked_results" / "stage20_publication_quality_figures_polished",
    ]

    rows = []
    for p in required:
        rows.append({
            "locked_folder": str(p),
            "exists": p.exists(),
            "file_count": len([x for x in p.rglob("*") if x.is_file()]) if p.exists() else 0,
            "passed": p.exists() and len([x for x in p.rglob("*") if x.is_file()]) > 0,
        })
    return pd.DataFrame(rows)


def write_readme(overall_passed: bool):
    readme = f"""# Stage 21 Final Manuscript Evidence Audit

Overall passed: **{overall_passed}**

This audit creates the manuscript-ready asset package and verifies:

1. Paper-facing tables use locked scientific names.
2. Paper-facing figures use locked scientific names.
3. Old internal names are absent from final CSV/SVG/JSON/TXT/MD paper-facing assets.
4. Final figures are available as PDF, SVG, and PNG.
5. Key DynaMER-ADF metrics are consistent with locked results.
6. Primary/secondary results are not mixed with diagnostic subject-mixed results.
7. Locked evidence folders exist.

## Manuscript-ready assets

- Tables: `{FINAL_TABLES}`
- Figures: `{FINAL_FIGURES}`

## Audit output

- `{OUT / "21_final_evidence_audit_checks.csv"}`
- `{OUT / "21_old_name_hits.csv"}`
- `{OUT / "21_final_table_inventory.csv"}`
- `{OUT / "21_final_figure_inventory.csv"}`
- `{OUT / "21_metric_consistency_report.csv"}`
- `{OUT / "21_figure_triplet_check.csv"}`
"""
    (OUT / "README_STAGE21_FINAL_QA.md").write_text(readme, encoding="utf-8")


def main():
    print("[Stage 21] Copying and sanitizing final paper-facing tables...")
    table_inv = copy_clean_tables()

    print("[Stage 21] Assembling final manuscript-ready figures...")
    fig_inv = copy_final_figures()

    print("[Stage 21] Checking PDF/SVG/PNG figure triplets...")
    triplet = figure_triplet_check(fig_inv)

    print("[Stage 21] Scanning final paper-facing assets for forbidden old names...")
    scan_paths = []
    scan_paths += list(FINAL_TABLES.rglob("*"))
    scan_paths += list(FINAL_FIGURES.rglob("*"))
    scan_paths += [OUT / "21_final_table_inventory.csv", OUT / "21_final_figure_inventory.csv"]
    old_hits = scan_old_names(scan_paths)

    print("[Stage 21] Checking metric consistency...")
    metric_report = metric_consistency_check()

    print("[Stage 21] Checking primary/diagnostic separation...")
    protocol_report = primary_secondary_diagnostic_check()

    print("[Stage 21] Checking locked evidence folders...")
    locked_report = locked_folder_check()

    # Save reports
    table_inv.to_csv(OUT / "21_final_table_inventory.csv", index=False)
    fig_inv.to_csv(OUT / "21_final_figure_inventory.csv", index=False)
    triplet.to_csv(OUT / "21_figure_triplet_check.csv", index=False)
    old_hits.to_csv(OUT / "21_old_name_hits.csv", index=False)
    metric_report.to_csv(OUT / "21_metric_consistency_report.csv", index=False)
    protocol_report.to_csv(OUT / "21_primary_diagnostic_separation_report.csv", index=False)
    locked_report.to_csv(OUT / "21_locked_folder_report.csv", index=False)

    checks = []

    checks.append({
        "check": "final tables generated",
        "observed": int(len(table_inv)),
        "expected": ">0",
        "passed": len(table_inv) > 0 and bool(table_inv["exists"].all()),
    })

    checks.append({
        "check": "final figures generated",
        "observed": int(len(fig_inv)),
        "expected": ">0",
        "passed": len(fig_inv) > 0 and bool(fig_inv["exists"].all()),
    })

    checks.append({
        "check": "all final figure stems have PDF/SVG/PNG triplets",
        "observed": int(triplet["complete_triplet"].sum()) if len(triplet) else 0,
        "expected": int(len(triplet)),
        "passed": len(triplet) > 0 and bool(triplet["complete_triplet"].all()),
    })

    checks.append({
        "check": "final paper-facing assets contain no old internal names",
        "observed": int(len(old_hits)),
        "expected": 0,
        "passed": len(old_hits) == 0,
    })

    checks.append({
        "check": "key locked DynaMER-ADF metrics are consistent",
        "observed": int(metric_report["passed"].sum()) if "passed" in metric_report.columns else 0,
        "expected": int(len(metric_report)),
        "passed": "passed" in metric_report.columns and bool(metric_report["passed"].all()),
    })

    checks.append({
        "check": "main primary tables do not contain diagnostic subject-mixed results",
        "observed": int(protocol_report["passed"].sum()) if "passed" in protocol_report.columns else 0,
        "expected": int(len(protocol_report)),
        "passed": "passed" in protocol_report.columns and bool(protocol_report["passed"].all()),
    })

    checks.append({
        "check": "locked evidence folders exist",
        "observed": int(locked_report["passed"].sum()) if "passed" in locked_report.columns else 0,
        "expected": int(len(locked_report)),
        "passed": "passed" in locked_report.columns and bool(locked_report["passed"].all()),
    })

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(OUT / "21_final_evidence_audit_checks.csv", index=False)

    overall_passed = bool(checks_df["passed"].all())

    summary = {
        "stage": "21_final_manuscript_evidence_audit",
        "overall_passed": overall_passed,
        "final_tables": str(FINAL_TABLES),
        "final_figures": str(FINAL_FIGURES),
        "audit_output": str(OUT),
        "locked_output": str(LOCK),
        "tables_count": int(len(table_inv)),
        "figure_files_count": int(len(fig_inv)),
        "figure_stems_count": int(len(triplet)),
        "old_name_hits": int(len(old_hits)),
    }

    (OUT / "21_final_evidence_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_readme(overall_passed)

    # Lock audit + manuscript-ready assets
    if LOCK.exists():
        shutil.rmtree(LOCK)
    LOCK.mkdir(parents=True, exist_ok=True)

    shutil.copytree(OUT, LOCK / "audit_reports", dirs_exist_ok=True)
    shutil.copytree(MANUSCRIPT_ASSETS, LOCK / "manuscript_ready_assets", dirs_exist_ok=True)

    print("\n=== Stage 21 final manuscript evidence audit complete ===")
    print(f"Overall passed: {overall_passed}")
    print("\nChecks:")
    print(checks_df.to_string(index=False))

    if len(old_hits):
        print("\nWARNING: Old-name hits found. See:")
        print(OUT / "21_old_name_hits.csv")

    print("\nManuscript-ready tables:")
    print(FINAL_TABLES)

    print("\nManuscript-ready figures:")
    print(FINAL_FIGURES)

    print("\nAudit reports:")
    print(OUT)

    print("\nLocked package:")
    print(LOCK)


if __name__ == "__main__":
    main()
