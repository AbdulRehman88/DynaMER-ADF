
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def to_num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def summarize(df, group_cols, metrics):
    rows = []
    df = df.copy()

    for m in metrics:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce")

    for keys, g in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)

        g = g.copy()
        row = dict(zip(group_cols, keys))
        row["folds"] = int(len(g))

        for m in metrics:
            if m in g.columns:
                vals = pd.to_numeric(g[m], errors="coerce").dropna()
                mean = float(vals.mean()) if len(vals) else float("nan")
                std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
                row[f"{m}_valid_n"] = int(len(vals))
                row[f"{m}_missing_n"] = int(len(g) - len(vals))
                row[f"{m}_mean"] = mean
                row[f"{m}_std"] = std
                row[f"{m}_min"] = float(vals.min()) if len(vals) else float("nan")
                row[f"{m}_max"] = float(vals.max()) if len(vals) else float("nan")
                row[f"{m}_mean_std"] = f"{mean:.3f}  {std:.3f}"

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--local-paths", required=True)
    ap.add_argument("--summary-config", required=True)
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    local = read_yaml(args.local_paths)
    scfg = read_yaml(args.summary_config)

    project_root = Path(local["PROJECT_ROOT"])
    out_dir = project_root / scfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    source_dir = project_root / scfg["source_dir"]

    metrics = [
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_roc_auc",
        "test_pr_auc",
    ]

    phase_files = scfg["phase_files"]

    all_best = []

    for phase_name, info in phase_files.items():
        path = source_dir / info["best_epoch_file"]
        if not path.exists():
            raise FileNotFoundError(f"Missing required file for {phase_name}: {path}")

        df = pd.read_csv(path)
        df["phase"] = phase_name
        df["claim_type"] = info["claim_type"]
        df["phase_label"] = info["label"]

        df = to_num(df, metrics)
        all_best.append(df)

    all_best = pd.concat(all_best, ignore_index=True)
    all_best.to_csv(out_dir / "13_all_best_epoch_metrics_with_claim_type.csv", index=False)

    primary = all_best[all_best["claim_type"] == "primary"].copy()
    diagnostic = all_best[all_best["claim_type"] == "diagnostic"].copy()

    primary_group_cols = ["dataset", "task", "protocol"]
    diagnostic_group_cols = ["dataset", "task", "protocol"]

    primary_summary = summarize(primary, primary_group_cols, metrics)
    diagnostic_summary = summarize(diagnostic, diagnostic_group_cols, metrics)

    primary_summary.to_csv(out_dir / "13_primary_results_mean_std.csv", index=False)
    diagnostic_summary.to_csv(out_dir / "13_diagnostic_results_mean_std.csv", index=False)

    compact_cols = [
        "dataset",
        "task",
        "protocol",
        "folds",
        "test_balanced_accuracy_mean_std",
        "test_balanced_accuracy_valid_n",
        "test_macro_f1_mean_std",
        "test_macro_f1_valid_n",
        "test_roc_auc_mean_std",
        "test_roc_auc_valid_n",
        "test_pr_auc_mean_std",
        "test_pr_auc_valid_n",
    ]

    primary_compact = primary_summary[[c for c in compact_cols if c in primary_summary.columns]].copy()
    diagnostic_compact = diagnostic_summary[[c for c in compact_cols if c in diagnostic_summary.columns]].copy()

    primary_compact.to_csv(out_dir / "13_primary_results_paper_compact.csv", index=False)
    diagnostic_compact.to_csv(out_dir / "13_diagnostic_results_paper_compact.csv", index=False)

    checks = []

    def add_check(name, observed, expected, passed):
        checks.append({
            "check": name,
            "observed": observed,
            "expected": expected,
            "passed": bool(passed),
        })

    add_check("phase files loaded", len(phase_files), len(phase_files), True)
    add_check("all best rows loaded", len(all_best), ">0", len(all_best) > 0)
    add_check("primary rows loaded", len(primary), ">0", len(primary) > 0)
    add_check("diagnostic rows loaded", len(diagnostic), ">0", len(diagnostic) > 0)
    add_check(
        "primary diagnostic separation clean",
        sorted(all_best["claim_type"].dropna().unique().tolist()),
        ["diagnostic", "primary"],
        sorted(all_best["claim_type"].dropna().unique().tolist()) == ["diagnostic", "primary"],
    )

    checks_df = pd.DataFrame(checks)
    checks_df.to_csv(out_dir / "13_result_summary_checks.csv", index=False)

    print("\n[Stage 13] Result summarization complete.\n")
    print("Checks:")
    print(checks_df.to_string(index=False))
    print("\nPrimary compact table:")
    print(primary_compact.to_string(index=False))
    print("\nDiagnostic compact table:")
    print(diagnostic_compact.to_string(index=False))


if __name__ == "__main__":
    main()
