from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing YAML file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def as_path(x: str) -> Path:
    return Path(str(x).replace("\\", "/")).expanduser().resolve()


def mean_std(vals) -> str:
    arr = pd.to_numeric(pd.Series(vals), errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return "NA"
    if len(arr) == 1:
        return f"{arr[0]:.3f}"
    return f"{np.mean(arr):.3f} ± {np.std(arr, ddof=1):.3f}"


def best_epoch_rows(epoch_path: Path) -> pd.DataFrame:
    if not epoch_path.exists():
        raise FileNotFoundError(f"Missing epoch file: {epoch_path}")
    epochs = pd.read_csv(epoch_path)
    required = ["model_variant", "fold_index", "epoch", "monitor_value", "test_balanced_accuracy", "test_macro_f1", "test_roc_auc", "test_pr_auc"]
    missing = [c for c in required if c not in epochs.columns]
    if missing:
        raise RuntimeError(f"{epoch_path} missing columns: {missing}")
    rows = []
    for (variant, fold), group in epochs.groupby(["model_variant", "fold_index"], dropna=False):
        group = group.copy()
        group["_monitor"] = pd.to_numeric(group["monitor_value"], errors="coerce")
        if group["_monitor"].notna().any():
            idx = group.sort_values(["_monitor", "epoch"], ascending=[False, True]).index[0]
        else:
            idx = group.sort_values("epoch").index[-1]
        rows.append(group.loc[idx].drop(labels=["_monitor"], errors="ignore"))
    return pd.DataFrame(rows).sort_values(["model_variant", "fold_index"]).reset_index(drop=True)


def summarize(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, g in best.groupby("model_variant"):
        rows.append({
            "model_variant": variant,
            "folds": int(g["fold_index"].nunique()),
            "BA": mean_std(g["test_balanced_accuracy"]),
            "Macro-F1": mean_std(g["test_macro_f1"]),
            "ROC-AUC": mean_std(g["test_roc_auc"]),
            "PR-AUC": mean_std(g["test_pr_auc"]),
            "BA_mean": float(pd.to_numeric(g["test_balanced_accuracy"], errors="coerce").mean()),
            "MacroF1_mean": float(pd.to_numeric(g["test_macro_f1"], errors="coerce").mean()),
            "mean_best_epoch": float(pd.to_numeric(g["epoch"], errors="coerce").mean()),
        })
    return pd.DataFrame(rows).sort_values(["BA_mean", "MacroF1_mean"], ascending=[False, False]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Stage 22 subject-mixed 5-fold training consistency and order sensitivity.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--local-paths", default="configs/local_paths.yaml")
    args = parser.parse_args()

    _cfg = load_yaml(Path(args.config))
    local_paths = load_yaml(Path(args.local_paths))
    project_root = as_path(local_paths["PROJECT_ROOT"])
    base = project_root / "outputs" / "protocol_extension" / "22_seed_iv_subject_mixed_5fold"
    out_dir = base / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for name in ["training", "training_order_safe"]:
        tdir = base / name
        ep = tdir / "22B_subject_mixed_5fold_all_epoch_metrics.csv"
        rr = tdir / "22B_subject_mixed_5fold_training_run_report.csv"
        if ep.exists():
            candidates.append((name, tdir, ep, rr))

    if not candidates:
        raise FileNotFoundError(f"No Stage 22B training epoch files found under {base}")

    all_summary = []
    all_best = []
    for name, tdir, ep, rr in candidates:
        best = best_epoch_rows(ep)
        best.insert(0, "training_dir_name", name)
        all_best.append(best)
        summ = summarize(best)
        summ.insert(0, "training_dir_name", name)
        all_summary.append(summ)
        print(f"\n[{name}]")
        print(summ.to_string(index=False))

        if rr.exists():
            report = pd.read_csv(rr)
            print(f"Run report: {len(report)} rows; status counts = {report['status'].value_counts(dropna=False).to_dict()}")
            if "run_seed" in report.columns:
                missing = int(report["run_seed"].isna().sum())
                print(f"Per-run seed column found; missing seeds = {missing}")
            else:
                print("WARNING: no run_seed column. This training directory may be order-dependent.")

    best_all = pd.concat(all_best, ignore_index=True)
    summary_all = pd.concat(all_summary, ignore_index=True)
    best_path = out_dir / "22D_best_epoch_rows_by_training_dir.csv"
    summary_path = out_dir / "22D_model_summary_by_training_dir.csv"
    best_all.to_csv(best_path, index=False)
    summary_all.to_csv(summary_path, index=False)

    # Explicit ADF comparison if multiple training dirs exist.
    adf = summary_all[summary_all["model_variant"] == "dynamer_v3"].copy()
    if len(adf) > 0:
        adf_path = out_dir / "22D_dynamer_adf_summary_by_training_dir.csv"
        adf.to_csv(adf_path, index=False)
        print("\nDynaMER-ADF by training directory:")
        print(adf.to_string(index=False))

    # If both original and order-safe exist, compare fold-level ADF.
    if set(best_all["training_dir_name"]) >= {"training", "training_order_safe"}:
        left = best_all[(best_all["training_dir_name"] == "training") & (best_all["model_variant"] == "dynamer_v3")]
        right = best_all[(best_all["training_dir_name"] == "training_order_safe") & (best_all["model_variant"] == "dynamer_v3")]
        cmp = left[["fold_index", "epoch", "monitor_value", "test_balanced_accuracy", "test_macro_f1"]].merge(
            right[["fold_index", "epoch", "monitor_value", "test_balanced_accuracy", "test_macro_f1"]],
            on="fold_index", suffixes=("_original", "_order_safe"), how="outer"
        )
        cmp["delta_BA_order_safe_minus_original"] = cmp["test_balanced_accuracy_order_safe"] - cmp["test_balanced_accuracy_original"]
        cmp["delta_MacroF1_order_safe_minus_original"] = cmp["test_macro_f1_order_safe"] - cmp["test_macro_f1_original"]
        cmp_path = out_dir / "22D_dynamer_adf_original_vs_order_safe_fold_comparison.csv"
        cmp.to_csv(cmp_path, index=False)
        print("\nDynaMER-ADF original vs order-safe fold comparison:")
        print(cmp.to_string(index=False))

    summary_json = {
        "outputs": {
            "best_epoch_rows_by_training_dir": str(best_path),
            "model_summary_by_training_dir": str(summary_path),
        },
        "interpretation": "If the same model/fold changes when trained alone versus with other models, the original Stage 22B was order-dependent. Use training_order_safe results for manuscript evidence."
    }
    (out_dir / "22D_audit_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    print(f"\nSaved audit outputs to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
