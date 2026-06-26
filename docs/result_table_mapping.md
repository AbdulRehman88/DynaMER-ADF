# Result Table Mapping

This document maps manuscript evidence categories to the scripts that generated or summarized them.

| Evidence category | Main scripts |
|---|---|
| Dataset audit and manifest integrity | 01_dataset_audit.py, 02_prepare_dataset_manifests.py |
| Leakage-safe split construction | 03_prepare_leakage_safe_splits.py |
| Temporal feature views | 08_prepare_temporal_feature_views.py |
| Baseline comparisons | 14_train_baselines.py, 13_summarize_paper_grade_results.py |
| DynaMER-BiTCN / ADF / ADF-LS / Anchor | 15_train_dynamer_v2.py, 16_train_dynamer_v3.py, 17_train_dynamer_v4.py, 18_train_dynamer_v5.py |
| Component ablations | 19B_train_dynamer_v3_ablation.py |
| Subject-mixed protocol gap | 22A, 22B order-safe, 22C order-safe, 22D |
| ADF-family capacity audit | 22E, 22F, 22G |
| Nested LOSO robustness | 23A, 23B |
| Protocol-extension evidence package | 24A |
| Physiological separability and prediction audit | 25A |
| Locked manuscript evidence collection | 26A |

Generated result files are not committed by default. Re-run the scripts locally to regenerate them.
