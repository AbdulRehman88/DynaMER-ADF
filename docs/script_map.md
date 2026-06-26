# Script Map

This file maps the original numbered pipeline scripts to their public manuscript role.

## Core preparation pipeline

| Script | Public role | Decision |
|---|---|---|
| 00_check_environment.py | Environment sanity check | Keep |
| 01_dataset_audit.py | Dataset audit and inventory | Keep |
| 02_prepare_dataset_manifests.py | Dataset manifest construction | Keep |
| 03_prepare_leakage_safe_splits.py | Leakage-safe split generation | Keep |
| 04_prepare_model_ready_datasets.py | Model-ready dataset preparation | Keep |
| 05_verify_dataset_loaders.py | Dataset loader verification | Keep |
| 06_define_model_data_modules.py | Data-module smoke definition | Keep |
| 07_define_model_architecture.py | Architecture smoke definition | Keep |
| 08_prepare_temporal_feature_views.py | Temporal feature-view generation | Keep |
| 09_verify_temporal_data_modules.py | Temporal data-module verification | Keep |
| 10_training_loop_smoke_test.py | Training-loop smoke test | Keep |
| 11_full_experiment_registry.py | Experiment registry construction | Keep |
| 12_controlled_full_training.py | Controlled full training runner | Keep |
| 13_summarize_paper_grade_results.py | Primary/diagnostic result summarization | Keep |

## Main model training

| Script | Public role | Decision |
|---|---|---|
| 14_train_baselines.py | Baseline model training | Keep |
| 15_train_dynamer_v2.py | DynaMER-BiTCN training | Keep |
| 16_train_dynamer_v3.py | DynaMER-ADF training | Keep |
| 17_train_dynamer_v4.py | DynaMER-ADF-LS training | Keep |
| 18_train_dynamer_v5.py | DynaMER-Anchor training | Keep |
| 19B_train_dynamer_v3_ablation.py | Architecture/component ablation | Keep |
| 19C_compile_paper_tables_figdata.py | Paper table and figure-data compilation | Keep |

## Protocol extensions and audits

| Script | Public role | Decision |
|---|---|---|
| 21_final_manuscript_evidence_audit.py | Final evidence consistency audit | Keep |
| 22A_seed_iv_subject_mixed_5fold_splits.py | SEED-IV subject-mixed 5-fold split generation | Keep |
| 22B_train_seed_iv_subject_mixed_5fold_order_safe.py | Order-safe subject-mixed 5-fold training | Keep |
| 22C_summarize_seed_iv_subject_mixed_5fold_order_safe.py | Order-safe subject-mixed 5-fold summarization | Keep |
| 22D_audit_subject_mixed_5fold_consistency.py | Subject-mixed consistency audit | Keep |
| 22E_train_dynamer_adf_subject_mixed_capacity_audit.py | ADF-family capacity audit | Keep |
| 22F_summarize_dynamer_adf_subject_mixed_capacity_audit.py | ADF-family capacity summary | Keep |
| 22G_validation_selected_adf_family_subject_mixed.py | Selected ADF-family validation | Keep |
| 23A_train_nested_loso_dynamer_adf.py | Nested LOSO model-selection protocol | Keep |
| 23B_summarize_nested_loso_dynamer_adf.py | Nested LOSO summarization | Keep |
| 24A_generate_protocol_extension_evidence_package.py | Protocol-extension evidence package | Keep |
| 25A_compute_physiological_evidence_audit.py | Physiological separability and prediction audit | Keep |
| 26A_collect_manuscript_locked_results.py | Locked manuscript-result collection | Keep |

## Archived scripts

These scripts are retained for provenance but excluded from the active public workflow.

| Script | Reason archived |
|---|---|
| 22B_train_seed_iv_subject_mixed_5fold.py | Superseded by order-safe version |
| 22C_summarize_seed_iv_subject_mixed_5fold.py | Superseded by order-safe version |
| 20_generate_publication_figures.py | Figure-generation draft/provenance script |
| 20_rebuild_named_paper_assets.py | Figure/table asset rebuild draft |
| 20D_polish_problem_figures.py | Figure polishing/provenance script |
| 20Z_generate_all_individual_paper_figures.py | Figure-generation draft/provenance script |
| 20Z_generate_locked_seediv_model_figures.py | Figure-generation draft/provenance script |
| fix_critical_issue_01_locked_model_names.py | One-off correction script; provenance only |
