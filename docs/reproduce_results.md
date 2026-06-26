# Reproducing Results

This repository provides the code pipeline used to reproduce the manuscript analyses for DynaMER-ADF.

## Step 1: Configure local paths

```powershell
Copy-Item configs\local_paths.example.yaml configs\local_paths.yaml
notepad configs\local_paths.yaml
```

## Step 2: Run dataset audit and manifests

```powershell
python scripts\01_dataset_audit.py --config configs\01_dataset_audit.yaml --local-paths configs\local_paths.yaml
python scripts\02_prepare_dataset_manifests.py --config configs\02_prepare_dataset_manifests.yaml --local-paths configs\local_paths.yaml
```

## Step 3: Build leakage-safe splits

```powershell
python scripts\03_prepare_leakage_safe_splits.py --config configs\03_prepare_leakage_safe_splits.yaml --local-paths configs\local_paths.yaml
```

## Step 4: Prepare temporal features

```powershell
python scripts\08_prepare_temporal_feature_views.py --config configs\08_prepare_temporal_feature_views.yaml --local-paths configs\local_paths.yaml
```

## Step 5: Train main models

```powershell
python scripts\14_train_baselines.py --config configs\14_train_baselines.yaml --local-paths configs\local_paths.yaml
python scripts\15_train_dynamer_bitcn.py --config configs\15_train_dynamer_bitcn.yaml --local-paths configs\local_paths.yaml
python scripts\16_train_dynamer_adf.py --config configs\16_train_dynamer_adf.yaml --local-paths configs\local_paths.yaml
python scripts\17_train_dynamer_adf_ls.py --config configs\17_train_dynamer_adf_ls.yaml --local-paths configs\local_paths.yaml
python scripts\18_train_dynamer_anchor.py --config configs\18_train_dynamer_anchor.yaml --local-paths configs\local_paths.yaml
```

## Step 6: Run ablations and protocol audits

```powershell
python scripts\19B_train_dynamer_adf_ablation.py --config configs\19B_train_dynamer_adf_ablation.yaml --local-paths configs\local_paths.yaml
python scripts\22B_train_seed_iv_subject_mixed_5fold_order_safe.py --config configs\config.yaml --local-paths configs\local_paths.yaml
python scripts\23A_train_nested_loso_dynamer_adf.py --config configs\config.yaml --local-paths configs\local_paths.yaml
python scripts\25A_compute_physiological_evidence_audit.py --config configs\config.yaml --local-paths configs\local_paths.yaml --overwrite
```

See `docs/script_map.md` for the role of each script.
