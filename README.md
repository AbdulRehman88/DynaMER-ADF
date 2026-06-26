# DynaMER-ADF

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20917825.svg)](https://doi.org/10.5281/zenodo.20917825)

Leakage-safe modality-adaptive temporal fusion for multimodal physiological emotion recognition across SEED-IV, DREAMER, and AMIGOS.

This repository contains the code used for dataset auditing, leakage-safe split construction, temporal feature preparation, baseline training, DynaMER-ADF training, controlled ablations, protocol-extension analysis, and manuscript result-table generation.

## Installation

```powershell
git clone https://github.com/AbdulRehman88/DynaMER-ADF.git
cd DynaMER_Release
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

## Dataset paths

Raw datasets are not included. Copy the example path file and edit it locally:

```powershell
Copy-Item configs\local_paths.example.yaml configs\local_paths.yaml
notepad configs\local_paths.yaml
```

Expected public datasets:

- SEED-IV
- DREAMER
- AMIGOS

## Important leakage-control rule

All preprocessing, normalization, model selection, and metric computation must be performed inside the corresponding training/test split. Subject-independent protocols must not mix samples from the same subject across train and test partitions.

## Main workflow

```powershell
python scripts\01_dataset_audit.py --config configs\01_dataset_audit.yaml --local-paths configs\local_paths.yaml
python scripts\02_prepare_dataset_manifests.py --config configs\02_prepare_dataset_manifests.yaml --local-paths configs\local_paths.yaml
python scripts\03_prepare_leakage_safe_splits.py --config configs\03_prepare_leakage_safe_splits.yaml --local-paths configs\local_paths.yaml
python scripts\08_prepare_temporal_feature_views.py --config configs\08_prepare_temporal_feature_views.yaml --local-paths configs\local_paths.yaml
```

Training and analysis scripts are provided in `scripts/`. Public names will be finalized after script cleanup.

## Outputs

Generated outputs, checkpoints, caches, and raw data are intentionally excluded from version control.

