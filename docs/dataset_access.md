# Dataset Access

Raw datasets are not included in this repository.

Users must obtain each dataset from its official source and configure local paths in:

```powershell
configs\local_paths.yaml
```

Create the local path file from the provided template:

```powershell
Copy-Item configs\local_paths.example.yaml configs\local_paths.yaml
notepad configs\local_paths.yaml
```

Expected dataset roots:

| Dataset | Modalities used | Protocols |
|---|---|---|
| SEED-IV | EEG, eye features | subject-LOSO, cross-session, subject-mixed 5-fold, nested LOSO |
| DREAMER | EEG, ECG | subject-LOSO |
| AMIGOS | EEG, ECG, GSR/EDA | subject-LOSO |

Do not commit raw datasets, generated feature arrays, checkpoints, or local path files.
