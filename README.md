# ConTopo

## Overview
ConTopo contains training and analysis scripts for CIFAR-10 experiments with topographic regularization. The repository is organized for running multiple training trials and post-hoc analyses used in the paper.

This README focuses on reproducibility and practical usage.

## Repository Layout
- `main_ce.py`: Cross-entropy training with topographic loss.
- `main_coscontr.py`: Cosine-contrastive training + linear readout.
- `main_supcon.py`: SupCon/SimCLR training + linear readout.
- `run_all.py`: Launches a full grid from `configs/experiments.json`.
- `run_all_experiments.py`: Runs one analysis script over all model folders in a root.
- `exp_*.py`: Analysis/visualization scripts.
- `losses/`: Task and topographic loss implementations.
- `networks/`: Model architectures.
- `utils/`: Data loading, checkpoint loading, and experiment helpers.
- `configs/cifar10.yaml`: CIFAR-10 class + animacy metadata.
- `configs/experiments.json`: Grid definition for full experiment sweeps.

## Environment Setup
Use Python 3.9+ (3.10+ recommended).

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision numpy matplotlib scikit-learn pyyaml tensorboard
```

Notes:
- CUDA is optional but recommended for training.
- If you install a CUDA-enabled PyTorch build, follow the official PyTorch install selector for your system.

## Data Setup
All scripts default to `./dataset` and automatically download CIFAR-10 when needed.

- Training scripts write/read from: `./dataset`
- Analysis scripts accept `--dataset-root` (default: `./dataset`)

## Quick Smoke Runs
Use small epoch counts to verify the pipeline and output structure.

```bash
# Cross-entropy
python3 main_ce.py ws resnet18 --epochs 2 --batch_size 128 --trial 0

# Cosine-contrastive + readout
python3 main_coscontr.py ws resnet18 --epochs 2 --readout_epochs 2 --batch_size 128 --readout_batch_size 256 --trial 0

# SupCon + readout
python3 main_supcon.py ws resnet18 --task_method supcon --epochs 2 --readout_epochs 2 --batch_size 128 --readout_batch_size 256 --trial 0
```

## Training Commands
### Single runs
```bash
# CE
python3 main_ce.py ws resnet18 --trial 0 --topographic_loss_rho 0.05

# Cosine-contrastive
python3 main_coscontr.py ws resnet18 --trial 0 --topographic_loss_rho 0.05

# SupCon
python3 main_supcon.py ws resnet18 --task_method supcon --trial 0 --topographic_loss_rho 0.05

# SimCLR
python3 main_supcon.py ws resnet18 --task_method simclr --trial 0 --topographic_loss_rho 0.05
```

### Full paper-scale grid
```bash
python3 run_all.py
```

`run_all.py` reads `configs/experiments.json` and launches the full command grid.

Important caveats:
- `run_all.py` does not define a CLI parser; it starts runs immediately.
- In `configs/experiments.json`, the default Python executable is currently `python`. If your environment only has `python3`, update:

```json
"defaults": {
  "python": "python3"
}
```

## Analysis / Experiment Scripts
Most analysis scripts load one run folder, one model folder with multiple trials, or a checkpoint path via shared arguments from `utils/load.py`:

- Positional `path`
- Optional `--prefer {best,last}`
- Optional `--device`, `--dataset-root`, `--batch-size`, `--num-workers`

### Core analyses
```bash
# Generate per-trial and averaged RDM artifacts
python3 exp_generateRDM.py <model_folder_or_run_folder> --prefer best

# Model-by-model RSA over generated RDMs
python3 exp_RSA.py <models_root> --trials-per-model 5

# Error-correlation and ensemble evaluation across noise conditions
python3 exp_errorcorr.py <model_folder_or_run_folder> --prefer best

# Moran's I smoothness summary
python3 exp_smoothness.py <model_folder_or_run_folder> --prefer best

# Unit-distance summary at correlation thresholds
python3 exp_unitdist.py <model_folder_or_run_folder> --prefer best

# t-SNE plot of embeddings
python3 exp_tsne.py <model_folder_or_run_folder> --prefer best

# Activation maps for one exemplar per CIFAR-10 class
python3 exp_actmaps.py <model_folder_or_run_folder> --prefer best

# L2 norms of final FC rows
python3 exp_weightnorms.py <model_folder_or_run_folder> --prefer best
```

### Run one analysis over all model folders
```bash
python3 run_all_experiments.py exp_generateRDM.py ./save/ResNet18/models
```

Pass extra args to each invocation after `--`:

```bash
python3 run_all_experiments.py exp_smoothness.py ./save/ResNet18/models -- --prefer best --batch-size 512
```

## Output Structure
### Training outputs
- Checkpoints:
  - `save/<Arch>/models/<model_name>/trial_XX/`
  - CE: `e2e_best.pth`, `e2e_last.pth`, periodic `e2e_epochXXXX.pth`
  - Contrastive: `contrastive_best.pth`, `contrastive_last.pth`, periodic `contrastive_epochXXXX.pth`
  - Readout: `readout_best.pth`, `readout_last.pth`, periodic `readout_epochXXXX.pth`
- TensorBoard logs:
  - `save/<Arch>/tensorboard/<task_or_method>/<model_name>/trial_XX/`

### Analysis outputs
- `exp_generateRDM.py` writes `RDM_*.pt`, `AvgRDM_*.pt`, `RDMConsistency_*.pt`, and related figures in the target model folder.
- `exp_RSA.py` writes RSA matrices/CSVs/figures under the provided `models_root`.
- `exp_tsne.py` and `exp_actmaps.py` write figures via `utils/experiments.resolve_figure_path` (typically under a `figures/` tree near `models/`).

## Repro Tips
- Use fixed trial ids (`--trial 0..4`) for consistent run naming (`trial_00`, `trial_01`, ...).
- For analyses, use `--prefer best` for validation-best checkpoints or `--prefer last` for final snapshots.
- Expected model-folder layout for multi-trial analyses:
  - `<model_folder>/trial_00`, `<model_folder>/trial_01`, ...
  - each run folder containing the expected checkpoint files.

