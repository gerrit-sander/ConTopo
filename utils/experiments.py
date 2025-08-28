import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from pathlib import Path
import re

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def get_cifar10_eval_loader(
    root: str = "./dataset",
    batch_size: int = 256,
    num_workers: int = 2,
    pin_memory: bool | None = None,
    drop_last: bool = False,
    subset: int | None = None,
):
    """
    Build a DataLoader for CIFAR-10 *test* split (eval only).

    - Normalizes with CIFAR-10 mean/std.
    - `pin_memory` defaults to CUDA availability.
    - `subset` lets you evaluate on the first N examples (quick checks).
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    ds = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    # Optional quick subset for faster eval/plots
    if subset is not None and subset < len(ds):
        ds = Subset(ds, range(subset))

    # Deterministic order for evaluation
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return loader

def resolve_figure_path(src_path: str, experiment: str | None = None) -> str:
    """
    Resolve an output path for a figure based on a checkpoint or model/run directory.

    Layout:
      <Arch>/figures/<experiment?>/<model_name>/<run_name>/<ckpt_basename>.png

    Where:
      - <Arch> is the parent directory of 'models' (e.g., 'ShallowCNN' or 'ResNet18').
      - <model_name> is the first folder under 'models' (your hyperparam folder).
      - <run_name> is the next folder under 'models' (e.g., 'trial_00').
      - <ckpt_basename> is the checkpoint filename without extension (e.g., 'e2e_epoch0200').

    Fallback when no 'models' ancestor exists:
      <run_root>/figures/<experiment?>/<model_dir>/<run_dir>/<ckpt_basename>.png

    Returns the absolute path as string, creating parent directories if needed.
    """
    p = Path(src_path).resolve()

    # If a file is given (ckpt), use its parent as the run directory and its stem as the figure name.
    is_file = bool(p.suffix)
    run_path = p.parent if is_file else p
    ckpt_stem = p.stem if is_file else "model"

    # Find nearest ancestor named 'models'
    models_dir = None
    for parent in [run_path] + list(run_path.parents):
        if parent.name == "models":
            models_dir = parent
            break

    if models_dir is not None:
        # figures root sits next to 'models', i.e. under the architecture folder
        arch_dir = models_dir.parent  # e.g., .../ShallowCNN or .../ResNet18
        figures_root = arch_dir / "figures"
        try:
            # Expect run_path like: <...>/models/<model_name>/<run_name>
            rel = run_path.relative_to(models_dir)
            parts = rel.parts
            model_name = parts[0] if len(parts) >= 1 else run_path.name
            run_name   = parts[1] if len(parts) >= 2 else run_path.name
        except ValueError:
            # If relative_to fails, fall back to directory names
            model_name = run_path.parent.name
            run_name   = run_path.name
    else:
        # No 'models' ancestor → keep figures next to the provided path
        figures_root = run_path / "figures"
        model_name = run_path.parent.name  # hyperparam folder
        run_name   = run_path.name         # run folder

    # Optional experiment subfolder (sanitize to safe filename)
    if experiment and experiment.strip():
        safe_exp = re.sub(r"[^\w.\-]+", "_", experiment.strip())
        out_dir = figures_root / safe_exp / model_name / run_name
    else:
        out_dir = figures_root / model_name / run_name

    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"{ckpt_stem}.png")