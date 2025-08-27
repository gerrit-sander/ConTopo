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
    Resolve an output path for a figure based on a checkpoint or model directory.

    Layout:
      - Find nearest ancestor named 'models'.
      - Save under sibling 'figures' of its parent (e.g., .../ShallowCNN/figures).
      - Filename is <model_name>.png.
      - If `experiment` is set, nest under figures/<experiment>/.

    Returns the absolute path as string.
    """
    p = Path(src_path).resolve()

    # If a file is given (e.g., ckpt), treat its parent as the model directory
    model_path = p.parent if p.suffix else p

    # Walk up to find ".../models"
    models_dir = None
    for parent in [model_path] + list(model_path.parents):
        if parent.name == "models":
            models_dir = parent
            break

    if models_dir is None:
        # No 'models' ancestor → place figures next to given path
        figures_root = model_path / "figures"
        model_name = model_path.name
    else:
        # Use architecture folder (.. / ShallowCNN or ResNet18) as figures root
        arch_dir = models_dir.parent  # e.g., .../ShallowCNN or .../ResNet18
        figures_root = arch_dir / "figures"
        try:
            # model_name = top-level folder under 'models'
            rel = model_path.relative_to(models_dir)
            model_name = rel.parts[0]
        except ValueError:
            # Fallback: use directory name as model name
            model_name = model_path.name

    # Optional experiment subfolder (sanitize to safe filename)
    if experiment and experiment.strip():
        safe_exp = re.sub(r"[^\w.\-]+", "_", experiment.strip())
        figures_dir = figures_root / safe_exp
    else:
        figures_dir = figures_root

    # Ensure output directory exists
    figures_dir.mkdir(parents=True, exist_ok=True)
    return str(figures_dir / f"{model_name}.png")