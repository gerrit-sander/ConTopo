import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from pathlib import Path

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

def get_cifar10_eval_loader(
    root: str = "./dataset",
    batch_size: int = 256,
    num_workers: int = 8,
    pin_memory: bool | None = None,
    drop_last: bool = False,
    subset: int | None = None,
):
    """
    Returns a DataLoader over the CIFAR-10 test split (evaluation only).
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    ds = datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

    if subset is not None and subset < len(ds):
        ds = Subset(ds, range(subset))

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
    return loader

def resolve_figure_path(src_path: str) -> str:
    """
    Given a checkpoint path or a model directory path, locate the nearest 'models' ancestor.
    Then place the figure under the sibling 'figures' folder of that ancestor's parent
    (i.e., .../ShallowCNN/figures or .../ResNet18/figures) with filename <model_name>.png.
    """
    p = Path(src_path).resolve()

    # If src points to a file (e.g., .../ckpt_xxx.pt), use its parent (the model directory).
    model_path = p.parent if p.suffix else p


    models_dir = None
    for parent in [model_path] + list(model_path.parents):
        if parent.name == "models":
            models_dir = parent
            break

    if models_dir is None:

        figures_dir = model_path / "figures"
        model_name = model_path.name
    else:
        arch_dir = models_dir.parent  # e.g., .../ShallowCNN or .../ResNet18
        figures_dir = arch_dir / "figures"

        try:
            rel = model_path.relative_to(models_dir)
            model_name = rel.parts[0]
        except ValueError:
            model_name = model_path.name

    figures_dir.mkdir(parents=True, exist_ok=True)
    return str(figures_dir / f"{model_name}.png")