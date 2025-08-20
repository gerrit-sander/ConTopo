import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

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