import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import math

from utils.load import load_encoder_from_path, parse_model_load_args
from utils.experiments import get_cifar10_eval_loader, resolve_figure_path
from utils.train import load_cifar10_metadata

# Compute Moran's I for a 2D grid using rook adjacency
def morans_I_2d(grid: torch.Tensor) -> torch.Tensor:
    x = grid.detach().to(dtype=torch.float32, device="cpu")
    N = float(x.numel())
    mean = x.mean()
    diff = x - mean
    denom = (diff * diff).sum()
    if denom.item() == 0:
        return torch.tensor(0.0)

    # Directed neighbor pairs (up, down, left, right)
    num = (
        (diff[1:, :] * diff[:-1, :]).sum() +
        (diff[:-1, :] * diff[1:, :]).sum() +
        (diff[:, 1:] * diff[:, :-1]).sum() +
        (diff[:, :-1] * diff[:, 1:]).sum()
    )

    # Total weight W: count of directed neighbor pairs
    h, w = x.shape
    W = float(2 * ((h - 1) * w + h * (w - 1)))

    return (N / W) * (num / denom)

def main():
    args = parse_model_load_args()

    # Load encoder
    encoder, meta = load_encoder_from_path(args.path, args.device, args.prefer, args.dp)
    src = meta.get("ckpt_path", args.path)

    # Eval-only CIFAR-10 loader
    val_loader = get_cifar10_eval_loader(
        root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    device = next(encoder.parameters()).device
    encoder.eval()

    config = load_cifar10_metadata()
    class_names = config["CIFAR10_CLASSES"]
    opt = meta["args"]

    total_I = 0.0
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            # Support loaders that return (images, labels, ...) or just images
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device)

            embeddings = encoder(images)  # expected shape: [B, 256]
            grids = embeddings.view(embeddings.size(0), 16, 16)

            for g in grids:
                I = morans_I_2d(g)
                total_I += float(I.item())
                n += 1

    avg_I = total_I / n if n > 0 else float('nan')
    print(avg_I)


if __name__ == "__main__":
    main()