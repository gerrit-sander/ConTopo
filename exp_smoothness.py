import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import math

from utils.load import (
    parse_model_load_args,
    load_encoders_from_model_folder,
)
from utils.experiments import get_cifar10_eval_loader
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

    # Load all encoders from the provided model folder (one per trial),
    # selecting the checkpoint indicated by --prefer (defaults to 'last' so we
    # reuse the encoder paired with the trained readout head).
    encoders_meta = load_encoders_from_model_folder(
        model_folder=args.path,
        prefer=args.prefer,
        device=args.device,
        dp_if_multi_gpu=args.dp,
        eval_mode=True,
        strict=True,
    )

    # Eval-only CIFAR-10 loader
    val_loader = get_cifar10_eval_loader(
        root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Compute Moran's I average for each encoder, then aggregate across trials
    trial_scores = []
    for encoder, meta in encoders_meta:
        device = next(encoder.parameters()).device
        encoder.eval()

        total_I = 0.0
        n = 0

        with torch.no_grad():
            for batch in val_loader:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(device)

                embeddings = encoder(images)  # expected shape: [B, 256]
                grids = embeddings.view(embeddings.size(0), 16, 16)

                for g in grids:
                    I = morans_I_2d(g)
                    total_I += float(I.item())
                    n += 1

        avg_I = total_I / n if n > 0 else float('nan')
        trial_scores.append(avg_I)

    # Aggregate statistics across trials
    import math as _m
    N = len(trial_scores)
    if N == 0:
        print('nan')
        return
    mean = sum(trial_scores) / N
    if N > 1:
        var = sum((x - mean) ** 2 for x in trial_scores) / (N - 1)
        std = _m.sqrt(var)
    else:
        std = 0.0
    sem = std / _m.sqrt(N) if N > 0 else float('nan')

    # Print concise summary for plotting error bands
    # Lines make it simple to parse if needed
    print(f"n_trials: {N}")
    print(f"mean: {mean}")
    print(f"std: {std}")
    print(f"sem: {sem}")


if __name__ == "__main__":
    main()
