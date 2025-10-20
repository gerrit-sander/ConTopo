import math
from collections import defaultdict

import torch

from utils.load import (
    parse_model_load_args,
    load_model_bundles,
)
from utils.experiments import get_cifar10_eval_loader


THRESHOLDS = (0.1, 0.3, 0.5, 0.6, 0.7, 0.8)


def infer_grid_shape(dim: int) -> tuple[int, int]:
    """Return a compact 2D grid shape that preserves the number of units."""
    if dim <= 0:
        raise ValueError("Embedding dimension must be positive.")
    h = int(math.sqrt(dim))
    while h > 1 and dim % h != 0:
        h -= 1
    return h, dim // h


def compute_trial_distances(embeddings: torch.Tensor) -> dict[float, float]:
    """Compute mean pairwise distances for each correlation threshold."""
    if embeddings.dim() != 2:
        raise ValueError("Expected embeddings shaped as [num_samples, num_units].")

    num_samples, num_units = embeddings.shape
    if num_units < 2 or num_samples < 2:
        return {thr: float("nan") for thr in THRESHOLDS}

    device = embeddings.device
    # Center activations and normalise to compute Pearson correlation.
    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=True)
    # Avoid division by zero for inactive units.
    std = std.masked_fill(std == 0, float("nan"))
    normed = centered / std
    # Replace NaNs from zero-variance units with zeros so they do not contribute.
    normed = torch.nan_to_num(normed, nan=0.0)

    denom = float(num_samples - 1) if num_samples > 1 else 1.0
    corr = (normed.T @ normed) / denom
    corr = corr.clamp(-1.0, 1.0)

    # Prepare 2D grid coordinates for distance computation.
    h, w = infer_grid_shape(num_units)
    rows = torch.arange(h, dtype=torch.float32, device=device)
    cols = torch.arange(w, dtype=torch.float32, device=device)
    grid_y, grid_x = torch.meshgrid(rows, cols)
    coords = torch.stack([grid_y.reshape(-1), grid_x.reshape(-1)], dim=1)

    idx_i, idx_j = torch.triu_indices(num_units, num_units, offset=1, device=device)
    pair_corr = corr[idx_i, idx_j]
    pair_dist = torch.linalg.norm(coords[idx_i] - coords[idx_j], dim=1)

    results = {}
    for thr in THRESHOLDS:
        mask = pair_corr >= thr
        if mask.any():
            mean_dist = pair_dist[mask].mean().item()
        else:
            mean_dist = float("nan")
        results[thr] = mean_dist
    return results


def main():
    args = parse_model_load_args()

    bundles = load_model_bundles(
        path=args.path,
        prefer=args.prefer,
        device=args.device,
        dp_if_multi_gpu=args.dp,
        eval_mode=True,
        strict=True,
    )
    if not bundles:
        raise RuntimeError("No checkpoints found for the provided path.")

    val_loader = get_cifar10_eval_loader(
        root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    trial_threshold_distances = defaultdict(list)

    for bundle in bundles:
        encoder = bundle.encoder
        device = next(encoder.parameters()).device
        encoder.eval()

        activations = []
        with torch.no_grad():
            for batch in val_loader:
                images = batch[0] if isinstance(batch, (list, tuple)) else batch
                images = images.to(device, non_blocking=True)
                embeds = encoder(images)
                activations.append(embeds.detach().cpu())

        if not activations:
            continue

        all_embeddings = torch.cat(activations, dim=0)
        trial_results = compute_trial_distances(all_embeddings)
        for thr, dist in trial_results.items():
            trial_threshold_distances[thr].append(dist)

    if not trial_threshold_distances:
        print("No embeddings collected; nothing to report.")
        return

    for thr in THRESHOLDS:
        values = trial_threshold_distances.get(thr, [])
        finite_vals = [v for v in values if not math.isnan(v)]
        n = len(finite_vals)
        if n == 0:
            mean = std = sem = float("nan")
        else:
            mean = sum(finite_vals) / n
            if n > 1:
                var = sum((v - mean) ** 2 for v in finite_vals) / (n - 1)
                std = math.sqrt(var)
            else:
                std = 0.0
            sem = std / math.sqrt(n) if n > 0 else float("nan")
        print(f"threshold: {thr}")
        print(f"  n_trials: {n}")
        print(f"  mean_distance: {mean}")
        print(f"  std: {std}")
        print(f"  sem: {sem}")


if __name__ == "__main__":
    main()
