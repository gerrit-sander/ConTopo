import torch
import numpy as np

from utils.load import (
    parse_model_load_args,
    load_model_bundles,
)
from utils.train import unwrap

def main():
    args = parse_model_load_args()
    # Load all encoders from the provided model folder (one per trial),
    # selecting the checkpoint combination indicated by --prefer (defaults to 'best').
    bundles = load_model_bundles(
        path=args.path,
        prefer=args.prefer,
        device=args.device,
        dp_if_multi_gpu=args.dp,
        eval_mode=True,
        strict=True,
    )
    # For each trial, compute the mean L2 norm across the rows of the final FC
    # weight matrix (one row per embedding dimension), then aggregate across trials.
    trial_means = []
    vectors_total = 0
    for bundle in bundles:
        encoder = bundle.encoder
        base = unwrap(encoder)
        if not hasattr(base, "fc"):
            raise AttributeError(
                "Loaded encoder does not expose a final 'fc' layer."
            )
        fc = base.fc
        if not isinstance(fc, torch.nn.Linear):
            raise TypeError(
                "Expected the final 'fc' layer to be nn.Linear."
            )
        # fc.weight shape: [out_features (embedding dim), in_features]
        W = fc.weight.detach().to(dtype=torch.float32, device="cpu")
        # Row-wise L2 norms → one value per embedding dimension
        norms = torch.linalg.norm(W, ord=2, dim=1)
        trial_means.append(float(norms.mean().item()))
        vectors_total += int(norms.numel())
    # Aggregate across trials: mean and std of per-trial mean norms
    if len(trial_means) == 0:
        print("nan")
        return
    N = len(trial_means)
    mean = float(np.mean(trial_means))
    std = float(np.std(trial_means, ddof=1 if N > 1 else 0))
    # Minimal, parse-friendly output
    print(f"n_trials: {len(bundles)}")
    print(f"vectors_total: {vectors_total}")
    print(f"mean: {mean}")
    print(f"std: {std}")
if __name__ == "__main__":
    main()
