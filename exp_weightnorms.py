import torch
import numpy as np
from utils.load import (
    parse_model_load_args,
    load_encoders_from_model_folder,
)
from utils.train import unwrap
def main():
    args = parse_model_load_args()
    # Load all encoders from the provided model folder (one per trial),
    # selecting the best/last validation checkpoint per run when available.
    encoders_meta = load_encoders_from_model_folder(
        model_folder=args.path,
        prefer=args.prefer,
        device=args.device,
        dp_if_multi_gpu=args.dp,
        eval_mode=True,
        strict=True,
    )
    # Collect L2 norms of the rows of the final FC weights (one per embedding dim)
    # across all loaded encoders.
    all_row_norms = []  # list[float]
    for encoder, meta in encoders_meta:
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
        norms = torch.linalg.norm(W, ord=2, dim=1).numpy()
        all_row_norms.extend(norms.tolist())
    # Aggregate across all trials' embedding rows
    if len(all_row_norms) == 0:
        print("nan")
        return
    M = len(all_row_norms)
    mean = float(np.mean(all_row_norms))
    std = float(np.std(all_row_norms, ddof=1 if M > 1 else 0))
    # Minimal, parse-friendly output
    print(f"n_models: {len(encoders_meta)}")
    print(f"vectors_total: {M}")
    print(f"mean: {mean}")
    print(f"std: {std}")
if __name__ == "__main__":
    main()