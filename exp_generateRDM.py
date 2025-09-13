import os
import torch
import matplotlib.pyplot as plt

from utils.load import (
    parse_model_load_args,
    load_encoders_from_model_folder,
)
from utils.experiments import get_cifar10_eval_loader


def _select_deterministic_cifar10_subset(val_loader, per_class: int = 100):
    """
    Deterministically collect exactly `per_class` samples for each of the 10 CIFAR-10 classes
    from the evaluation loader (which is ordered and not shuffled).

    Returns a single stacked tensor of images [1000, C, H, W] and a list of labels (length 1000).
    """
    class_quota = {i: per_class for i in range(10)}
    imgs_out, labs_out = [], []

    with torch.no_grad():
        for imgs, labs in val_loader:
            # Iterate samples in order; take until class quotas are met
            for img, lab in zip(imgs, labs):
                c = int(lab)
                if class_quota[c] > 0:
                    imgs_out.append(img)
                    labs_out.append(c)
                    class_quota[c] -= 1
                    # Early exit if all quotas reached
                    if all(v == 0 for v in class_quota.values()):
                        stacked = torch.stack(imgs_out, dim=0)
                        return stacked, labs_out

    # If we exit loop without meeting quotas, raise
    missing = {k: v for k, v in class_quota.items() if v > 0}
    raise RuntimeError(f"Could not collect required samples per class; missing: {missing}")


def _compute_embeddings(encoder: torch.nn.Module, images: torch.Tensor, device: torch.device, batch_size: int) -> torch.Tensor:
    """
    Run images through encoder in batches and return a [N, D] tensor of embeddings (CPU float32).
    """
    encoder.eval()
    feats = []
    N = images.size(0)
    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = images[i:i+batch_size].to(device, non_blocking=True)
            out = encoder(batch)
            if out.ndim > 2:
                out = out.flatten(1)
            feats.append(out.detach().cpu().to(dtype=torch.float32))
    return torch.cat(feats, dim=0)


def _pearson_rdm(X: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute 1 - Pearson correlation matrix for row-vectors in X (shape [N, D]).
    Returns a [N, N] tensor on CPU.
    """
    X = X.to(dtype=torch.float32, device="cpu")
    Xc = X - X.mean(dim=1, keepdim=True)
    norms = Xc.norm(dim=1, keepdim=True).clamp_min(eps)
    Y = Xc / norms
    corr = Y @ Y.t()
    rdm = 1.0 - corr
    # Ensure perfect self-similarity maps to 0 exactly
    rdm.fill_diagonal_(0.0)
    return rdm


def main():
    args = parse_model_load_args()

    # Load all encoders from the provided model folder (one per trial),
    # selecting the best validation checkpoint per run when available.
    encoders_meta = load_encoders_from_model_folder(
        model_folder=args.path,
        prefer=args.prefer,
        device=args.device,
        dp_if_multi_gpu=args.dp,
        eval_mode=True,
        strict=True,
    )

    if len(encoders_meta) == 0:
        raise RuntimeError("No encoders found in the provided model folder.")

    # Eval-only CIFAR-10 loader (deterministic order)
    val_loader = get_cifar10_eval_loader(
        root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Deterministically select 100 samples per class (1000 total)
    samples_cpu, labels = _select_deterministic_cifar10_subset(val_loader, per_class=100)

    # For each encoder, compute embeddings on the fixed 1000 samples and then the RDM
    rdms = []
    metas = []
    for encoder, meta in encoders_meta:
        device = next(encoder.parameters()).device
        feats = _compute_embeddings(encoder, samples_cpu, device, args.batch_size)
        rdm = _pearson_rdm(feats)
        rdms.append(rdm)
        # Keep compact meta info for traceability
        metas.append({
            "epoch": meta.get("epoch"),
            "stage": meta.get("stage"),
            "ckpt_path": meta.get("ckpt_path"),
        })

    # Save all trial RDMs in a single file under the given model folder
    model_folder = args.path if os.path.isdir(args.path) else os.path.dirname(args.path)
    base = os.path.basename(os.path.normpath(model_folder))
    out_name = f"RDM_{base}.pt"
    out_path = os.path.join(model_folder, out_name)

    payload = {
        "rdms": rdms,               # list of [1000, 1000] float32 tensors (CPU)
        "labels": labels,           # list[int] length 1000, class labels of rows/cols
        "metas": metas,             # minimal metadata per trial
        "model_folder": model_folder,
        "prefer": args.prefer,
    }
    torch.save(payload, out_path)
    print(f"Saved {len(rdms)} RDMs to: {out_path}")

    # Also save a corresponding figure for each RDM in the same folder
    # Name pattern: RDM_<base>__<runname>.png (fallback to index if unavailable)
    for idx, (rdm, meta) in enumerate(zip(rdms, metas)):
        try:
            run_folder = os.path.dirname(meta.get("ckpt_path", ""))
            run_name = os.path.basename(run_folder) if run_folder else f"trial_{idx:02d}"
        except Exception:
            run_name = f"trial_{idx:02d}"

        fig_name = f"RDM_{base}__{run_name}.png"
        fig_path = os.path.join(model_folder, fig_name)

        plt.figure(figsize=(8, 8))
        im = plt.imshow(rdm.numpy(), cmap="viridis", interpolation="nearest")
        plt.title(f"RDM: {run_name}")
        plt.xlabel("Samples (N=1000)")
        plt.ylabel("Samples (N=1000)")
        plt.colorbar(im, fraction=0.046, pad=0.04, label="1 - Pearson r")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved RDM figure: {fig_path}")


if __name__ == "__main__":
    main()
