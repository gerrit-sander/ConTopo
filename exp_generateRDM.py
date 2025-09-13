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

    Returns images stacked in CLASS-GROUPED order ([1000, C, H, W]) and labels where the
    first 100 belong to class 0, next 100 to class 1, ..., last 100 to class 9.
    """
    imgs_by_class = {i: [] for i in range(10)}

    with torch.no_grad():
        for imgs, labs in val_loader:
            for img, lab in zip(imgs, labs):
                c = int(lab)
                lst = imgs_by_class[c]
                if len(lst) < per_class:
                    lst.append(img)
            # Early exit if all classes are filled
            if all(len(lst) >= per_class for lst in imgs_by_class.values()):
                break

    # Verify and stack in class order 0..9
    for c in range(10):
        if len(imgs_by_class[c]) < per_class:
            raise RuntimeError(f"Could not collect required samples for class {c}: "
                               f"got {len(imgs_by_class[c])}, need {per_class}")

    ordered_imgs = []
    ordered_labels = []
    for c in range(10):
        ordered_imgs.extend(imgs_by_class[c][:per_class])
        ordered_labels.extend([c] * per_class)

    stacked = torch.stack(ordered_imgs, dim=0)
    return stacked, ordered_labels


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


def _upper_triangle_vector(M: torch.Tensor, include_diagonal: bool = True) -> torch.Tensor:
    """Return the upper-triangular values of square matrix M as a 1D tensor."""
    N = M.size(0)
    offset = 0 if include_diagonal else 1
    idx = torch.triu_indices(N, N, offset=offset)
    return M[idx[0], idx[1]].to(dtype=torch.float32, device="cpu")


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

    # Compute averaged RDM across trials (full matrix for plotting), then compress for saving
    avg_rdm = torch.stack(rdms, dim=0).mean(dim=0) if len(rdms) > 0 else None

    # Save per-trial figures (full RDMs for visualization)
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

    # Save averaged RDM figure if available
    if avg_rdm is not None:
        avg_fig_name = f"AvgRDM_{base}.png"
        avg_fig_path = os.path.join(model_folder, avg_fig_name)
        plt.figure(figsize=(8, 8))
        im = plt.imshow(avg_rdm.numpy(), cmap="viridis", interpolation="nearest")
        plt.title("Average RDM across trials")
        plt.xlabel("Samples (N=1000)")
        plt.ylabel("Samples (N=1000)")
        plt.colorbar(im, fraction=0.046, pad=0.04, label="1 - Pearson r")
        plt.tight_layout()
        plt.savefig(avg_fig_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Saved Avg RDM figure: {avg_fig_path}
")

    # Compress RDMs to upper-triangular vectors for saving (saves space)
    rdms_upper = [
        _upper_triangle_vector(rdm, include_diagonal=True) for rdm in rdms
    ]
    avg_rdm_upper = _upper_triangle_vector(avg_rdm, include_diagonal=True) if avg_rdm is not None else None

    # Second-level analysis: correlate RDMs across trials (unique pairs, exclude self)
    # Use upper-triangle vectors WITHOUT the diagonal for correlation to avoid trivial zeros.
    if len(rdms) >= 2:
        vecs_no_diag = [
            _upper_triangle_vector(rdm, include_diagonal=False) for rdm in rdms
        ]
        corrs = []
        eps = 1e-8
        for i in range(len(vecs_no_diag)):
            xi = vecs_no_diag[i]
            xi_c = xi - xi.mean()
            xi_n = xi_c.norm().clamp_min(eps)
            for j in range(i + 1, len(vecs_no_diag)):
                xj = vecs_no_diag[j]
                xj_c = xj - xj.mean()
                xj_n = xj_c.norm().clamp_min(eps)
                r = float((xi_c @ xj_c) / (xi_n * xj_n))
                corrs.append(r)
        if corrs:
            import math as _m
            mean_corr = sum(corrs) / len(corrs)
            if len(corrs) > 1:
                var = sum((c - mean_corr) ** 2 for c in corrs) / (len(corrs) - 1)
                std_corr = _m.sqrt(var)
            else:
                std_corr = 0.0
            print(f"RDM consistency across trials — mean: {mean_corr:.6f}, std: {std_corr:.6f}")
        else:
            print("RDM consistency across trials — insufficient pairs to compute correlation.")
    else:
        print("RDM consistency across trials — need at least 2 trials.")

    # Save all trial RDMs (upper triangle only)
    payload = {
        "rdms_upper": rdms_upper,   # list of length n; each is 1D tensor (upper triangle incl. diag)
        "N": 1000,                  # original matrix size
        "include_diagonal": True,
        "labels": labels,           # list[int] length 1000
        "metas": metas,             # minimal metadata per trial
        "model_folder": model_folder,
        "prefer": args.prefer,
    }
    torch.save(payload, out_path)
    print(f"Saved {len(rdms_upper)} upper-triangular RDMs to: {out_path}")

    # Save averaged RDM (upper triangle only) in a separate file
    if avg_rdm_upper is not None:
        avg_name = f"AvgRDM_{base}.pt"
        avg_path = os.path.join(model_folder, avg_name)
        avg_payload = {
            "avg_rdm_upper": avg_rdm_upper,
            "N": 1000,
            "include_diagonal": True,
            "labels": labels,
            "num_trials": len(rdms_upper),
            "model_folder": model_folder,
            "prefer": args.prefer,
        }
        torch.save(avg_payload, avg_path)
        print(f"Saved averaged upper-triangular RDM to: {avg_path}")

    # (Figures saved above.)


if __name__ == "__main__":
    main()
