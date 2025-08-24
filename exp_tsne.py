import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from utils.load import load_encoder_from_path, parse_model_load_args
from utils.experiments import get_cifar10_eval_loader, resolve_figure_path
from utils.train import load_cifar10_metadata

def main():
    args = parse_model_load_args()

    # Load encoder
    encoder, meta = load_encoder_from_path(args.path, args.device, args.prefer, args.dp)
    src = meta.get("ckpt_path", args.path)
    print(f"Loaded stage: {meta.get('stage')} epoch: {meta.get('epoch')} from {src}")

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
    args = meta["args"]

    torch.manual_seed(42)
    np.random.seed(42)

    target_samples = 2000
    feats = []
    labs_all = []

    with torch.no_grad():
        for imgs, labs in val_loader:
            if len(labs_all) >= target_samples:
                break

            imgs = imgs.to(device, non_blocking=True)
            out = encoder(imgs)
            if out.ndim > 2:
                out = out.flatten(1)
            out = out.detach().cpu()

            remaining = target_samples - len(labs_all)
            if out.size(0) > remaining:
                out = out[:remaining]
                labs = labs[:remaining]

            feats.append(out)
            labs_all.extend(labs.tolist())

    if not feats:
        raise RuntimeError("No embeddings collected from the eval loader.")
    X = torch.cat(feats, dim=0).numpy()
    y = np.array(labs_all)
    print(f"[tSNE] Collected {X.shape[0]} embeddings with dim {X.shape[1]}.")

    tsne = TSNE(n_components=2, random_state=42)
    X2 = tsne.fit_transform(X)

    # Plot
    plt.figure(figsize=(10, 8))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'brown']
    for i in range(10):
        idx = (y == i)
        if idx.any():
            plt.scatter(
                X2[idx, 0], X2[idx, 1],
                c=colors[i], s=8, alpha=0.8,
                label=class_names[i] if i < len(class_names) else f"class {i}"
            )

    plt.title('t-SNE Visualization of Encoder Feature Space (CIFAR-10 eval, 2000 samples)')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(markerscale=2, fontsize=9, frameon=True)
    plt.grid(True, linewidth=0.3, alpha=0.5)

    plt.tight_layout()
    figurepath = resolve_figure_path(src, experiment="tsne")
    plt.savefig(figurepath, dpi=200, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()