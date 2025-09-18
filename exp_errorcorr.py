"""Evaluate pairwise error correlations across model trials.

This script runs a classification experiment for every model folder contained in
the path supplied on the command line.  It leverages the unified loading helper
`load_model_bundles` so both cross-entropy and contrastive+readout checkpoints
are supported transparently.
"""

from __future__ import annotations

from pathlib import Path

import torch

from utils.load import parse_model_load_args, load_model_bundles
from utils.experiments import get_cifar10_eval_loader


def _collect_errors_and_preds(
    encoder: torch.nn.Module,
    classifier: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total = len(loader.dataset)
    errors = torch.zeros(total, dtype=torch.float32)
    preds = torch.empty(total, dtype=torch.long)
    targets = torch.empty(total, dtype=torch.long)
    offset = 0

    encoder.eval()
    classifier.eval()

    with torch.no_grad():
        for batch in loader:
            images, labels = batch[:2] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            feats = encoder(images)
            if isinstance(feats, (tuple, list)):
                feats = feats[0]
            logits = classifier(feats)
            if isinstance(logits, (tuple, list)):
                logits = logits[-1]

            batch_preds = logits.argmax(dim=1)
            batch_errors = (batch_preds != labels).float().cpu()

            size = batch_errors.numel()
            errors[offset : offset + size] = batch_errors
            preds[offset : offset + size] = batch_preds.cpu()
            targets[offset : offset + size] = labels.cpu()
            offset += size

    return errors, preds, targets


def _pearson_corrcoef(error_matrix: torch.Tensor) -> torch.Tensor:
    if error_matrix.size(0) == 0:
        return torch.empty(0, 0)
    centered = error_matrix - error_matrix.mean(dim=1, keepdim=True)
    cov = centered @ centered.T
    var = centered.pow(2).sum(dim=1)
    denom = torch.sqrt(var).unsqueeze(0) * torch.sqrt(var).unsqueeze(1)
    corr = cov.clone()
    mask = denom == 0
    corr[~mask] = corr[~mask] / denom[~mask]
    corr.masked_fill_(mask, float("nan"))
    idx = torch.arange(corr.size(0))
    corr[idx, idx] = 1.0
    return corr


def _run_name(meta: dict) -> str:
    run_folder = meta.get("run_folder")
    if run_folder:
        return Path(run_folder).name
    ckpt = meta.get("ckpt_path")
    if ckpt:
        return Path(ckpt).parent.name
    return "run"


def main() -> None:
    args = parse_model_load_args()

    bundles = load_model_bundles(
        path=args.path,
        prefer=args.prefer,
        device=args.device,
        dp_if_multi_gpu=args.dp,
        eval_mode=True,
        strict=True,
    )

    loader = get_cifar10_eval_loader(
        root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )

    errors_all: list[torch.Tensor] = []
    preds_all: list[torch.Tensor] = []
    labels_ref: torch.Tensor | None = None
    counts: list[int] = []
    accuracies: list[float] = []
    run_names: list[str] = []

    for bundle in bundles:
        encoder = bundle.encoder
        classifier = bundle.classifier
        if classifier is None:
            print(f"Skipping {_run_name(bundle.meta)}: classifier head not found.")
            continue

        device = next(encoder.parameters()).device
        errors, preds, labels = _collect_errors_and_preds(encoder, classifier, loader, device)

        if labels_ref is None:
            labels_ref = labels
        elif not torch.equal(labels_ref, labels):
            raise RuntimeError("Mismatched label ordering across trials.")

        errors_all.append(errors)
        preds_all.append(preds)
        counts.append(int(errors.sum().item()))
        accuracies.append(float((preds == labels).float().mean().item()))
        run_names.append(_run_name(bundle.meta))

    if not errors_all:
        print("No trials evaluated.")
        return

    print("Error counts per trial:")
    for name, count in zip(run_names, counts):
        print(f"{name}: {count}")

    print("Individual accuracies:")
    for name, acc in zip(run_names, accuracies):
        print(f"{name}: {acc:.4f}")

    error_matrix = torch.stack(errors_all)
    corr = _pearson_corrcoef(error_matrix)
    print("Correlation matrix:")
    print(corr.tolist())

    if corr.numel() and corr.size(0) > 1:
        idx = torch.triu_indices(corr.size(0), corr.size(1), offset=1)
        vals = corr[idx[0], idx[1]]
        vals = vals[torch.isfinite(vals)]
        if vals.numel():
            mean = float(vals.mean().item())
            std = float(vals.std(unbiased=vals.numel() > 1).item())
            print(f"Pairwise mean: {mean}")
            print(f"Pairwise std: {std}")
        else:
            print("Pairwise mean: nan")
            print("Pairwise std: nan")
    else:
        print("Pairwise mean: nan")
        print("Pairwise std: nan")

    if labels_ref is not None and preds_all:
        pred_matrix = torch.stack(preds_all)
        votes = torch.mode(pred_matrix, dim=0).values
        ensemble_acc = float((votes == labels_ref).float().mean().item())
        print(f"Ensemble accuracy: {ensemble_acc}")


if __name__ == "__main__":
    main()
