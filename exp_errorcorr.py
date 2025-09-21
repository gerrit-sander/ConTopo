"""Evaluate pairwise error correlations across model trials.

This script runs a classification experiment for every model folder contained in
the path supplied on the command line.  It leverages the unified loading helper
`load_model_bundles` so both cross-entropy and contrastive+readout checkpoints
are supported transparently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from utils.load import parse_model_load_args, load_model_bundles
from utils.experiments import get_cifar10_eval_loader


def _collect_errors_and_preds(
    encoder: torch.nn.Module,
    classifier: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    total = len(loader.dataset)
    errors = torch.zeros(total, dtype=torch.float32)
    preds = torch.empty(total, dtype=torch.long)
    targets = torch.empty(total, dtype=torch.long)
    offset = 0
    logits_store: torch.Tensor | None = None

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
            logits_cpu = logits.detach().cpu()
            if logits_store is None:
                logits_store = torch.empty(total, logits_cpu.size(1), dtype=logits_cpu.dtype)
            logits_store[offset : offset + size] = logits_cpu
            offset += size

    if logits_store is None:
        logits_store = torch.empty(total, 0)

    return errors, preds, targets, logits_store


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


def ensemble_accuracy(
    logits_list: list[torch.Tensor],
    labels: torch.Tensor,
    method: str = "soft",
) -> float:
    """Replicate professor's ensemble routines using numpy for parity."""

    if not logits_list:
        raise ValueError("logits_list must contain at least one model output")

    logits_np = [logits.detach().cpu().numpy() for logits in logits_list]
    labels_np = labels.detach().cpu().numpy()

    num_samples, num_classes = logits_np[0].shape
    for logits in logits_np:
        if logits.shape != (num_samples, num_classes):
            raise ValueError("All logits must share the same shape")

    probs = [np.exp(l) / np.exp(l).sum(axis=1, keepdims=True) for l in logits_np]

    if method == "hard":
        preds = np.array([np.argmax(l, axis=1) for l in logits_np])
        final_preds = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=num_classes).argmax(), axis=0, arr=preds
        )
    elif method == "soft":
        avg_probs = np.mean(probs, axis=0)
        final_preds = np.argmax(avg_probs, axis=1)
    elif method == "max_confidence":
        probs_stack = np.stack(probs, axis=0)
        max_conf = probs_stack.max(axis=2)
        best_model = np.argmax(max_conf, axis=0)
        final_preds = np.array(
            [np.argmax(probs_stack[best_model[i], i]) for i in range(num_samples)]
        )
    elif method == "conf_weighted":
        probs_stack = np.stack(probs, axis=0)
        confs = probs_stack.max(axis=2)
        weights = confs / confs.sum(axis=0, keepdims=True)
        weighted_probs = np.einsum("mn,mnc->nc", weights, probs_stack)
        final_preds = np.argmax(weighted_probs, axis=1)
    else:
        raise ValueError(
            "Unknown method. Choose from ['hard', 'soft', 'max_confidence', 'conf_weighted']."
        )

    return float(np.mean(final_preds == labels_np))


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
    logits_all: list[torch.Tensor] = []
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
        errors, preds, labels, logits = _collect_errors_and_preds(encoder, classifier, loader, device)

        if labels_ref is None:
            labels_ref = labels
        elif not torch.equal(labels_ref, labels):
            raise RuntimeError("Mismatched label ordering across trials.")

        errors_all.append(errors)
        preds_all.append(preds)
        logits_all.append(logits)
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

    non_ensemble_mean = float(np.mean(accuracies)) if accuracies else float("nan")
    non_ensemble_std = (
        float(np.std(accuracies, ddof=1)) if len(accuracies) > 1 else 0.0
    ) if accuracies else float("nan")
    if accuracies:
        print(
            f"Non-ensemble mean +/- std: {non_ensemble_mean:.4f} +/- {non_ensemble_std:.4f}"
        )

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

    ensemble_results: dict[str, float] = {}
    if labels_ref is not None and logits_all:
        for method in ("soft", "hard", "max_confidence", "conf_weighted"):
            acc = ensemble_accuracy(logits_all, labels_ref, method=method)
            ensemble_results[method] = acc
            print(f"Ensemble accuracy ({method}): {acc:.4f}")

    if accuracies or ensemble_results:
        headers = [
            "Non-ensemble",
            "Soft Vote",
            "Hard Vote",
            "Max Conf",
            "Conf Weighted",
        ]
        row_values = [
            (
                f"{non_ensemble_mean:.4f} +/- {non_ensemble_std:.4f}"
                if accuracies
                else "nan"
            ),
            (
                f"{ensemble_results.get('soft', float('nan')):.4f}"
                if 'soft' in ensemble_results
                else "nan"
            ),
            (
                f"{ensemble_results.get('hard', float('nan')):.4f}"
                if 'hard' in ensemble_results
                else "nan"
            ),
            (
                f"{ensemble_results.get('max_confidence', float('nan')):.4f}"
                if 'max_confidence' in ensemble_results
                else "nan"
            ),
            (
                f"{ensemble_results.get('conf_weighted', float('nan')):.4f}"
                if 'conf_weighted' in ensemble_results
                else "nan"
            ),
        ]

        widths = [max(len(h), len(v)) for h, v in zip(headers, row_values)]
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
        sep_line = "-+-".join("-" * w for w in widths)
        value_line = " | ".join(v.ljust(w) for v, w in zip(row_values, widths))

        print("\nAccuracy summary:")
        print(header_line)
        print(sep_line)
        print(value_line)


if __name__ == "__main__":
    main()
