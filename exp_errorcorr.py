import os
import math
import torch

from utils.load import (
    parse_model_load_args,
    list_run_folders_from_model_folder,
    load_encoder_from_ckpt,
)
from utils.experiments import get_cifar10_eval_loader
from networks.shallowCNN import LinearShallowCNN, LinearClassifier
from networks.modified_ResNet18 import LinearResNet18

_E2E_BEST = ("e2e_best.pth", "e2e_last.pth")
_E2E_LAST = ("e2e_last.pth", "e2e_best.pth")
_CONTRASTIVE_BEST = ("contrastive_best.pth", "contrastive_last.pth")
_CONTRASTIVE_LAST = ("contrastive_last.pth", "contrastive_best.pth")
_READOUT_BEST = ("readout_best.pth", "readout_last.pth")
_READOUT_LAST = ("readout_last.pth", "readout_best.pth")


def _pick_ckpt(run: str, prefer: str, best: tuple[str, ...], last: tuple[str, ...]) -> str | None:
    order = best if prefer == "best" else last
    for name in order:
        path = os.path.join(run, name)
        if os.path.isfile(path):
            return path
    return None


def _clean_state_dict(state_dict: dict, model: torch.nn.Module) -> dict:
    has_module = any(k.startswith("module.") for k in state_dict)
    is_dp = isinstance(model, torch.nn.DataParallel)
    if has_module and not is_dp:
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if (not has_module) and is_dp:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


def _build_e2e_model(args: dict, device: torch.device) -> torch.nn.Module:
    model_type = args.get("model_type", "shallowcnn")
    emb_dim = int(args.get("embedding_dim", 256))
    num_classes = int(args.get("num_classes", 10))
    use_dropout = bool(args.get("use_dropout", True))
    p_dropout = float(args.get("p_dropout", 0.5))

    if model_type == "resnet18":
        model = LinearResNet18(
            emb_dim=emb_dim,
            num_classes=num_classes,
            use_dropout=use_dropout,
            p_dropout=p_dropout,
            ret_emb=True,
        )
    else:
        model = LinearShallowCNN(
            emb_dim=emb_dim,
            num_classes=num_classes,
            use_dropout=use_dropout,
            p_dropout=p_dropout,
            ret_emb=True,
        )
    return model.to(device)


def _load_e2e_model(path: str, device: torch.device) -> torch.nn.Module:
    ckpt = torch.load(path, map_location=device)
    model = _build_e2e_model(ckpt.get("args", {}), device)
    state_dict = ckpt.get("state_dict")
    if state_dict is None:
        raise KeyError(f"Missing 'state_dict' in {path}")
    model.load_state_dict(_clean_state_dict(state_dict, model), strict=True)
    model.eval()
    return model


def _load_readout_model(run: str, prefer: str, device: torch.device) -> torch.nn.Module | None:
    readout_path = _pick_ckpt(run, prefer, _READOUT_BEST, _READOUT_LAST)
    if readout_path is None:
        return None

    ckpt = torch.load(readout_path, map_location=device)
    state_dict = ckpt.get("linear_state_dict")
    if state_dict is None:
        raise KeyError(f"Missing 'linear_state_dict' in {readout_path}")

    classifier = LinearClassifier(
        emb_dim=int(ckpt.get("args", {}).get("embedding_dim", 256)),
        num_classes=int(ckpt.get("args", {}).get("num_classes", 10)),
    ).to(device)
    classifier.load_state_dict(state_dict, strict=True)

    # Linear readouts in `main_supcon.py` are trained on top of the encoder *as it stood at
    # the end of contrastive training (the "last" snapshot), not the validation-best one.
    # Pairing `readout_best.pth` with `contrastive_best.pth` therefore evaluates a
    # mismatched encoder/head combination and can tank accuracy (observed primarily for
    # SimCLR). Prefer the "last" encoder weights here and fall back to other choices only
    # if that file is missing.
    contrastive_path = None
    for candidate in ("contrastive_last.pth", "contrastive_best.pth"):
        path = os.path.join(run, candidate)
        if os.path.isfile(path):
            contrastive_path = path
            break
    if contrastive_path is None:
        contrastive_path = _pick_ckpt(run, prefer, _CONTRASTIVE_BEST, _CONTRASTIVE_LAST)
    if contrastive_path is None:
        raise FileNotFoundError(f"No contrastive checkpoint found for {run}")

    encoder, _ = load_encoder_from_ckpt(
        contrastive_path,
        device=device,
        dp_if_multi_gpu=False,
        eval_mode=True,
        strict=True,
    )

    class ReadoutModel(torch.nn.Module):
        def __init__(self, enc: torch.nn.Module, clf: torch.nn.Module):
            super().__init__()
            self.encoder = enc
            self.classifier = clf

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            embeddings = self.encoder(x)
            return self.classifier(embeddings)

    model = ReadoutModel(encoder, classifier).to(device)
    model.eval()
    return model


def _load_model_for_run(run: str, prefer: str, device: torch.device) -> torch.nn.Module | None:
    e2e_path = _pick_ckpt(run, prefer, _E2E_BEST, _E2E_LAST)
    if e2e_path is not None:
        try:
            return _load_e2e_model(e2e_path, device)
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {run}: failed to load e2e checkpoint ({exc}).")

    try:
        return _load_readout_model(run, prefer, device)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"Skipping {run}: failed to load readout ({exc}).")
        return None


def _collect_errors_and_preds(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total = len(loader.dataset)
    errors = torch.zeros(total, dtype=torch.float32)
    preds = torch.empty(total, dtype=torch.long)
    targets = torch.empty(total, dtype=torch.long)
    offset = 0

    with torch.no_grad():
        for batch in loader:
            images, labels = batch[:2] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
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


def main() -> None:
    args = parse_model_load_args()
    device = torch.device(args.device)

    try:
        runs = list_run_folders_from_model_folder(args.path)
    except FileNotFoundError as exc:
        print(exc)
        return

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

    for run in runs:
        model = _load_model_for_run(run, args.prefer, device)
        if model is None:
            print(f"Skipping {run}: no suitable checkpoint found.")
            continue

        errors, preds, labels = _collect_errors_and_preds(model, loader, device)
        if labels_ref is None:
            labels_ref = labels
        elif not torch.equal(labels_ref, labels):
            raise RuntimeError("Mismatched label ordering across trials.")

        errors_all.append(errors)
        preds_all.append(preds)
        counts.append(int(errors.sum().item()))
        accuracies.append(float((preds == labels).float().mean().item()))
        run_names.append(os.path.basename(run.rstrip(os.sep)))

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
