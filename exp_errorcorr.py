import os
import math
import torch

from utils.load import parse_model_load_args, list_run_folders_from_model_folder
from utils.experiments import get_cifar10_eval_loader
from networks.shallowCNN import LinearShallowCNN
from networks.modified_ResNet18 import LinearResNet18


def _infer_stage(ckpt: dict, ckpt_path: str) -> str:
    stage = ckpt.get("stage")
    if stage in {"e2e", "contrastive"}:
        return stage
    name = os.path.basename(ckpt_path).lower()
    if "contrastive" in name:
        return "contrastive"
    args = ckpt.get("args", {})
    return "contrastive" if "projection_dim" in args else "e2e"


def _maybe_fix_state_dict_keys(state_dict: dict, model: torch.nn.Module) -> dict:
    has_module = any(k.startswith("module.") for k in state_dict.keys())
    is_dp = isinstance(model, torch.nn.DataParallel)
    if has_module and not is_dp:
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if (not has_module) and is_dp:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


def _build_e2e_model(args: dict, device: torch.device, dp_if_multi_gpu: bool) -> torch.nn.Module:
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

    model = model.to(device)
    if dp_if_multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    return model


def _load_e2e_model_from_ckpt(
    ckpt_path: str,
    device: torch.device,
    dp_if_multi_gpu: bool,
    strict: bool = True,
) -> tuple[torch.nn.Module, dict]:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    stage = _infer_stage(ckpt, ckpt_path)
    if stage != "e2e":
        raise ValueError(f"Checkpoint is not an e2e model: {ckpt_path}")

    args = ckpt.get("args", {})
    wrapper = _build_e2e_model(args, device, dp_if_multi_gpu)
    state_dict = ckpt.get("state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint missing 'state_dict' entry")

    state_dict = _maybe_fix_state_dict_keys(state_dict, wrapper)
    wrapper.load_state_dict(state_dict, strict=strict)
    wrapper.eval()

    meta = {
        "epoch": ckpt.get("epoch"),
        "stage": stage,
        "args": args,
        "metrics": ckpt.get("metrics"),
        "ckpt_path": ckpt_path,
    }
    return wrapper, meta


def _select_e2e_checkpoint(run_folder: str, prefer: str) -> str | None:
    order_best = ["e2e_best.pth", "e2e_last.pth"]
    order_last = ["e2e_last.pth", "e2e_best.pth"]
    candidates = order_best if prefer == "best" else order_last
    for fname in candidates:
        ckpt_path = os.path.join(run_folder, fname)
        if os.path.isfile(ckpt_path):
            return ckpt_path
    return None


def _collect_errors_and_preds(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total = len(loader.dataset)
    errors = torch.zeros(total, dtype=torch.float32)
    predictions = torch.empty(total, dtype=torch.long)
    targets = torch.empty(total, dtype=torch.long)
    idx = 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                images, labels = batch[:2]
            else:
                images, labels = batch
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                logits = outputs[-1]
            else:
                logits = outputs
            preds = logits.argmax(dim=1)

            batch_errors = (preds != labels).to(dtype=torch.float32).cpu()
            batch_preds = preds.detach().cpu()
            batch_targets = labels.detach().cpu()
            batch_size = batch_errors.numel()

            errors[idx : idx + batch_size] = batch_errors
            predictions[idx : idx + batch_size] = batch_preds
            targets[idx : idx + batch_size] = batch_targets
            idx += batch_size
    return errors, predictions, targets


def _pearson_corrcoef(error_matrix: torch.Tensor) -> torch.Tensor:
    # error_matrix shape: [num_models, num_samples]
    if error_matrix.size(0) == 0:
        return torch.empty((0, 0), dtype=torch.float32)
    centered = error_matrix - error_matrix.mean(dim=1, keepdim=True)
    cov = centered @ centered.T
    var = centered.pow(2).sum(dim=1)
    denom = torch.sqrt(var).unsqueeze(1) * torch.sqrt(var).unsqueeze(0)
    corr = cov.clone()
    mask = denom == 0
    corr[~mask] = corr[~mask] / denom[~mask]
    corr.masked_fill_(mask, float("nan"))
    diag_indices = torch.arange(corr.size(0))
    corr[diag_indices, diag_indices] = 1.0
    return corr


def main():
    args = parse_model_load_args()

    if isinstance(args.device, str):
        device = torch.device(args.device)
    else:
        device = torch.device(args.device)

    try:
        run_folders = list_run_folders_from_model_folder(args.path)
    except FileNotFoundError as exc:
        print(str(exc))
        return

    loader = get_cifar10_eval_loader(
        root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )

    error_vectors = []
    pred_vectors = []
    counts = []
    trial_labels = []
    targets_ref = None

    for run in run_folders:
        ckpt_path = _select_e2e_checkpoint(run, args.prefer)
        if ckpt_path is None:
            print(f"Skipping {run}: no e2e checkpoint found.")
            continue
        try:
            model, _meta = _load_e2e_model_from_ckpt(
                ckpt_path=ckpt_path,
                device=device,
                dp_if_multi_gpu=args.dp,
                strict=True,
            )
        except Exception as exc:
            print(f"Skipping {run}: failed to load checkpoint ({exc}).")
            continue

        model = model.to(device)
        model.eval()
        errors, preds, targets = _collect_errors_and_preds(model, loader, device)
        if targets_ref is None:
            targets_ref = targets
        elif not torch.equal(targets_ref, targets):
            raise RuntimeError("Inconsistent target ordering between trials.")

        error_vectors.append(errors)
        pred_vectors.append(preds)
        counts.append(int(errors.sum().item()))
        trial_labels.append(os.path.basename(run.rstrip(os.sep)))

    if len(error_vectors) == 0:
        print("No trials evaluated.")
        return

    print("Error counts per trial:")
    for label, count in zip(trial_labels, counts):
        print(f"{label}: {count}")

    error_matrix = torch.stack(error_vectors, dim=0)
    corr_matrix = _pearson_corrcoef(error_matrix)
    print("Correlation matrix:")
    print(corr_matrix.tolist())

    if corr_matrix.numel() == 0 or corr_matrix.size(0) < 2:
        print("Pairwise mean: nan")
        print("Pairwise std: nan")
        return

    triu_indices = torch.triu_indices(corr_matrix.size(0), corr_matrix.size(1), offset=1)
    upper_vals = corr_matrix[triu_indices[0], triu_indices[1]]
    valid = upper_vals[torch.isfinite(upper_vals)]
    if valid.numel() == 0:
        print("Pairwise mean: nan")
        print("Pairwise std: nan")
        return

    mean = float(valid.mean().item())
    if valid.numel() > 1:
        var = float(((valid - mean) ** 2).sum().item() / (valid.numel() - 1))
        std = math.sqrt(var)
    else:
        std = 0.0
    print(f"Pairwise mean: {mean}")
    print(f"Pairwise std: {std}")

    if targets_ref is not None and len(pred_vectors) > 0:
        pred_matrix = torch.stack(pred_vectors, dim=0)
        votes = torch.mode(pred_matrix, dim=0).values
        ensemble_acc = float((votes == targets_ref).float().mean().item())
        print(f"Ensemble accuracy: {ensemble_acc}")


if __name__ == "__main__":
    main()
