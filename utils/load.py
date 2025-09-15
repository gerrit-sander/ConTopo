import argparse
import os
import torch

from utils.train import unwrap

from networks.shallowCNN import LinearShallowCNN, ProjectionShallowCNN
from networks.modified_ResNet18 import LinearResNet18, ProjectionResNet18


def _maybe_fix_state_dict_keys(state_dict: dict, model: torch.nn.Module) -> dict:
    """Normalize DataParallel prefixes: add/remove 'module.' to match `model`."""
    has_module = any(k.startswith("module.") for k in state_dict.keys())
    is_dp = isinstance(model, torch.nn.DataParallel)
    if has_module and not is_dp:
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    if (not has_module) and is_dp:
        return {f"module.{k}": v for k, v in state_dict.items()}
    return state_dict


def _infer_stage(ckpt: dict, ckpt_path: str) -> str:
    """
    Decide whether the checkpoint is 'e2e' (CE) or 'contrastive'.

    Priority:
    1) Use ckpt['stage'] if present.
    2) Else infer from filename.
    3) Else infer from args (presence of 'projection_dim' → contrastive).
    """
    stage = ckpt.get("stage")
    if stage in {"e2e", "contrastive"}:
        return stage
    name = os.path.basename(ckpt_path).lower()
    if "contrastive" in name:
        return "contrastive"
    args = ckpt.get("args", {})

    return "contrastive" if "projection_dim" in args else "e2e"


def _build_head_from_args(args: dict, stage: str, device: torch.device, dp_if_multi_gpu: bool):
    """
    Recreate the SAME head used at train time (Linear* for CE, Projection* for contrastive),
    so weights load cleanly and we can then extract `.encoder`.
    """
    model_type = args.get("model_type", "shallowcnn")
    emb_dim = int(args.get("embedding_dim", 256))

    if stage == "contrastive":
        proj_dim = int(args.get("projection_dim", 128))
        if model_type == "resnet18":
            model = ProjectionResNet18(emb_dim=emb_dim, feat_dim=proj_dim, ret_emb=True)
        else:
            model = ProjectionShallowCNN(emb_dim=emb_dim, feat_dim=proj_dim, ret_emb=True, use_dropout=False)
    else:  # 'e2e' CE
        num_classes = int(args.get("num_classes", 10))
        if model_type == "resnet18":
            model = LinearResNet18(emb_dim=emb_dim, num_classes=num_classes, ret_emb=True)
        else:
            model = LinearShallowCNN(emb_dim=emb_dim, num_classes=num_classes, ret_emb=True, use_dropout=False)

    if torch.cuda.is_available():
        model = model.to(device)
        if dp_if_multi_gpu and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

    return model

def load_encoder_from_ckpt(
    ckpt_path: str,
    device: str | torch.device | None = None,
    eval_mode: bool = True,
    dp_if_multi_gpu: bool = False,
    strict: bool = True,
):
    """
    Load ONLY the encoder (ShallowCNN/ResNet18) from a single checkpoint.

    Supported sources
    -----------------
    • CE (end-to-end) checkpoints: e2e_*.pth  → Linear* wrapper
    • Contrastive checkpoints:     contrastive_*.pth → Projection* wrapper

    Returns
    -------
    encoder : torch.nn.Module
        The backbone encoder moved to `device`. Wrapped in DataParallel if requested and available.
    meta : dict
        Minimal metadata: {'epoch', 'stage', 'args', 'metrics', 'ckpt_path'}.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    stage = _infer_stage(ckpt, ckpt_path)

    # Reject linear-readout-only checkpoints (no encoder weights inside)
    if stage == "linear_readout" or ("linear_state_dict" in ckpt and "state_dict" not in ckpt):
        raise ValueError(
            "This checkpoint contains only the linear readout head (no encoder). "
            "Load a 'contrastive_*' or 'e2e_*' checkpoint instead."
        )

    args = ckpt.get("args", {})
    head = _build_head_from_args(args, stage, device, dp_if_multi_gpu)

    state_dict = ckpt.get("state_dict")
    if state_dict is None:
        raise KeyError("Checkpoint does not contain 'state_dict' with encoder+head weights.")

    # Align DP prefixes and load weights
    state_dict = _maybe_fix_state_dict_keys(state_dict, head)
    head.load_state_dict(state_dict, strict=strict)

    # Extract ONLY the encoder (unwrap in case of DataParallel)
    encoder = unwrap(head).encoder

    if torch.cuda.is_available():
        encoder = encoder.to(device)
        if dp_if_multi_gpu and torch.cuda.device_count() > 1:
            encoder = torch.nn.DataParallel(encoder)

    if eval_mode:
        encoder.eval()

    meta = {
        "epoch": ckpt.get("epoch"),
        "stage": stage,
        "args": args,
        "metrics": ckpt.get("metrics"),
        "ckpt_path": ckpt_path,
    }
    return encoder, meta


def load_encoder_from_run_folder(
    run_folder: str,
    prefer: str = "best",
    device: str | torch.device | None = None,
    **kwargs,
):
    """
    Convenience helper: pick a good checkpoint from a run folder and load the encoder.

    Selection order
    ---------------
    prefer='best': contrastive_best.pth → e2e_best.pth → contrastive_last.pth → e2e_last.pth
    prefer='last': contrastive_last.pth → e2e_last.pth → contrastive_best.pth → e2e_best.pth
    """
    order_best = ["contrastive_best.pth", "e2e_best.pth", "contrastive_last.pth", "e2e_last.pth"]
    order_last = ["contrastive_last.pth", "e2e_last.pth", "contrastive_best.pth", "e2e_best.pth"]
    candidates = order_best if prefer == "best" else order_last

    for fname in candidates:
        ckpt_path = os.path.join(run_folder, fname)
        if os.path.isfile(ckpt_path):
            return load_encoder_from_ckpt(ckpt_path, device=device, **kwargs)

    raise FileNotFoundError(
        f"No suitable checkpoint found in {run_folder}. "
        "Expected one of: contrastive_best/last.pth or e2e_best/last.pth"
    )

def load_encoder_from_path(path: str, device: str, prefer: str, dp: bool):
    """Accept either a checkpoint file or a run folder and return (encoder, meta)."""
    if os.path.isdir(path):
        encoder, meta = load_encoder_from_run_folder(
            run_folder=path,
            prefer=prefer,
            device=device,
            dp_if_multi_gpu=dp,
            eval_mode=True,
        )
    else:
        encoder, meta = load_encoder_from_ckpt(
            ckpt_path=path,
            device=device,
            dp_if_multi_gpu=dp,
            eval_mode=True,
        )
    return encoder, meta

EXPECTED_CKPT_NAMES = {
    "contrastive_best.pth",
    "contrastive_last.pth",
    "e2e_best.pth",
    "e2e_last.pth",
}


def _dir_contains_expected_ckpts(d: str) -> bool:
    """Return True only if directory has one of the expected run checkpoint files."""
    if not os.path.isdir(d):
        return False
    try:
        entries = os.listdir(d)
    except FileNotFoundError:
        return False
    return any(name in EXPECTED_CKPT_NAMES for name in entries)

def list_run_folders_from_model_folder(model_folder: str) -> list[str]:
    """
    Given a model folder that may aggregate multiple trials, return a list of
    run folders (each containing checkpoints).

    Handles both layouts:
    - Nested trials: <model_folder>/trial_00, trial_01, ... (each with *.pth)
    - Flat trial folder: <model_folder> itself contains *.pth
    """
    model_folder = os.path.abspath(model_folder)
    # If the model folder itself contains expected checkpoint files, treat it as a run folder.
    if _dir_contains_expected_ckpts(model_folder):
        return [model_folder]
    # Otherwise, look for immediate child dirs that contain checkpoints
    runs: list[str] = []
    for name in sorted(os.listdir(model_folder)):
        d = os.path.join(model_folder, name)
        if os.path.isdir(d) and _dir_contains_expected_ckpts(d):
            runs.append(d)
    if runs:
        return runs
    raise FileNotFoundError(
        f"No checkpoints found under model folder: {model_folder}. "
        "Expected *.pth files directly or inside subfolders (e.g., trial_00)."
    )

def load_encoders_from_model_folder(
    model_folder: str,
    prefer: str = "best",
    device: str | torch.device | None = None,
    dp_if_multi_gpu: bool = False,
    eval_mode: bool = True,
    strict: bool = True,
):
    """
    Load encoders for all trials inside a given model folder.

    Returns a list of (encoder, meta) for each discovered run folder.
    """
    run_folders = list_run_folders_from_model_folder(model_folder)
    out = []
    for run in run_folders:
        enc, meta = load_encoder_from_run_folder(
            run_folder=run,
            prefer=prefer,
            device=device,
            dp_if_multi_gpu=dp_if_multi_gpu,
            eval_mode=eval_mode,
            strict=strict,
        )
        out.append((enc, meta))
    return out

def parse_model_load_args():
    parser = argparse.ArgumentParser(
        description="Load a trained encoder (from CE or contrastive) and run eval-time experiments."
    )
    parser.add_argument(
        "path",
        help=(
            "Relative path to a checkpoint file (e2e_*.pth / contrastive_*.pth) "
            "or a run folder that contains them."
        ),
    )
    parser.add_argument(
        "--prefer",
        choices=["best", "last"],
        default="best",
        help="When 'path' is a run folder, choose which checkpoint to load.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to load the model on, e.g. 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--dp",
        action="store_true",
        help="Wrap the returned encoder in DataParallel if multiple GPUs are available.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--dataset-root", default="./dataset")
    return parser.parse_args()
