import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import math

from utils.load import load_encoder_from_path, parse_model_load_args
from utils.experiments import get_cifar10_eval_loader
from utils.train import load_cifar10_metadata

def main():
    args = parse_model_load_args()

    # Load encoder
    encoder, meta = load_encoder_from_path(args.path, args.device, args.prefer, args.dp)
    src = meta.get("ckpt_path", args.path)

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
    opt = meta["args"]

    ### HERE ###


if __name__ == "__main__":
    main()