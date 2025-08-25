from utils.load import load_encoder_from_path, parse_model_load_args
from utils.experiments import get_cifar10_eval_loader, resolve_figure_path
from utils.train import load_cifar10_metadata
import matplotlib.pyplot as plt

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

    ### TODO: IMPLEMENT AN EXPERIMENT ###

    plt.tight_layout()
    figurepath = resolve_figure_path(src, experiment="template")
    plt.savefig(figurepath, dpi=200, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()