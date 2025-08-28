import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import math

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
    opt = meta["args"]

    images, labels = [], []

    # iterate until we have all 10
    for imgs, labs in val_loader:
        for img, lab in zip(imgs, labs):
            # if we see a class for the first time, add it to the list
            if lab.item() not in labels:
                images.append(img)
                labels.append(lab.item())
            if len(images) == 10:
                break
        if len(images) == 10:
            break

    if len(images) < 10:
        raise RuntimeError(f"Only found {len(images)} distinct classes; need 10.")

    sample_imgs = torch.stack(images).to(device) # (10, 3, 32, 32)
    n_images = 10

    # hook function to grab fc activations (flattened)
    activations = []
    def hook_fn(module, inp, out):
        activations.append(out.detach().cpu().flatten(1))  # [B, D]

    enc = encoder.module if hasattr(encoder, 'module') else encoder
    if not hasattr(enc, 'fc'):
        raise AttributeError("Encoder has no 'fc' module to hook.")
    handle = enc.fc.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = encoder(sample_imgs)
    handle.remove()

    if not activations:
        raise RuntimeError("Forward hook did not fire; check that 'fc' exists on the encoder.")
    act = activations[0]

    emb_dim = act.shape[1]
    h = int(math.sqrt(emb_dim))
    while emb_dim % h != 0:
        h -= 1
    w = emb_dim // h

    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

    # undo dataset normalisation
    inv_norm = T.Normalize(
    mean=[-m/s for m, s in zip(CIFAR10_MEAN, CIFAR10_STD)],
    std=[1/s for s in CIFAR10_STD],
    )
    imgs_show = inv_norm(sample_imgs.cpu()).clamp(0, 1)

    rows = n_images
    fig, axes = plt.subplots(rows, 2, figsize=(6, 3*rows))

    v = act.abs().max().item()

    for i in range(rows):
        # original image
        axes[i, 0].imshow(imgs_show[i].permute(1, 2, 0))
        axes[i, 0].set_title(f"{class_names[labels[i]]}", fontsize=10)
        axes[i, 0].axis('off')

        # heat-map
        heat = act[i].reshape(h, w)
        im = axes[i, 1].imshow(heat, cmap='bwr', interpolation='nearest',
                            vmin=-v, vmax=v)
        axes[i, 1].set_title("FC1 activations", fontsize=10)
        axes[i, 1].axis('off')
        fig.colorbar(im, ax=axes[i, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    figurepath = resolve_figure_path(src, experiment="actmaps")
    plt.savefig(figurepath, dpi=200, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()