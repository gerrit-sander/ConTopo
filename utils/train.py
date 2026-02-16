import yaml
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets

class TwoCropTransform:
    """
    Apply the same transform twice and return both views.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def load_cifar10_metadata(config_path="configs/cifar10.yaml"):
    """
    Load CIFAR-10 class and animacy metadata from YAML.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

class AverageMeter(object):
    """Track current value, running sum, and average."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def unwrap(model):
    return model.module if isinstance(model, torch.nn.DataParallel) else model

def save_checkpoint(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(payload, path)
    print(f'>> Saved checkpoint to: {path}')

def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    Returns a list of tensors (percentages), one per k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # Output shape: [B, C] logits.
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # [B, maxk]
        pred = pred.t()                                                # [maxk, B]
        correct = pred.eq(target.view(1, -1).expand_as(pred))          # [maxk, B]

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
class tb_logger:
        class Logger:
            def __init__(self, logdir, flush_secs=2):
                os.makedirs(logdir, exist_ok=True)
                # Set flush_secs when supported.
                try:
                    self.writer = SummaryWriter(log_dir=logdir, flush_secs=flush_secs)
                except TypeError:
                    # Older PyTorch versions may not support flush_secs.
                    self.writer = SummaryWriter(log_dir=logdir)

            def log_value(self, tag, value, step):
                self.writer.add_scalar(tag, value, step)

            def close(self):
                self.writer.close()

def grad_norm(loss, params):
    # L2 norm of gradients of `loss` with respect to `params`.
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat = [g.detach().reshape(-1) for g in grads if g is not None]
    if len(flat) == 0:
        return torch.tensor(0.0, device=loss.device)
    v = torch.cat(flat)
    return torch.linalg.norm(v, ord=2)

def split_cifar10_train_val_indices(root, val_per_class=500):
    """
    Build a deterministic 45k/5k split from the CIFAR-10 train set.
    """
    base = datasets.CIFAR10(root=root, train=True, transform=None, download=True)
    targets = base.targets if hasattr(base, 'targets') else base.train_labels
    class_counts = {c: 0 for c in range(10)}
    val_idx = []
    for idx, y in enumerate(targets):
        y_int = int(y)
        if class_counts[y_int] < val_per_class:
            val_idx.append(idx)
            class_counts[y_int] += 1
        if all(class_counts[c] >= val_per_class for c in range(10)):
            break
    all_idx = set(range(len(targets)))
    train_idx = sorted(list(all_idx.difference(val_idx)))
    val_idx = sorted(val_idx)
    return train_idx, val_idx
