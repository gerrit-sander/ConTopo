import yaml
import os
import torch
from torch.utils.tensorboard import SummaryWriter

class TwoCropTransform:
    """
    A transform that applies the same transformation to two different crops of the input.
    This is useful for contrastive learning tasks.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def load_cifar10_metadata(config_path="configs/cifar10.yaml"):
    """
    Load CIFAR-10 metadata (classes, animacy labels, mappings) from a YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        dict: Dictionary with keys 'CIFAR10_CLASSES', 'ANIMACY', and 'ANIMACY_MAPPING'.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

class AverageMeter(object):
    """Computes and stores the average and current value"""
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

        # output: [B, C] logits
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
                # SummaryWriter flushes on close; set flush_secs if supported
                try:
                    self.writer = SummaryWriter(log_dir=logdir, flush_secs=flush_secs)
                except TypeError:
                    # Older PyTorch versions may not support flush_secs
                    self.writer = SummaryWriter(log_dir=logdir)

            def log_value(self, tag, value, step):
                self.writer.add_scalar(tag, value, step)

            def close(self):
                self.writer.close()

def grad_norm(loss, params):
    # L2 norm of grads of `loss` wrt `params` (no weight update)
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    flat = [g.detach().reshape(-1) for g in grads if g is not None]
    if len(flat) == 0:
        return torch.tensor(0.0, device=loss.device)
    v = torch.cat(flat)
    return torch.linalg.norm(v, ord=2)