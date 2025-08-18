import yaml

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