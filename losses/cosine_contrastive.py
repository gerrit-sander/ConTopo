import torch
import torch.nn as nn
import torch.nn.functional as F

def build_label_to_animacy(classnames, superclass, superclass_mapping):
    # Precompute once, outside the training loop if possible
    num_classes = len(classnames)
    la = [superclass_mapping[superclass[cls]] for cls in classnames]
    return torch.tensor(la, dtype=torch.long)  # shape [C]

class CosineContrastiveLoss(nn.Module):
    """
    Contrastive loss on cosine distance with animacy-aware margins.
    - Positive pairs: penalize distance above a small margin.
    - Negative pairs: enforce a margin that depends on animacy (same/different superclass).
    Returns mean loss over all pairs.
    """
    def __init__(self, label_to_animacy, margin_same=0.3, margin_diff=0.5, posdist_margin=0.05):
        super().__init__()
        self.margin_same = float(margin_same)
        self.margin_diff = float(margin_diff)
        self.posdist_margin = float(posdist_margin)
        # label_to_animacy: LongTensor [num_classes] mapping class idx -> animacy id
        self.register_buffer("label_to_animacy", label_to_animacy.clone().long(), persistent=False)

    def forward(self, projections, labels):
        # projections: [B, D], labels: [B] (class indices)
        if projections.dim() != 2:
            raise ValueError(f"'projections' must be [B, D], got {tuple(projections.shape)}")

        z = F.normalize(projections, dim=-1)                 # unit-norm once
        sim = z @ z.t()                                      # [B, B] cosine similarity
        dist = (1.0 - sim).clamp_(0.0, 2.0)                  # cosine distance in [0, 2]
        B = z.size(0)

        # Use only unordered pairs i<j
        triu = torch.triu(torch.ones(B, B, dtype=torch.bool, device=z.device), diagonal=1)
        d = dist[triu]                                       # [num_pairs]

        # Same-class mask -> y_true
        same_class = (labels[:, None] == labels[None, :])
        y_true = same_class[triu].float()                    # [num_pairs]

        # Animacy per sample, then pairwise equality
        anim = self.label_to_animacy[labels]                 # [B]
        same_anim = (anim[:, None] == anim[None, :])[triu]   # [num_pairs]

        # Margins per pair
        margins = torch.where(
            same_anim,
            torch.as_tensor(self.margin_same, device=z.device, dtype=d.dtype),
            torch.as_tensor(self.margin_diff, device=z.device, dtype=d.dtype),
        )

        # Loss terms (identical to yours)
        pos = y_true * F.relu(d - self.posdist_margin).pow(2)
        neg = (1.0 - y_true) * F.relu(margins - d).pow(2)
        return (pos + neg).mean()