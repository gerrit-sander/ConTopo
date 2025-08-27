import torch
import torch.nn as nn
import torch.nn.functional as F

def prepare_pairs(batch_size, embeddings, device, labels, superclass, superclass_mapping, classnames):
    """
    Build all unordered pairs from a batch.
    Each pair gets:
      - y_true: 1 if same class, else 0
      - animacy labels: mapped superclass IDs for each item
      - pairs: stacked embeddings shaped [num_pairs, 2, D]
    """
    if embeddings.dim() != 2:
        raise ValueError(f"'embeddings' must be 2D [B, D], got {tuple(embeddings.shape)}")
    if labels.dim() != 1:
        raise ValueError(f"'labels' must be 1D [B], got {tuple(labels.shape)}")
    if batch_size != embeddings.size(0) or batch_size != labels.size(0):
        raise ValueError(f"'batch_size' must match embeddings/labels (got {batch_size}, "
                         f"{embeddings.size(0)}, {labels.size(0)})")
    if batch_size < 2:
        raise ValueError("Batch size must be at least 2 to form pairs.")

    y_true = []
    pairs = []
    animacy_labels = []

    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            # 1 if same class, 0 otherwise
            y_true.append(1 if labels[i].item() == labels[j].item() else 0)

            # Map class → superclass → integer ID (animacy)
            cls_i = classnames[labels[i].item()]
            cls_j = classnames[labels[j].item()]
            animacy_labels.append([
                superclass_mapping[superclass[cls_i]],
                superclass_mapping[superclass[cls_j]],
            ])

            # Pair of embeddings: [2, D]
            pairs.append(torch.stack([embeddings[i], embeddings[j]]))  # [2, D]

    # Tensorize and keep shapes explicit
    y_true = torch.tensor(y_true, dtype=torch.float32, device=device)
    animacy_labels = torch.tensor(animacy_labels, dtype=torch.int32, device=device)
    pairs = torch.stack(pairs)  # [num_pairs, 2, D]
    pairs = pairs.view(-1, 2, embeddings.size(-1))
    return y_true, pairs, animacy_labels


class CosineContrastiveLoss(nn.Module):
    """
    Contrastive loss on cosine distance with animacy-aware margins.
    - Positive pairs: penalize distance above a small margin.
    - Negative pairs: enforce a margin that depends on animacy (same/different superclass).
    Returns mean loss over all pairs.
    """
    def __init__(self, superclass, superclass_mapping, classnames, margin_same=0.3, margin_diff=0.5):
        super(CosineContrastiveLoss, self).__init__()
        self.margin_same = margin_same
        self.margin_diff = margin_diff
        self.superclass = superclass
        self.superclass_mapping = superclass_mapping
        self.classnames = classnames

    def forward(self, projections, labels):
        # Expect 2D projection matrix: [B, D]
        if projections.dim() != 2:
            raise ValueError(f"'projections' must be 2D [B, D], got {tuple(projections.shape)}")

        # Build pairwise targets and pair tensors
        y_true, y_pred, animacy_labels = prepare_pairs(
            batch_size=projections.size(0),
            embeddings=projections,
            device=projections.device,
            labels=labels,
            superclass=self.superclass,
            superclass_mapping=self.superclass_mapping,
            classnames=self.classnames
        )

        # Cosine distance in [0, 2]; lower means more similar
        cosine_distances = 1 - F.cosine_similarity(y_pred[:, 0, :], y_pred[:, 1, :], dim=-1)

        # Animacy-aware margins: tighter for same animacy, looser for different
        posdist_margin = 0.05  # small tolerance for positives
        same_animacy = (animacy_labels[:, 0] == animacy_labels[:, 1])
        margins = torch.where(
            same_animacy,
            torch.full_like(cosine_distances, float(self.margin_same)),
            torch.full_like(cosine_distances, float(self.margin_diff)),
        )

        # Positive: push distance toward ≤ posdist_margin
        positive_distances = y_true * torch.square(torch.clamp(cosine_distances - posdist_margin, min=0.0))
        # Negative: push distance to be ≥ margin
        negative_distances = (1 - y_true) * torch.square(torch.clamp(margins - cosine_distances, min=0.0))

        return torch.mean(positive_distances + negative_distances)