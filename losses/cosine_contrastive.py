import torch
import torch.nn as nn
import torch.nn.functional as F

def prepare_pairs(batch_size, embeddings, device, labels, superclass, superclass_mapping, classnames):
    """
    Prepare pairs of embeddings and their corresponding labels for contrastive loss computation.
    This function generates pairs of embeddings based on their labels and animacy classes.
    It returns the true labels for the pairs, the pairs themselves, and animacy labels.
    """
    y_true = []
    pairs = []
    animacy_labels = []

    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            if labels[i].item() == labels[j].item():
                y_true.append(1)
            else:
                y_true.append(0)

            cls_i = classnames[labels[i].item()]
            cls_j = classnames[labels[j].item()]
            animacy_labels.append([
                superclass_mapping[superclass[cls_i]],
                superclass_mapping[superclass[cls_j]],
            ])

            pairs.append(torch.stack([embeddings[i], embeddings[j]]))

    y_true = torch.tensor(y_true, dtype=torch.float32, device=device)
    animacy_labels = torch.tensor(animacy_labels, dtype=torch.int32, device=device)
    pairs = torch.stack(pairs)
    pairs = pairs.view(-1, 2, embeddings.size(-1))
    return y_true, pairs, animacy_labels  

class CosineContrastiveLoss(nn.Module):
    """
    Cosine Contrastive Loss for training models with contrastive learning.
    This loss function computes the cosine distance between pairs of embeddings and applies dynamic margins
    based on animacy labels.
    The animacy labels are used to determine whether the pairs belong to the same or different animacy classes.
    The loss is computed as the mean of the positive and negative distances, with margins applied to the negative distances.
    """
    def __init__(self, superclass, superclass_mapping, classnames, margin_same=0.3, margin_diff=0.5):
        super(CosineContrastiveLoss, self).__init__()
        self.margin_same = margin_same
        self.margin_diff = margin_diff
        self.superclass = superclass
        self.superclass_mapping = superclass_mapping
        self.classnames = classnames

    def forward(self, projections, labels):
        y_true, y_pred, animacy_labels = prepare_pairs(
            batch_size=projections.size(0),
            embeddings=projections,
            device=projections.device,
            labels=labels,
            superclass=self.superclass,
            superclass_mapping=self.superclass_mapping,
            classnames=self.classnames
        )

        cosine_distances = 1 - F.cosine_similarity(y_pred[:, 0, :], y_pred[:, 1, :], dim=-1)

        # Determine dynamic margins based on animacy
        # NOTE there is a also a hard wired positvedistance margin
        posdist_margin = 0.05
        margins = torch.where(animacy_labels[:, 0] == animacy_labels[:, 1], self.margin_same, self.margin_diff)
        positive_distances = y_true * torch.square(torch.clamp(cosine_distances - posdist_margin, min=0.0))
        # positive_distances = y_true * torch.square(cosine_distances)
        negative_distances = (1 - y_true) * torch.square(torch.clamp(margins - cosine_distances, min=0.0))

        return torch.mean(positive_distances + negative_distances)