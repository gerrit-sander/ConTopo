import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def pos_dist(embedding_dim):
        """
        Generate a distance matrix for the positions in a grid of shape determined by embedding_dim. 
        (i,j) of D indicates the distance between units i and j.
        The grid is created such that the positions are evenly spaced in a 2D plane.
        The distance matrix is computed using the Euclidean distance.
        """
        
        h = int(math.sqrt(embedding_dim))
        while embedding_dim % h != 0:
            h -= 1
        w = embedding_dim // h

        y = torch.linspace(0, 1, steps=h)
        x = torch.linspace(0, 1, steps=w)
        YY, XX = torch.meshgrid(y, x, indexing='ij')
        pos_hw2 = torch.stack([XX, YY], dim=-1)
        pos = pos_hw2.reshape(-1, 2)
        D = torch.cdist(pos, pos, p=2)
        return D

class Global_Topographic_Loss(nn.Module):
    """
    Global topographic loss based on the cosine similarity of the pre-activation features.
    The loss encourages the cosine similarity between units to be inversely proportional to their distance in a
    precomputed distance matrix D.
    The distance matrix D is computed internally in __init__ using the pos_dist function with emb_dim.
    """
    def __init__(self, weight=1.0, emb_dim=256):
        super(Global_Topographic_Loss, self).__init__()
        self.weight = weight
        self.D = pos_dist(emb_dim)

    def forward(self, pre_relu):
        if pre_relu is None:
            raise ValueError("pre_relu must be provided.")
        if pre_relu.dim() != 2:
            raise ValueError(f"pre_relu must be 2D [B, C], got shape {tuple(pre_relu.shape)}")

        # Ensure D is on the same device as inputs
        self.D = self.D.to(pre_relu.device)

        _, n_units = pre_relu.shape

        if self.D.shape != (n_units, n_units):
            raise ValueError(f"D must have shape ({n_units}, {n_units}), got {tuple(self.D.shape)}")

        # Cosine similarity between units (columns)
        Xn = F.normalize(pre_relu, p=2, dim=0, eps=1e-12)  # [B, C]
        S = Xn.t() @ Xn                                    # (C, C)

        i_idx, j_idx = torch.triu_indices(
            n_units, n_units, offset=1, device=pre_relu.device
        )
        d = self.D[i_idx, j_idx]
        s = S[i_idx, j_idx]

        topo_loss_val = ((s - (1.0 / (d + 1.0))) ** 2).sum()
        return self.weight * (2.0 / (n_units * (n_units - 1))) * topo_loss_val
