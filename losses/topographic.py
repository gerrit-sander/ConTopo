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
    The distance matrix D should be precomputed using the pos_dist function.
    """
    def __init__(self, weight=1.0, emb_dim=256):
        super(Global_Topographic_Loss, self).__init__()
        self.weight = weight
        D = pos_dist(repr_dim=emb_dim)

    def forward(self, pre_relu):

        device = (torch.device('cuda')
                  if pre_relu.is_cuda
                  else torch.device('cpu'))
        
        self.D.to(device)

        if pre_relu is None:
            raise ValueError("pre_relu must be provided.")

        _, n_units = pre_relu.shape

        if self.D is None:
            raise ValueError("D must be provided (precompute with pos_dist(n_units)).")
        if self.D.shape != (n_units, n_units):
            raise ValueError(f"D must have shape ({n_units}, {n_units}), got {tuple(self.D.shape)}")

        # Cosine similarity between units
        Xn = F.normalize(pre_relu, p=2, dim=0, eps=1e-12)   # [B, C]
        S = Xn.t() @ Xn                                     # (C, C)

        i_idx, j_idx = torch.triu_indices(n_units, n_units, offset=1, device=pre_relu.device)
        d = self.D[i_idx, j_idx]
        s = S[i_idx, j_idx]

        topo_loss_val = ((s - (1.0 / (d + 1.0))) ** 2).sum()

        return self.weight * (2.0 / (n_units * (n_units - 1))) * topo_loss_val
