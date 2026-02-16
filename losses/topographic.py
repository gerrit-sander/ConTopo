import torch
import torch.nn as nn
import math
import torch.nn.functional as F

def get_grid_shape(n_units):
    """
    Choose (h, w) with h * w == n_units and h close to w.
    """
    h = int(math.sqrt(n_units))
    while n_units % h != 0:
        h -= 1
    w = n_units // h
    return h, w

def pos_dist(embedding_dim):
        """
        Build pairwise Euclidean distances for a near-square 2D unit grid.
        """
        h, w = get_grid_shape(embedding_dim)
        y = torch.linspace(0, 1, steps=h)
        x = torch.linspace(0, 1, steps=w)
        YY, XX = torch.meshgrid(y, x, indexing='ij')
        pos_hw2 = torch.stack([XX, YY], dim=-1)
        pos = pos_hw2.reshape(-1, 2)
        D = torch.cdist(pos, pos, p=2)
        return D

class Global_Topographic_Loss(nn.Module):
    """
    Global topographic regularizer on pre-activation features.
    Encourages unit similarity to decay with spatial distance.
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

        # Keep D on the same device as inputs.
        self.D = self.D.to(pre_relu.device)

        _, n_units = pre_relu.shape

        if self.D.shape != (n_units, n_units):
            raise ValueError(f"D must have shape ({n_units}, {n_units}), got {tuple(self.D.shape)}")

        # Cosine similarity across units (columns).
        Xn = F.normalize(pre_relu, p=2, dim=0, eps=1e-12)  # [B, C], normalized per unit.
        S = Xn.t() @ Xn                                    # [C, C] cosine similarity matrix.

        # Use upper triangle (i<j) to avoid double-counting/self-pairs.
        i_idx, j_idx = torch.triu_indices(
            n_units, n_units, offset=1, device=pre_relu.device
        )
        d = self.D[i_idx, j_idx]
        s = S[i_idx, j_idx]

        # Quadratic penalty toward 1/(d+1), averaged over unordered pairs.
        topo_loss_val = ((s - (1.0 / (d + 1.0))) ** 2).sum()
        return self.weight * (2.0 / (n_units * (n_units - 1))) * topo_loss_val
    
class Local_WS_Loss(nn.Module):
    """
    Local weight-smoothing regularizer for a linear layer.
    Arrange output units on a grid and penalize neighbor-row differences.
    """
    def __init__(self, weight=1.0):
        super(Local_WS_Loss, self).__init__()
        self.weight = weight

    def forward(self, linear_layer=None):
        if linear_layer is None:
            raise ValueError("linear_layer must be provided.")

        if not isinstance(linear_layer, nn.Linear):
            raise ValueError("linear_layer must be an instance of nn.Linear.")

        W = linear_layer.weight
        out_feats, in_feats = W.shape
        if W.ndim != 2:
            raise ValueError("linear_layer must have 2 dimensions (out_feats, in_feats).")
        
        # Arrange output units on an (h, w) grid.
        h, w = get_grid_shape(out_feats) # (h, w)
        G = W.reshape(h, w, in_feats) # (h, w, in_feats)
        diffs = []
        # Horizontal neighbors.
        if w > 1:
            diffs.append(G[:, :-1, :] - G[:, 1:, :])          # (H, W-1, C)
        # Vertical neighbors.
        if h > 1:
            diffs.append(G[:-1, :, :] - G[1:, :, :])          # (H-1, W, C)
        # Diagonal down-right neighbors.
        if h > 1 and w > 1:
            diffs.append(G[:-1, :-1, :] - G[1:, 1:, :])       # (H-1, W-1, C)
        # Diagonal down-left neighbors.
        if h > 1 and w > 1:
            diffs.append(G[:-1, 1:, :] - G[1:, :-1, :])       # (H-1, W-1, C)

        if not diffs:
            return torch.zeros((), device=W.device, dtype=W.dtype)

        # L2 over feature dimension, then mean over all neighbor pairs.
        dists = [torch.linalg.norm(d, dim=-1) for d in diffs]
        topo_loss_val = torch.cat([x.reshape(-1) for x in dists]).mean()

        return self.weight * topo_loss_val
