import torch
import torch.nn as nn
import torch.nn.functional as F

class ShallowCNN(nn.Module):
    """
    A simple CNN bachbone architecture.
    This architecture consists of several convolutional layers followed by batch normalization,
    ReLU activation, and max pooling.
    The final output is a feature vector of specified embedding dimension used for the topographic loss.
    """
    def __init__(self, in_channels=3, emb_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1); self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1); self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1); self.bn4 = nn.BatchNorm2d(256)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        out = F.max_pool2d(F.relu(self.bn2(self.conv2(out))), 2)
        out = F.max_pool2d(F.relu(self.bn3(self.conv3(out))), 2)
        out = F.relu(self.bn4(self.conv4(out)))
        out = self.gap(out)
        out = torch.flatten(out, 1)
        return out
    
class ProjectionShallowCNN(nn.Module):
    """Projection head for the ShallowCNN architecture."""
    def __init__(self, emb_dim=256, feat_dim=128, p_dropout=0.2, use_dropout=True, ret_emb=False):
        super(ProjectionShallowCNN, self).__init__()
        self.ret_emb = ret_emb
        self.encoder = ShallowCNN(emb_dim=emb_dim, p_dropout=p_dropout, use_dropout=use_dropout)
        self.head = nn.Sequential(
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p_dropout) if use_dropout else nn.Identity(),
            nn.Linear(emb_dim, feat_dim, bias=True),
        )
    
    def forward(self, x):
        embeddings = self.encoder(x)
        features = F.normalize(self.head(embeddings), dim=1)
        return (embeddings, features) if self.ret_emb else features
    
class LinearShallowCNN(nn.Module):
    """ ShallowCNN architecture with a linear layer at the end for classification tasks."""
    def __init__(self, emb_dim=256, num_classes=10, p_dropout=0.2, use_dropout=True, ret_emb=False):
        super(LinearShallowCNN, self).__init__()
        self.ret_emb = ret_emb
        self.encoder = ShallowCNN(embedding_dim=emb_dim, p_dropout=p_dropout, use_dropout=use_dropout)
        self.fc = nn.Linear(emb_dim, num_classes)
    
    def forward(self, x):
        embeddings = self.encoder(x)
        logits = self.fc(embeddings)
        return (embeddings, logits) if self.ret_emb else logits


class LinearClassifier(nn.Module):
    """Linear classifier on top of a ShallowCNN encoder."""
    def __init__(self, emb_dim=256, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, features):
        logits = self.fc(features)
        return logits