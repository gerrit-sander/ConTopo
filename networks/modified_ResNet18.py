import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    """Basic block for ResNet18 architecture. Each block consists of two convolutional layers."""
    def __init__(self, in_channels, channels, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    
class ResNet18(nn.Module):
    """
    Basic ResNet18 encoder architecture. To have a quadratic number as output dimension for the
    topographic constraint, the number of channels is reduced from 512 to 256 in the last layer.
    """
    def __init__(self, in_channels=3, emb_dim=256):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = nn.Sequential(Block(64, 64, stride=1), Block(64, 64, stride=1))
        self.layer2 = nn.Sequential(Block(64, 128, stride=2), Block(128, 128, stride=1))
        self.layer3 = nn.Sequential(Block(128, 256, stride=2), Block(256, 256, stride=1))
        self.layer4 = nn.Sequential(Block(256, emb_dim, stride=2), Block(emb_dim, emb_dim, stride=1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out
    
class ProjectionResNet18(nn.Module):
    """ResNet18 architecture with a projection head for contrastive learning."""
    def __init__(self, emb_dim=256, feat_dim=128, ret_emb=False):
        super(ProjectionResNet18, self).__init__()
        self.ret_emb = ret_emb
        self.emb_dim = emb_dim
        self.encoder = ResNet18(emb_dim=emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Linear(emb_dim, feat_dim)
        )
    
    def forward(self, x):
        embeddings = self.encoder(x)
        features = F.normalize(self.head(embeddings), dim=1)
        return (embeddings, features) if self.ret_emb else features

class LinearResNet18(nn.Module):
    """ResNet18 architecture with a linear layer at the end for classification tasks."""
    def __init__(self, emb_dim=256, num_classes=10, ret_emb=False):
        super(LinearResNet18, self).__init__()
        self.ret_emb = ret_emb
        self.encoder = ResNet18(emb_dim=emb_dim)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        embeddings = self.encoder(x)
        logits = self.fc(embeddings)
        return (embeddings, logits) if self.ret_emb else logits