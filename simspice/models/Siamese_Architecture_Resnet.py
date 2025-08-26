import torch
from torch import nn
import tqdm
import torch.nn.functional as F
import pytorch_lightning as pl # type: ignore
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead # type: ignore
from lightly.loss import NTXentLoss # type: ignore


# ---------- Contrastive Loss ----------
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Euclidean distance
        euclidean_distance = F.pairwise_distance(output1, output2)
        # Contrastive loss formula
        loss = torch.mean(
            label * euclidean_distance.pow(2) +
            (1 - label) * torch.clamp(self.margin - euclidean_distance, min=0.0).pow(2)
        )
        return loss

# ---------- ResNet1D Backbone ----------
class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
    

class Bottleneck1D(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * self.expansion)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)
    

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, in_channels=1,
                 base_channels=64, output_dim=128):
        super(ResNet1D, self).__init__()
        self.in_planes = base_channels

        self.conv1 = nn.Conv1d(in_channels, base_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base_channels)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, base_channels,
                                       num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels*2,
                                       num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels*4,
                                       num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels*8,
                                       num_blocks[3], stride=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(base_channels*8*block.expansion, output_dim)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (B, C, L)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.global_pool(out)  # (B, C, 1)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc(out))
        return F.normalize(out, dim=1)


def ResNet18_1D(output_dim=128):
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], output_dim=output_dim)


def ResNet50_1D(output_dim=128):
    return ResNet1D(Bottleneck1D, [3, 4, 6, 3], output_dim=output_dim)


# ---------- SimSiam Model ----------
class SimSiam(pl.LightningModule):
    def __init__(self, output_dim=64, backbone_output_dim=128, 
                 hidden_layer_dim=128):
        super().__init__()
        self.backbone = ResNet50_1D(output_dim=backbone_output_dim)
        self.projection_head = SimSiamProjectionHead(
            backbone_output_dim, hidden_layer_dim, output_dim
        )
        self.prediction_head = SimSiamPredictionHead(
            output_dim, hidden_layer_dim, output_dim
        )

    @staticmethod
    def negative_cosine_similarity(p, z):
        # Normalize vectors to unit length
        p = F.normalize(p, dim=1)
        z = F.normalize(z, dim=1)
        # Return negative cosine similarity
        return -(p * z).sum(dim=1).mean()

    def forward(self, x):
        f = self.backbone(x)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1) = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)

        # Stop-gradient on target representations
        loss = 0.5 * (
            self.negative_cosine_similarity(p0, z1.detach()) +
            self.negative_cosine_similarity(p1, z0.detach())
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(), lr=0.06, momentum=0.9, weight_decay=1e-4
        )

