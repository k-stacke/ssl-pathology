import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
from torchvision.models import resnet50

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class Model(nn.Module):
    def __init__(self, feature_dim=128, pretrained=False):
        super(Model, self).__init__()

        self.f = resnet50(pretrained=pretrained)
        self.f.fc = Identity()
        # projection head
        self.g = nn.Sequential(nn.Linear(2048, 512, bias=False),
                               nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True),
                               nn.Linear(512, feature_dim, bias=True))

    @amp.autocast()
    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
