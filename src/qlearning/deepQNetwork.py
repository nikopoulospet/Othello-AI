import torch.nn as nn
import torch.nn.functional as F
import torch


class DQN(nn.Module):
    def __init__(self, inChannels, kernelSize, stride):
        super().__init__()

        # Model architecture defined by:
        # https://www.diva-portal.org/smash/get/diva2:1121059/FULLTEXT01.pdf
        # https://arxiv.org/pdf/1711.06583.pdf
        self.conv1 = nn.Conv2d(in_channels=6, out_channels=60, kernel_size=kernelSize, stride=stride)
        self.conv2 = nn.Conv2d(in_channels=60, out_channels=60, kernel_size=kernelSize, stride=stride)
        self.out = nn.Conv2d(in_channels=60, out_channels=1, kernel_size=kernelSize, stride=stride)

    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.relu(self.conv2(t))
        t = self.out(t)
        return t


class FCN(nn.Module):
    def __init__(self, inChannels, kernelSize, stride):
        super().__init__()

        self.fc1 = nn.Linear(64, 128).float()
        self.fc2 = nn.Linear(128, 128).float()
        self.fc3 = nn.Linear(128, 64).float()

    def forward(self, t):
        t = F.relu(self.fc1(t))
        t = F.relu(self.fc2(t))
        t = self.fc3(t)
        return t
