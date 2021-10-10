import torch.nn as nn
import torch.nn.functional as F
import torch


class DQN(nn.Module):
    def __init__(self, inChannels, kernelSize, stride):
        super().__init__()

        # Model architecture defined by:
        # https://www.diva-portal.org/smash/get/diva2:1121059/FULLTEXT01.pdf
        # https://arxiv.org/pdf/1711.06583.pdf
        self.conv1 = nn.Conv2d(in_channels=7, out_channels=60, kernel_size=kernelSize, stride=stride)
        self.conv2 = nn.Conv2d(in_channels=60, out_channels=60, kernel_size=kernelSize, stride=stride)
        self.conv3 = nn.Conv2d(in_channels=60, out_channels=60, kernel_size=kernelSize, stride=stride)
        self.conv4 = nn.Conv2d(in_channels=60, out_channels=60, kernel_size=kernelSize, stride=stride)
        self.conv5 = nn.Conv2d(in_channels=60, out_channels=60, kernel_size=kernelSize, stride=stride)
        self.conv6 = nn.Conv2d(in_channels=60, out_channels=60, kernel_size=kernelSize, stride=stride)
        self.conv7 = nn.Conv2d(in_channels=60, out_channels=60, kernel_size=kernelSize, stride=stride)
        self.conv8 = nn.Conv2d(in_channels=60, out_channels=60, kernel_size=kernelSize, stride=stride)
        self.conv9 = nn.Conv2d(in_channels=60, out_channels=60, kernel_size=kernelSize, stride=stride)
        self.conv10 = nn.Conv2d(in_channels=60, out_channels=60, kernel_size=kernelSize, stride=stride)
        self.out = nn.Conv2d(in_channels=60, out_channels=1, kernel_size=kernelSize, stride=stride)

    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.relu(self.conv2(t))
        t = F.relu(self.conv3(t))
        t = F.relu(self.conv4(t))
        t = F.relu(self.conv5(t))
        t = F.relu(self.conv6(t))
        t = F.relu(self.conv7(t))
        t = F.relu(self.conv8(t))
        t = F.relu(self.conv9(t))
        t = F.relu(self.conv10(t))
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
