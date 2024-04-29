import torch
from torch import nn

class ResNet(nn.Module):
    def __init__(self, useConv=False):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=8)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=8)

        if useConv:
            self.conv3 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=1, stride=1)
        else:
            self.conv3 = None

        self.conv4 = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=1, stride=1)

        self.fc = nn.Linear(in_features=28 * 28, out_features=10)

    def forward(self, X):
        Y =self.bn1(self.conv1(X))
        Y = torch.relu(Y)

        Y = self.bn2(self.conv2(Y))

        if self.conv3:
            X = self.conv3(X)

        Y += X
        Y = self.conv4(Y)
        Y = Y.view(-1, 28 * 28)
        Y = self.fc(Y)

        return Y

   