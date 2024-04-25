import torch
import torch.nn as nn

# LeNet-5模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)
        
    def forward(self, X):
        x = self.pool1(torch.relu(self.conv1(X)))
        x = self.pool2(torch.relu(self.conv2(X)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(X))
        x = torch.relu(self.fc2(X))
        x = self.fc3(X)
        return X
    
# net = LeNet5()
# print(net)