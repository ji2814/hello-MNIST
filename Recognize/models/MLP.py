import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(in_features=28 * 28, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, X):
        X = X.view(-1, 28 *28)
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))

        return X
