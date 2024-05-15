import torch
import torch.nn as nn

class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()

        self.gru = nn.GRU(input_size=28, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(in_features=64, out_features=10)

    def forward(self, X):
        X = X.view(-1, 28, 28)
        Y, _ = self.gru(X, None)

        Y = Y[:, -1, :]
        output = self.fc(Y)

        return output
    
