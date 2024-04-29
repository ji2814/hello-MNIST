import torch
import torch.nn as nn

# 生成器，输入100噪声输出（1，28，28）
class Generator(nn.Module):
    def __init__(self, input_dim):
        super(Generator, self).__init__()
    
        self.fc1 = nn.Linear(in_features=input_dim, out_features=256)
        self.tanh1 = nn.Tanh()
        self.fc2 = nn.Linear(in_features=256, out_features=512)
        self.tanh2 = nn.Tanh()
        self.fc3 = nn.Linear(in_features=512, out_features=28*28)
        self.tanh3 = nn.Tanh()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh1(x)
        x = self.fc2(x)
        x = self.tanh2(x)
        x = self.fc3(x)
        x = self.tanh3(x)

        output = x.view(-1, 28, 28)

        return output

#  辨别器,输入（1，28，28）
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(in_features=28*28, out_features=512)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.relu2 = nn.LeakyReLU()
        self.fc3 = nn.Linear(in_features=256, out_features=1)
        self.sigmoid = nn.Sigmoid()
        

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)

        output = self.sigmoid(x)

        return output
