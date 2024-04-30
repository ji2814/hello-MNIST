import torch  
import torch.nn as nn  
  
class Embedding(nn.Module):
    def __init__(self, embed_dim, num_class=10):
        super().__init__()

        self.embed = nn.Embedding(num_class, embed_dim)

    # x[batch_size] -> output[batch_size, embed_dim]
    def forward(self, x):
        output = self.embed(x.long())

        return output
    
# CGAN生成器，输入噪声和条件变量，输出（1，28，28）  
class Generator(nn.Module):  
    def __init__(self, input_dim, embed_dim=64):  
        super(Generator, self).__init__()  

        self.embed_dim = embed_dim
        self.embed = nn.Embedding(embed_dim, num_class=10)

        self.fc1 = nn.Linear(input_dim + embed_dim, 32 * 32)  # 拼接噪声和条件变量  
        self.br1 = nn.Sequential(  
            nn.BatchNorm1d(32 * 32),  
            nn.ReLU()  
        )  
        self.fc2 = nn.Linear(32 * 32, 128 * 7 * 7)  
        self.br2 = nn.Sequential(  
            nn.BatchNorm1d(128 * 7 * 7),  
            nn.ReLU()  
        )  
        self.conv1 = nn.Sequential(  
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  
            nn.BatchNorm2d(64),  
            nn.ReLU(),  
        )  
        self.conv2 = nn.Sequential(  
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),  
            nn.Tanh()  # 使用Tanh作为激活函数可能更适合图像生成  
        )  
    # x[batch_size,input_dim=100], cond[batch_size,]
    def forward(self, x, cond):  
        cond = self.embed(cond) # cond[batch_size,] -> cond[batch_size, embed_dim=64]

        x = torch.cat([x, cond], dim=1)
        x = self.br1(self.fc1(x))  
        x = self.br2(self.fc2(x))  
        x = x.reshape(-1, 128, 7, 7)  
        x = self.conv1(x)  
        output = self.conv2(x)  

        return output  
  
# CGAN辨别器
class Discriminator(nn.Module):  
    def __init__(self, embed_dim=64):  
        super(Discriminator, self).__init__()  
        
        self.embed = Embedding(embed_dim=embed_dim) # 嵌入层
        self.conv1 = nn.Sequential(  
            nn.Conv2d(embed_dim + 1, 32, 5, stride=1),  # cond和images在通道方向拼接  
            nn.LeakyReLU(0.2)  
        )  
        self.pl1 = nn.MaxPool2d(2, stride=2)  
        self.conv2 = nn.Sequential(  
            nn.Conv2d(32, 64, 5, stride=1),  
            nn.LeakyReLU(0.2)  
        )  
        self.pl2 = nn.MaxPool2d(2, stride=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 1024)  
        self.fc2 = nn.Sequential(  
            nn.Linear(1024 + embed_dim, 1),  # 拼接全连接层的输出和条件变量  
            nn.Sigmoid()  
        )  
  
  # x[batch_size, 1, 28, 28], cond[batch_size]
    def forward(self, x, cond):  
        cond = self.embed(cond) # cond[batch_size,] -> cond[batch_size, embed_dim=64]
        # 将条件变量复制并reshape以匹配图像的batch size和height、width  
        cond_embed = cond.view(cond.size(0), -1, 1, 1)  # cond_embed[batch_size, embed_dim, 1, 1]  
        cond_embed = cond_embed.repeat(1, 1, x.size(2), x.size(3))  # 复制以匹配图像尺寸

        # 将条件变量拼接到图像的通道上  
        x = torch.cat([x, cond_embed], 1)  
        x = self.conv1(x)  
        x = self.pl1(x)  
        x = self.conv2(x)  
        x = self.pl2(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)  

        x = torch.cat([x, cond.view(cond.size(0), -1)], dim=1)     
        output = self.fc2(x) 

        return output        
    