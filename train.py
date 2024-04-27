import os
import torch
import torch.nn as nn
import torchvision

from models.MLP import MLP
from models.LeNet5 import LeNet5
from models.ResNet import ResNet
from models.GRU import GRU
from models.ViT import ViT

#定义超参数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_epochs = 1
lr = 0.9
batch_size = 64

# 加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
# 定义数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 定义模型
# net = LeNet5()
# net = MLP()
# net = ResNet()
# net = GRU()
net = ViT()

net = net.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

# 训练函数
def train(images, lables):
        images = images.to(device)
        lables = lables.to(device)

        optimizer.zero_grad()

        outputs = net(images)
        loss = criterion(outputs, lables)
        loss.backward()
        optimizer.step()

        return loss

# 训练过程
print(device)
for epoch in range(num_epochs):
    for i, (images, lables) in enumerate(train_loader):

        loss = train(images, lables)
        
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
        fina_loss = loss.item()

# 保存模型
current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save') + os.sep
torch.save(net.state_dict(), current_dir + net._get_name() + '.pth')