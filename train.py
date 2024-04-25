import os
import torch
import torch.nn as nn
import torchvision

from models.MLP import MLP
from models.LeNet5 import LeNet5
from models.ResNet import ResNet

# 加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
# 定义数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

#定义超参数
num_epochs = 1
lr = 0.9

# 定义模型、损失函数和优化器
# net = LeNet5()
# net = MLP()
net = ResNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

# 训练过程
fina_loss = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
        fina_loss = loss.item()

# 保存模型
current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save') + os.sep
torch.save(net.state_dict(), current_dir + net._get_name() + '.pth')