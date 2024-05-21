import os, sys
import torch
import torch.nn as nn
import torchvision

from models.difusion.UNet import MyTinyUNet
from models.difusion.Diffusion import DDPM

#定义超参数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

lr = 1e-3
num_epochs = 50
num_timesteps = 1000

batch_size = 64

# 加载MNIST数据集
transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(32),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5), (0.5))
    ])
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# 定义数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)

# 定义模型
network = MyTinyUNet()
network = network.to(device)
model = DDPM(network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device)

# 定义损失函数和优化器
criterion = nn.MSELoss()

optimizer = torch.optim.Adam(network.parameters(), lr=lr)

# 训练过程
print(device)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        noise = torch.randn(images.shape).to(device)
        timesteps = torch.randint(0, num_timesteps, (images.shape[0],)).long().to(device)

        noisy = model.add_noise(images, noise, timesteps)
        noise_pred = model.reverse(noisy, timesteps)

        loss = criterion(noise_pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 打印loss
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, 
                    num_epochs, i+1, len(train_loader), loss.detach().item()))
            
# 保存模型
current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save') + os.sep
torch.save(network.state_dict(), current_dir + 'unet.pth')