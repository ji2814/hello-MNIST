import os
import torch
import torch.nn as nn
import torchvision

from models._importGAN import Generator, Discriminator

#定义超参数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

lr = 0.0002
num_epochs = 10
input_dim = 100
embed_dim = 64
batch_size = 64

# 加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
# 定义数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last = True)

# 定义模型
G = Generator(input_dim).to(device)
D = Discriminator().to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()

optim_G = torch.optim.Adam(G.parameters(), lr=lr)
optim_D = torch.optim.Adam(D.parameters(), lr=lr)

# 训练GAN函数
def train_GAN(x):
    '''判别器'''
    real_x = x.to(device)

    optim_D.zero_grad()
    
    real_output = D(real_x)
    real_loss = criterion(real_output, torch.ones_like(real_output).to(device))

    fake_x = G(torch.randn([batch_size, input_dim]).to(device)).detach()
    fake_output = D(fake_x)
    fake_loss = criterion(fake_output, torch.zeros_like(fake_output).to(device))

    loss_D = real_loss + fake_loss

    loss_D.backward()
    optim_D.step()

    '''生成器'''
    optim_G.zero_grad()

    fake_x = G(torch.randn([batch_size, input_dim]).to(device))
    fake_output = D(fake_x)
    loss_G = criterion(fake_output, torch.ones_like(fake_output).to(device))

    loss_G.backward()
    optim_G.step()

    return loss_D, loss_G

# 训练过程
print(device)
for epoch in range(num_epochs):
    loss_D, loss_G = 0, 0
    for i, (images, lables) in enumerate(train_loader):
        loss_D, loss_G = train_GAN(images)

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss_D: {:.4f}, Loss_G: {:.4f}'.format(epoch+1, 
                num_epochs, i+1, len(train_loader), loss_D, loss_G))

# 保存模型
current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save') + os.sep
torch.save(G.state_dict(), current_dir + 'Generator_GAN.pth')