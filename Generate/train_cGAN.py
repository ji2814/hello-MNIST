import os
import torch
import torch.nn as nn
import torchvision

from models.cDCGAN import Generator, Discriminator

#定义超参数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

lr = 0.0002
num_epochs = 1
input_dim = 100
embed_dim = 64
batch_size = 64

# 加载MNIST数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=True)
# 定义数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

# 定义模型
G = Generator(input_dim, embed_dim=embed_dim).to(device)
D = Discriminator(embed_dim=embed_dim).to(device)

# 定义损失函数和优化器
criterion = nn.BCELoss()

optim_G = torch.optim.Adam(G.parameters(), lr=lr)
optim_D = torch.optim.Adam(D.parameters(), lr=lr)

# 训练cGAN函数
def train_cGAN(x, conditions):  
    '''训练cGAN'''  
    # 将真实数据和条件移动到适当的设备（如GPU）  
    real_x = x.to(device)  
    real_conditions = conditions.to(device)  
  
    # 清零判别器的梯度  
    optim_D.zero_grad()  
  
    # 计算判别器对真实数据的损失  
    real_output = D(real_x, real_conditions)  
    real_loss = criterion(real_output, torch.ones_like(real_output))  
  
    # 生成假数据  
    fake_conditions = real_conditions.detach()
    noise = torch.randn([batch_size, input_dim]).to(device)  
    fake_x = G(noise, fake_conditions)  
  
    # 计算判别器对假数据的损失  
    fake_output = D(fake_x.detach(), fake_conditions)  # .detach()是为了在反向传播时不更新生成器  
    fake_loss = criterion(fake_output, torch.zeros_like(fake_output))  
  
    # 判别器的总损失  
    loss_D = real_loss + fake_loss  
  
    # 反向传播并更新判别器的权重  
    loss_D.backward()  
    optim_D.step()  
  
    # 清零生成器的梯度  
    optim_G.zero_grad()  
  
    # 生成器希望判别器将假数据判断为真  
    fake_output = D(fake_x, fake_conditions)  # 这次不需要.detach()，因为我们要更新生成器  
    loss_G = criterion(fake_output, torch.ones_like(fake_output))  
  
    # 反向传播并更新生成器的权重  
    loss_G.backward()  
    optim_G.step()  
  
    return loss_D.item(), loss_G.item()  # 返回损失值而不是tensor

# 训练过程
print(device)
for epoch in range(num_epochs):
    loss_D, loss_G = 0, 0
    for i, (images, lables) in enumerate(train_loader):
        loss_D, loss_G = train_cGAN(images, lables)

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss_D: {:.4f}, Loss_G: {:.4f}'.format(epoch+1, 
                num_epochs, i+1, len(train_loader), loss_D, loss_G))

# 保存模型
current_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save') + os.sep
torch.save(G.state_dict(), current_dir + 'Generator_cGAN.pth')