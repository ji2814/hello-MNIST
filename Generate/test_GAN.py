import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from models._importGAN import Generator

# 定义超参数
input_dim = 100
batch_size = 100

# 定义模型
G = Generator(input_dim)
G.eval()

# 加载模型参数
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save', 'Generator_GAN.pth')
G.load_state_dict(torch.load(model_dir))

'''测试过程'''
noise = torch.randn(batch_size, input_dim)

images = G(noise).unsqueeze(1)
grid = make_grid(images, nrow=10, normalize=True)

# 展示图像
fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(grid.permute(1, 2, 0).detach().cpu().numpy(), cmap='binary')
ax.axis('off')

plt.tight_layout() 
plt.show()