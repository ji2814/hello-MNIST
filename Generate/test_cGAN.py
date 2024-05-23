import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from models.cGAN import Generator

# 定义超参数
input_dim = 100
embed_dim = 10

# 定义模型
G = Generator(input_dim, embed_dim)
G.eval()

# 加载模型参数
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save', 'Generator_cGAN.pth')
G.load_state_dict(torch.load(model_dir)) 

'''测试过程'''
noise = torch.randn(100, 100)
labels = torch.LongTensor([i for i in range(10) for _ in range(10)])

images = G(noise, labels).unsqueeze(1)
grid = make_grid(images, nrow=10, normalize=True) # images[-1, 1] -> [0, 1]

# 展示图像
fig, ax = plt.subplots(figsize=(10,10))
ax.imshow(grid.permute(1, 2, 0).detach().cpu().numpy(), cmap='binary')
ax.axis('off')

plt.tight_layout() 
plt.show()



