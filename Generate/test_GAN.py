import os
import torch
import matplotlib.pyplot as plt

from models.GAN import Generator

# 定义超参数
input_dim = 28

# 定义模型
G = Generator()

# 加载模型参数
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save', 'Generator.pth')
G.load_state_dict(torch.load(model_dir))

# 测试模型
x = torch.randn(64, input_dim)
img = G(x)
img = img.reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.show()
