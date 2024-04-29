import os
import torch
import matplotlib.pyplot as plt

from models._import import Generator

# 定义超参数
input_dim = 100

# 定义模型
G = Generator(input_dim)
G.eval()

# 加载模型参数
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save', 'Generator.pth')
G.load_state_dict(torch.load(model_dir))

# 测试模型
with torch.no_grad():
    x = torch.randn(1, input_dim)
    img = G(x)

    img = img.squeeze()
    img = img.reshape(28, 28).numpy()

plt.imshow(img, cmap='gray')
plt.show()
