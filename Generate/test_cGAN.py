import os
import torch
import matplotlib.pyplot as plt

from models.cDCGAN import Generator

# 定义超参数
input_dim = 100
batch_size = 64

# 定义模型
G = Generator(input_dim)
G.eval()

# 加载模型参数
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save', 'Generator_cGAN.pth')
G.load_state_dict(torch.load(model_dir)) 
  
# 创建一个空的图像网格来保存所有生成的图像  
images = []  
  
# 生成随机噪声和条件变量  
batch_size = 16  
noise = torch.randn(batch_size, 100)
cond = torch.randint(0, 10, (batch_size,)) # 随机选择0到9之间的数字  
  
# 生成图像  
with torch.no_grad():  
    images = G(noise, cond)  
  
# 因为输出是在[-1, 1]范围内的，我们可以直接将其转换为matplotlib可显示的格式  
images = images.cpu().numpy() * 0.5 + 0.5  # 转换到[0, 1]范围  
images = images.squeeze(1)  # 移除通道维度（如果有的话）  

# 绘制图像  
fig, axes = plt.subplots(4, 4, figsize=(8, 8), subplot_kw={'xticks': [], 'yticks': []})  
for i, ax in enumerate(axes.flat):  
    ax.imshow(images[i], cmap='gray')  
plt.show()


