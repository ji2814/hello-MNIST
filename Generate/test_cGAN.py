import os
import torch
import matplotlib.pyplot as plt

from models.cDCGAN import Generator

# 定义超参数
input_dim = 100

# 定义模型
G = Generator(input_dim)
G.eval()

# 加载模型参数
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save', 'Generator_cGAN.pth')
G.load_state_dict(torch.load(model_dir)) 
  
# 设置显示
_, axs = plt.subplots(nrows=10, ncols=10, figsize=(10, 10), subplot_kw={'xticks': [], 'yticks': []}) 

for label in range(10):
        # 生成随机噪声和条件变量  
        noise = torch.randn(10, input_dim)
        label_tensor = torch.tensor([label] * 10, dtype=torch.long)

        # 生成图像  
        with torch.no_grad():  
            imgs = G(noise, label_tensor)  
        imgs = imgs.view(10, 1, 28, 28).squeeze(1).cpu().numpy()

        # 显示图像  
        for column in range(10):  
            axs[label][column].imshow(imgs[column], cmap='gray')   
            axs[label][column].axis('off')  # 关闭坐标轴 

plt.tight_layout()  
plt.show()


