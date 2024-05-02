import os
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from models.cGAN import Generator

# 定义超参数
input_dim = 100
embed_dim = 64

# 定义模型
G = Generator(input_dim, embed_dim)
G.eval()

# 加载模型参数
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save', 'Generator_cGAN.pth')
G.load_state_dict(torch.load(model_dir)) 

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

'''
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
'''

