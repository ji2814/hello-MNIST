import os
import torch
import matplotlib.pyplot as plt

from models._importGAN import Generator

# 定义超参数
input_dim = 100
batch_size = 10

# 定义模型
G = Generator(input_dim)
G.eval()

# 加载模型参数
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save', 'Generator_GAN.pth')
G.load_state_dict(torch.load(model_dir))

'''测试GAN模型'''
# 设置显示
_, axs = plt.subplots(nrows=10, ncols=10, figsize=(10, 10), subplot_kw={'xticks': [], 'yticks': []}) 

with torch.no_grad():  
    for row in range(10):
        noise = torch.randn(10, input_dim)  # 一次性生成10个噪声向量  
        imgs = G(noise)  # 生成图片  
        # imgs[batch_size, channels, height, width]  
        imgs = imgs.view(10, 1, 28, 28).squeeze(1).cpu().numpy() 

        for column in range(10):  
            axs[row][column].imshow(imgs[column], cmap='gray')  

plt.tight_layout()  
plt.show()  
