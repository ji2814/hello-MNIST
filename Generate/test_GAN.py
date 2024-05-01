import os
import torch
import matplotlib.pyplot as plt

from models._importGAN import Generator

# 定义超参数
input_dim = 100
batch_size = 64

# 定义模型
G = Generator(input_dim)
G.eval()

# 加载模型参数
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save', 'Generator_GAN.pth')
G.load_state_dict(torch.load(model_dir))

# 测试GAN模型
def test_GAN(G, input_dim, batch_size=100):  
    with torch.no_grad():  
        noise = torch.randn(batch_size, input_dim)  # 一次性生成batch_size个噪声向量  
        imgs = G(noise)  # 生成图片  
        
        # 假设imgs的形状是 [batch_size, channels, height, width]  
        # 对于MNIST，channels通常是1（灰度图像）  
        # 我们需要调整图像的形状以便matplotlib可以显示它们  
        imgs = imgs.view(batch_size, 1, 28, 28).squeeze(1).cpu().numpy()  # 假设imgs是[batch_size, 1, 28, 28]  
          
        # 使用matplotlib的subplot_kw参数来避免重叠的图像标签  
        fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(10, 10), subplot_kw={'xticks': [], 'yticks': []})  
  
        for i, ax in enumerate(axs.flat):  
            if i < batch_size:  # 确保不会超出实际的图片数量  
                ax.imshow(imgs[i], cmap='gray')  
          
        plt.tight_layout()  
        plt.show()  

# 调用函数来生成并显示图像  
test_GAN(G, input_dim)

