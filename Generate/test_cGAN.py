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

# 测试cGAN模型
def test_cGAN():
    pass
    with torch.no_grad():
        noise = torch.randn(1, input_dim)
        
        cond = torch.randint(0, 10, (batch_size,))
        print(cond)
        img = G(noise, cond)

        img = img.squeeze()
        img = img.reshape(28, 28).numpy()

    plt.imshow(img, cmap='gray')
    plt.show()


# 调用函数来生成并显示图像  
test_cGAN(G, input_dim)


