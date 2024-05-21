import os
import torch
import matplotlib.pyplot as plt

from models.difusion.UNet import MyTinyUNet
from models.difusion.Diffusion import DDPM

# 定义超参数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

learning_rate = 1e-3
num_epochs = 50
num_timesteps = 1000

# 定义模型
network = MyTinyUNet()
network = network.to(device)
ddpm = DDPM(network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device)

# 加载模型参数
network.eval()
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save', 'unet.pth')
network.load_state_dict(torch.load(model_dir)) 
 
# 生成图像
frames = []  
frames_mid = [] 

ddpm.eval()  
with torch.no_grad():  
    timesteps = list(range(ddpm.num_timesteps))[::-1]  
    sample = torch.randn(100, 1, 32, 32).to(device)  
        
    for t in timesteps:  
        time_tensor = (torch.ones(100,1) * t).long().to(device)  
        residual = ddpm.reverse(sample, time_tensor)  
        sample = ddpm.step(residual, time_tensor[0], sample)  

        if t == ddpm.num_timesteps // 2:  # 在中间的时间步保存图像  
            for i in range(100):  
                frames_mid.append(sample[i].detach().cpu())  

    for i in range(100):  
        frames.append(sample[i].detach().cpu())  

# 画出图像
images = [im.permute(1,2,0).numpy() for im in frames]

fig = plt.figure(figsize=(8, 8))
rows = int(len(images) ** (1 / 2))
cols = round(len(images) / rows)

idx = 0
for r in range(rows):
    for c in range(cols):
        fig.add_subplot(rows, cols, idx + 1)

        if idx < len(images):
            plt.imshow(images[idx], cmap="gray")
            plt.axis('off')
            idx += 1

plt.show()