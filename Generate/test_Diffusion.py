import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.difusion.UNet import MyTinyUNet
from models.difusion.Diffusion import DDPM

# 定义超参数
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

learning_rate = 1e-3
num_epochs = 50
num_timesteps = 1000
network = MyTinyUNet()
network = network.to(device)
model = DDPM(network, num_timesteps, beta_start=0.0001, beta_end=0.02, device=device)

frames = []
frames_mid = []
model.eval()

