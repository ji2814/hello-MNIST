import torch  
import torch.nn as nn

def sinusoidal_embedding(n, d):
    # 位置编码
    embedding = torch.tensor([[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)])
    sin_mask = torch.arange(0, n, 2)

    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])

    return embedding

class MyConv(nn.Module):
    # 卷积模块 
    def __init__(self, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):  
        super(MyConv, self).__init__()  

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)  
        self.ln = nn.LayerNorm([out_c]) if normalize else None  # 只有在normalize为True时才添加LayerNorm  
        self.activation = nn.SiLU() if activation is None else activation  
  
    def forward(self, x):  
        out = self.conv1(x)  
        if self.ln is not None: 
            out = out.permute(0, 2, 3, 1) # (bs, c, h, w)->(bs, h, w, c)
            out = self.ln(out) 
            out = out.permute(0, 3, 2, 1) # 变换回来
        if self.activation is not None:  
            out = self.activation(out)  

        return out

def MyTinyBlock(in_c, out_c, size=None, normalize=True):    
    return nn.Sequential(  
        MyConv(in_c, out_c, normalize=normalize),  # 第一个卷积层  
        MyConv(out_c, out_c, normalize=normalize),  # 第二个卷积层  
        MyConv(out_c, out_c, normalize=normalize)   # 第三个卷积层  
    )  

def MyTinyUp(in_c, normalize=True):
    # 上采样
    return nn.Sequential(  
        MyConv(in_c, in_c//2, normalize=normalize),  # 第一个卷积层，通道数减半  
        MyConv(in_c//2, in_c//4, normalize=normalize),  # 第二个卷积层，通道数再减半  
        MyConv(in_c//4, in_c//4, normalize=normalize)  # 第三个卷积层，保持通道数不变  
    )  

class MyTinyUNet(nn.Module):
  # UNet网络
  # 三层上采样和三层下采样部分
    def __init__(self, in_c=1, out_c=1, size=32, n_steps=1000, time_emb_dim=100):
        super(MyTinyUNet, self).__init__()

        # 时间序列位置编码
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # 左半部分
        self.te1 = self.EmbedFC(time_emb_dim, 1)
        self.b1 = MyTinyBlock(in_c=in_c, out_c=10)
        self.down1 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=4, stride=2, padding=1)
        self.te2 = self.EmbedFC(time_emb_dim, 10)
        self.b2 = MyTinyBlock(in_c=10, out_c=20)
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)
        self.te3 = self.EmbedFC(time_emb_dim, 20)
        self.b3 = MyTinyBlock(20, 40)
        self.down3 = nn.Conv2d(40, 40, 4, 2, 1)

        # 底层
        self.te_mid = self.EmbedFC(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyConv(40, 20),
            MyConv(20, 20),
            MyConv(20, 40)
        )

        # 右半部分
        self.up1 = nn.ConvTranspose2d(40, 40, 4, 2, 1)
        self.te4 = self.EmbedFC(time_emb_dim, 80)
        self.b4 = MyTinyUp(80)
        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self.EmbedFC(time_emb_dim, 40)
        self.b5 = MyTinyUp(40)
        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self.EmbedFC(time_emb_dim, 20)
        self.b_out = MyTinyBlock(20, 10)
        self.conv_out = nn.Conv2d(10, out_c, 3, 1, 1)

    def forward(self, x, t): # x(bs, channal, h, w), t(bs)
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (bs, 10, h/2, w/2)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (bs, 20, h/4, w/4)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (bs, 40, h/8, w/8)

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (bs, 40, h/8, w/8)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (bs, 80, h/8, w/8)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (bs, 20, h/8, w/8)
        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (bs, 40, h/4, w/4)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (bs, 10, h/2, w/2)
        out = torch.cat((out1, self.up3(out5)), dim=1)  # (bs, 20, h, w)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (bs, 10, h, w)
        out = self.conv_out(out) # (bs, out_c, h, w)

        return out
    
    def EmbedFC(self, dim_in, dim_out):
        # 嵌入层
        return nn.Sequential(nn.Linear(dim_in, dim_out), nn.SiLU(), nn.Linear(dim_out, dim_out))

"""
bs = 3
x = torch.randn(bs, 1, 32, 32)
n_steps=1000
timesteps = torch.randint(0, n_steps, (bs,)).long()
unet = MyTinyUNet(in_c =1, out_c =1, size=32)
y = unet(x, timesteps)
"""
