import torch
import torch.nn as nn

# Transformer基本块
class TransformerBlock(nn.Module):  
    def __init__(self, dim, heads, mlp_dim):  
        super().__init__()  

        self.norm1 = nn.LayerNorm(dim)  
        self.attn = nn.MultiheadAttention(dim, heads)  
        self.norm2 = nn.LayerNorm(dim)  
        self.mlp = nn.Sequential(  
            nn.Linear(dim, mlp_dim),  
            nn.GELU(),  
            nn.Linear(mlp_dim, dim), 
        )  

    def forward(self, x, mask=None):  
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), attn_mask=mask)[0]  
        x = x + self.mlp(self.norm2(x))  
        
        return x  


# 编码器
class Encoder(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels):  
        super().__init__()  

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'  
        num_patches = (image_size // patch_size) ** 2  
        patch_dim = channels * patch_size ** 2  
  		
		# 使用CNN来嵌入
        self.to_patch_embedding = nn.Sequential(  
            nn.Conv2d(in_channels=channels, out_channels=dim, kernel_size=patch_size, stride=patch_size),  
            nn.Flatten(2)
        ) 
 		
		#添加分类token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) 
         
		# 位置编码层
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  

		# depth层基本块
        self.transformer = nn.Sequential(*[TransformerBlock(dim, heads, mlp_dim) for _ in range(depth)])  
  
        self.to_cls_token = nn.Identity()  
  
    def forward(self, x):
        x = self.to_patch_embedding(x) 
        x = x.transpose(1, 2) # 交换维度，以便与匹配Transformer维度
        b, n, _ = x.shape  
        cls_tokens = self.cls_token.expand(b, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1) # 将token与原始输入拼接
        x += self.pos_embedding # 添加位置编码

        x = self.transformer(x)  # 通过多层transformer基本块
        output = self.to_cls_token(x[:, 0]) # 取最后一个token

        return output


# 解码器结构
class Decoder(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()

        self.mlp_head = nn.Sequential(
            nn.Linear(in_features=dim, out_features=num_classes)
        )

    def forward(self, x):
        output = self.mlp_head(x)

        return output


# ViT模型
class ViT(nn.Module):  
    def __init__(self, image_size=28, patch_size=4, num_classes=10, 
                    dim=64, depth=4, heads=8, mlp_dim=128, channels=1):  
        super().__init__()  
  
        self.encoder = Encoder(image_size, patch_size, dim, depth, heads, mlp_dim, channels)
        self.decoder = Decoder(dim, num_classes)

    def forward(self, img):
        x = self.encoder(img)
        output = self.decoder(x)

        return output