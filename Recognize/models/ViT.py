import torch  
import torch.nn as nn  
  
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

class Encoder(nn.Module):

    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, channels):  
        super().__init__()  

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'  
        num_patches = (image_size // patch_size) ** 2  
        patch_dim = channels * patch_size ** 2  
  
        self.to_patch_embedding = nn.Sequential(  
            nn.Conv2d(in_channels=channels, out_channels=dim, kernel_size=patch_size, stride=patch_size),  
            nn.Flatten(2)
        )  
  
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  
  
        self.transformer = nn.Sequential(*[TransformerBlock(dim, heads, mlp_dim) for _ in range(depth)])  
  
        self.to_cls_token = nn.Identity()  
  

    def forward(self, x):

        x = self.to_patch_embedding(x) 
        x = x.transpose(1, 2)   
        b, n, _ = x.shape  
        cls_tokens = self.cls_token.expand(b, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)  
        x += self.pos_embedding  

        x = self.transformer(x)  
        output = self.to_cls_token(x[:, 0])

        return output

class Decoder(nn.Module):
    def __init__(self, dim, num_classes):
        super().__init__()

        self.mlp_head = nn.Sequential(
            nn.Linear(in_features=dim, out_features=num_classes)
        )

    def forward(self, x):
        output = self.mlp_head(x)

        return output

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