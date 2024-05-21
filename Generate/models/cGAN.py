import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embedding, embedding_dim):
        super().__init__()

        self.embed = nn.Embedding(num_embeddings=num_embedding,
                                  embedding_dim=embedding_dim)

    def forward(self, x):
        output = self.embed(x)

        return output


class Discriminator(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.embed = Embedding(num_embedding=10,
                                embedding_dim=self.embed_dim)
        
        self.model = nn.Sequential(
            nn.Linear(in_features=28 * 28 + embed_dim, out_features=1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, cond):
        x = x.view(x.size(0), -1) # x.size() = batch_size
        c = self.embed(cond)
        x = torch.cat([x, c], dim=1)
        out = self.model(x)
        return out.squeeze()
    

class Generator(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.embed = Embedding(num_embedding=10, 
                                   embedding_dim=self.embed_dim)
        
        self.model = nn.Sequential(
            nn.Linear(in_features=self.input_dim + self.embed_dim,
                       out_features=256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, x, cond):
        x = x.view(x.size(0), self.input_dim)
        c = self.embed(cond)
        x = torch.cat([x, c], dim=1)

        output = self.model(x)
        
        return output.view(x.size(0), 28, 28)