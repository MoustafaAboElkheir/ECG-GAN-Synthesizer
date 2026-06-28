import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=100, seq_len=3000):
        super(Generator, self).__init__()
        self.seq_len = seq_len
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, seq_len),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img.view(img.size(0), 1, self.seq_len)

class Discriminator(nn.Module):
    def __init__(self, seq_len=3000):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(seq_len, 512),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity
