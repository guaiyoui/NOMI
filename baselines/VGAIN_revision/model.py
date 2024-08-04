from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import xavier_init

class Generator(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Generator, self).__init__()

        self.Encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels*2),
            nn.Sigmoid()
        )

        self.Decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
            nn.Sigmoid()
        )

        self.fc_m = nn.Linear(hidden_channels, hidden_channels)
        self.fc_sigma = nn.Linear(hidden_channels, hidden_channels)
  
    def reparameterise(self, mu, logvar):
        epsilon = torch.randn_like(mu)
        return mu + epsilon * torch.exp(logvar / 2)

# based on implementation of GAIN and material "https://adaning.github.io/posts/9047.html"
    def forward(self, x, m):
        inputs = torch.cat([x, m], axis = 1)
        
        code = self.Encoder(inputs)

        mu, logvar = code.chunk(2, dim=1)
        z = self.reparameterise(mu, logvar)
        recon_x = self.Decoder(z)

        return recon_x, mu, logvar
        


class Discriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Discriminator, self).__init__()

        self.w_1 = nn.Linear(in_channels, hidden_channels)
        self.w_2 = nn.Linear(hidden_channels, hidden_channels)
        self.w_3 = nn.Linear(hidden_channels, out_channels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x, h):
        inputs = torch.cat([x, h], axis = 1)
        D_h1 = self.relu(self.w_1(inputs))
        D_h2 = self.relu(self.w_2(D_h1))
        Mask_prob = self.sigmoid(self.w_3(D_h2))

        return Mask_prob
    



def loss_computation(M, X, G_sample, Mask_prob, alpha, mu, logvar):
    # mse loss
    loss_mse = torch.mean((M * X - M * G_sample)**2) / torch.mean(M)*1000

    # discriminator loss
    loss_discriminator = -torch.mean(M * torch.log(Mask_prob + 1e-8) + (1-M) * torch.log(1. - Mask_prob + 1e-8))

    # KL loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # reconstruction loss
    recon_loss = F.binary_cross_entropy(M*G_sample, M*X)

    return alpha*loss_mse  + loss_discriminator + 0.1*(kl_loss + recon_loss)
    
