from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import xavier_init
from layer import GraphAttentionLayer

class Generator(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout, nheads, alpha, k_neighbors, retrieval):
        super(Generator, self).__init__()
        self.retrieval = retrieval
        if self.retrieval == 'none':
            self.w_1 = nn.Linear(in_channels*2, hidden_channels)
        else:
            self.w_1 = nn.Linear(in_channels*3, hidden_channels)
        self.w_2 = nn.Linear(hidden_channels, hidden_channels)
        self.w_3 = nn.Linear(hidden_channels, hidden_channels)
        
        self.w_4 = nn.Linear(in_channels*2, hidden_channels)
        self.w_5 = nn.Linear(hidden_channels, hidden_channels)
        self.w_6 = nn.Linear(hidden_channels, hidden_channels)

        self.Encoder = nn.Sequential(
            nn.Linear(in_channels*2, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Sigmoid()
        )

        self.Decoder = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc_m = nn.Linear(hidden_channels, hidden_channels)
        self.fc_sigma = nn.Linear(hidden_channels, hidden_channels)

        self.dropout = dropout
        self.k_neighbors = k_neighbors
        self.attentions = [GraphAttentionLayer(in_channels, hidden_channels, dropout=dropout, alpha=alpha, 
                                                k_neighbors = k_neighbors, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att1 = GraphAttentionLayer(hidden_channels * nheads, out_channels, dropout=dropout, alpha=alpha, k_neighbors = k_neighbors, concat=False)

        
    def forward(self, x, m):
        inputs = torch.cat([x, m], axis = 1)
        
        # G_h1 = self.relu(self.w_1(inputs))
        # G_h2 = self.relu(self.w_2(G_h1))
        # G_h2 = self.sigmoid(self.w_3(G_h2))
        code = self.Encoder(inputs)
        m = self.fc_m(code)
        sigma = self.fc_sigma(code)
        e = torch.randn_like(sigma)
        c = torch.exp(sigma) * e + m
        G_h2 = self.Decoder(c)
        return G_h2

    def forward_mask(self, x, h):
        inputs = torch.cat([x, h], axis = 1)
        D_h1 = self.relu(self.w_4(inputs))
        D_h2 = self.relu(self.w_5(D_h1))
        Mask_prob = self.sigmoid(self.w_6(D_h2))

        return Mask_prob

    
    def loss_mse(self, M, X, G_sample):
        MSE_loss = torch.mean((M * X - M * G_sample)**2) / torch.mean(M)*1000
        return MSE_loss
    
    def loss_discriminator(self, M, Mask_prob):
        return -torch.mean(M * torch.log(Mask_prob + 1e-8) + (1-M) * torch.log(1. - Mask_prob + 1e-8)) 

    def loss(self, M, X, G_sample, Mask_prob, alpha):
        # G_binary = self.forward(x, m, permutate_X_mb, permutate_M_mb)
        loss_mse = self.loss_mse(M, X, G_sample)
        loss_discriminator = self.loss_discriminator(M, Mask_prob)
        
        return alpha*loss_mse  + loss_discriminator
