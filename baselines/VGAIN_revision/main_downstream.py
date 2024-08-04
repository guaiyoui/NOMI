import os
import glob
import time
import random
import argparse
from traceback import print_tb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from data_loader import data_loader
from utils import normalization, renormalization, sample_batch_index, rmse_loss, uniform_sampler, binary_sampler, rounding, construct_dataset, MAE, RMSE
from model import Generator, Discriminator, loss_computation
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

# Training settings

def parse_args():
    parser = argparse.ArgumentParser(description='Dual enhanced dbnn for missing data imputation')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0010, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--data_name', choices=['wine', 'letter','spam', 'heart', 'breast', 'phishing', 'wireless', 'turkiye', 'credit', 'connect', 'car', 'chess', 'news', 'shuttle', 'poker', 'abalone', 'yeast', 'poker', 'retail', 'wisdm', 'higgs'], default='wine', type=str)
    parser.add_argument('--miss_rate', help='missing data probability', default=0.2, type=float)
    parser.add_argument('--batch_size', help='the number of samples in mini-batch', default=128, type=int)
    parser.add_argument('--hint_rate', help='hint probability', default=0.9, type=float)
    parser.add_argument('--alpha_mse', help='mse proportion in the final loss', default=100, type=float)
    parser.add_argument('--gpuid', choices=['0','1'], default='0', type=str)

    return parser.parse_args()



def main(args):
    
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ori_data_x, data_x, data_m = data_loader(args.data_name, args.miss_rate)

    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    no, dim = data_x.shape
    
    # Hidden state dimensions
    h_dim = int(dim)  
    
    # Normalization

    
    generator = Generator(2*dim, h_dim, dim).to(device)
    discriminator = Discriminator(2*dim, h_dim, dim).to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print('\n##### Start training...')
    
    start = time.time()

    for epoch in tqdm(range(args.epochs)):
        generator.train()
        optimizer_G.zero_grad()

        batch_size = args.batch_size
        batch_idx = sample_batch_index(no, batch_size)

        X_mb = norm_data_x[batch_idx, :]  
        M_mb = data_m[batch_idx, :]  
        Z_mb = uniform_sampler(0, 0.01, batch_size, dim)
        X_mb = M_mb * X_mb + (1-M_mb) * Z_mb

        H_mb_temp = binary_sampler(args.hint_rate, batch_size, dim)
        H_mb = M_mb * H_mb_temp

        G_sample, mu, logvar = generator.forward(torch.Tensor(X_mb).to(device), torch.Tensor(M_mb).to(device))
        Hat_X = X_mb * M_mb + G_sample.cpu().detach().numpy() * (1-M_mb)
        Mask_prob = discriminator.forward(torch.Tensor(Hat_X).to(device), torch.Tensor(H_mb).to(device))
        
        loss = loss_computation(torch.Tensor(M_mb).to(device), torch.Tensor(X_mb).to(device), G_sample, Mask_prob, args.alpha_mse, mu, logvar)
        
        loss.backward(retain_graph=True)
        optimizer_G.step()
        optimizer_D.step()

    end = time.time() 
    print('Train Time taken: %0.6f' % (end - start))

    start = time.time()
    generator.eval()
    a = [i for i in range(no)]
    batch_idx_ = np.array_split(a, args.batch_size)
    imputed_data_all = torch.Tensor([])
    for batch_idx in batch_idx_:
        X_mb = norm_data_x[batch_idx, :]  
        M_mb = data_m[batch_idx, :]  
        Z_mb = uniform_sampler(0, 0.01, len(batch_idx), dim)
        X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
    
        imputed_data, mu, logvar = generator.forward(torch.Tensor(X_mb).to(device), torch.Tensor(M_mb).to(device))
        imputed_data = X_mb * M_mb + (1-M_mb) * imputed_data.cpu().detach().numpy()
        imputed_data_all = torch.cat((imputed_data_all, torch.Tensor(imputed_data)),dim=0)


    # Renormalization
    imputed_data = renormalization(imputed_data_all.cpu().detach().numpy(), norm_parameters)  
    
    end = time.time()
    print('Test Time taken: %0.6f' % (end - start))
    np.savetxt( "./data_imputed/"+args.data_name+"_imputed.csv", imputed_data, delimiter="," )
    # Rounding
    imputed_data = rounding(imputed_data, data_x)  

    print('\n##### RMSE Performance: ' + str(np.round(RMSE(imputed_data, ori_data_x, data_m), 4)))
    print('\n##### MAE Performance: ' + str(np.round(MAE(imputed_data, ori_data_x, data_m), 4)))
    norm_data_x, norm_parameters = normalization(imputed_data)
    # plt.plot(epoch_idx, train_loss)
    # plt.plot(epoch_idx, test_loss, color='red')
    # plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)

