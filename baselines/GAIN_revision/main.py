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
from knn import knn_search, knn_search_matrix, knn_search_sklearn
from model import Generator
from tqdm import tqdm
from skimage import data,io
import matplotlib.pyplot as plt
import time

# Training settings
print("I am here")
def parse_args():
    parser = argparse.ArgumentParser(description='Dual enhanced dbnn for missing data imputation')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=17, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0010, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=4, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=500, help='Patience')
    parser.add_argument('--data_name', choices=['wine', 'letter','spam', 'mnist', 'heart', 'breast', 'phishing', 'wireless', 'shuttle', 'turkiye', 'credit', 'connect', 'chess', 'poker', 'abalone', 'yeast', 'car', 'retail', 'wisdm', 'higgs'], default='wine', type=str)
    parser.add_argument('--retrieval', choices=['none','knn', 'attention', 'hybrid'], default='none', type=str)
    parser.add_argument('--miss_rate', help='missing data probability', default=0.2, type=float)
    parser.add_argument('--batch_size', help='the number of samples in mini-batch', default=128, type=int)
    parser.add_argument('--hint_rate', help='hint probability', default=0.9, type=float)
    parser.add_argument('--alpha_mse', help='mse proportion in the final loss', default=100, type=float)
    parser.add_argument('--k_neighbors', help='how many neighbors to save', type=int, default=5)
    parser.add_argument('--gpuid', choices=['0','1'], default='0', type=str)
    parser.add_argument('--feature_dim', type=float, default=1.0, help='the rate of feature dimension ')
    parser.add_argument('--training_size', type=float, default=1.0, help='the rate of training size ')

    return parser.parse_args()



def main(args):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    ori_data_x, data_x, data_m = data_loader(args.data_name, args.miss_rate, args)

    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    # print('\n##### KNN Search...')
    # start = time.time()
    # knn, similarity = knn_search_sklearn(norm_data_x, data_m, args.k_neighbors)
    # print(knn.shape, similarity.shape)
    # end = time.time()
    # print('KNN Search Time used: ', end-start) 

    # data_m = 1-np.isnan(data_x)
    epoch_idx = []
    rmse_loss_all = []
    mae_loss_all = []
    # Other parameters
    no, dim = data_x.shape
    
    # Hidden state dimensions
    h_dim = int(dim)  
    
    # Normalization

    
    generator = Generator(dim, int(dim), dim, args.dropout, args.nb_heads, args.alpha, args.k_neighbors, args.retrieval).to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    min_rmse = 10000
    print('\n##### Start training...')
    
    start = time.time()
    permutate_x, permutate_m, label = construct_dataset(norm_data_x, data_m)

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
    

        G_sample = generator.forward(torch.Tensor(X_mb).to(device), torch.Tensor(M_mb).to(device))
        Hat_X = X_mb * M_mb + G_sample.cpu().detach().numpy() * (1-M_mb)

        Mask_prob = generator.forward_mask(torch.Tensor(Hat_X).to(device), torch.Tensor(H_mb).to(device))
        g_loss = generator.loss(M = torch.Tensor(M_mb).to(device), 
                                X = torch.Tensor(X_mb).to(device), 
                                G_sample = G_sample, 
                                Mask_prob = Mask_prob, 
                                alpha = args.alpha_mse)
                                
        g_loss.backward(retain_graph=True)
        optimizer_G.step()

    end = time.time() 
    print('Train Time taken: %0.6f' % (end - start))

    start = time.time()
    generator.eval()
    a = [i for i in range(no)]
    batch_idx_ = np.array_split(a,args.batch_size)
    imputed_data_all = torch.Tensor([])
    for batch_idx in batch_idx_:
        X_mb = norm_data_x[batch_idx, :]  
        M_mb = data_m[batch_idx, :]  
        Z_mb = uniform_sampler(0, 0.01, len(batch_idx), dim)
        X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
        H_mb_temp = binary_sampler(args.hint_rate, len(batch_idx), dim)
        H_mb = M_mb * H_mb_temp
    
        imputed_data = generator.forward(torch.Tensor(X_mb).to(device), torch.Tensor(M_mb).to(device))
        imputed_data = X_mb * M_mb + (1-M_mb) * imputed_data.cpu().detach().numpy()
        imputed_data_all = torch.cat((imputed_data_all, torch.Tensor(imputed_data)),dim=0)


    # Renormalization
    imputed_data = renormalization(imputed_data_all.cpu().detach().numpy(), norm_parameters)  
    
    end = time.time()
    print('Test Time taken: %0.6f' % (end - start))

    # Rounding
    imputed_data = rounding(imputed_data, data_x)  

    rmse = rmse_loss(ori_data_x, imputed_data, data_m)

    print('\n##### RMSE Performance: ' + str(np.round(rmse, 4)))
    print('\n##### Minimal RMSE Performance: ' + str(min_rmse))
    print('\n##### RMSE Performance: ' + str(np.round(RMSE(imputed_data, ori_data_x, data_m), 4)))
    print('\n##### MAE Performance: ' + str(np.round(MAE(imputed_data, ori_data_x, data_m), 4)))
    norm_data_x, norm_parameters = normalization(imputed_data)
    # plt.plot(epoch_idx, train_loss)
    # plt.plot(epoch_idx, test_loss, color='red')
    # plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)

