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
    parser.add_argument('--data_name', choices=['wine', 'letter','spam', 'mnist', 'heart', 'breast', 'car', 'abalone', 'chess', 'shuttle', 'phishing', 'wireless', 'turkiye', 'credit', 'connect', 'poker', 'retail', 'wisdm', 'higgs'], default='wine', type=str)
    parser.add_argument('--miss_rate', help='missing data probability', default=0.2, type=float)
    parser.add_argument('--batch_size', help='the number of samples in mini-batch', default=128, type=int)
    parser.add_argument('--hint_rate', help='hint probability', default=0.9, type=float)
    parser.add_argument('--alpha_mse', help='mse proportion in the final loss', default=100, type=float)
    parser.add_argument('--gpuid', choices=['0','1'], default='0', type=str)

    return parser.parse_args()


def to_onehot(ori_data_x, data_x, data_m, category_idx):
    category_len_map = {}
    category_data_map = {}
    category_data_inv_map = {}
    new_data_x = []
    new_data_m = []
    for idx in category_idx:
        unique_data = np.unique(ori_data_x[:, idx])
        print(unique_data)
        data_map = {}
        data_inv_map = {}
        for i in range(unique_data.shape[0]):
            data_map[unique_data[i]] = i
            data_inv_map[i] = unique_data[i]
        category_data_map[idx] = data_map
        category_data_inv_map[idx] = data_inv_map
        category_len_map[idx] = unique_data.shape[0]
        temp_data_x = np.zeros((ori_data_x.shape[0], unique_data.shape[0]))
        temp_data_m = np.ones((ori_data_x.shape[0], unique_data.shape[0]))
        for i in range(ori_data_x.shape[1]):
            if data_m[i][idx] == 0:
                for j in range(unique_data.shape[0]):
                    temp_data_m[i][j] = 0
            else:
                temp_data_x[i][data_map[ori_data_x[i][idx]]] = 1
        new_data_x.append(temp_data_x)
        new_data_m.append(temp_data_m)

    new_data_x = np.concatenate((np.delete(data_x, category_idx, axis=1), np.concatenate(new_data_x, axis=1)), axis=1)
    new_data_m = np.concatenate((np.delete(data_m, category_idx, axis=1), np.concatenate(new_data_m, axis=1)), axis=1)

    return new_data_x, new_data_m, category_len_map, category_data_inv_map

def back_from_onehot(new_data_x, data_m, category_idx, ori_data_x, category_len_map, category_data_inv_map):
    not_category_idx = [i for i in range(ori_data_x.shape[1]) if i not in category_idx]
    # print(not_category_idx)

    new_predicted_data = []
    for idx in category_idx:
        category_len = category_len_map[idx]
        category_data = new_data_x[:, len(not_category_idx):len(not_category_idx)+category_len]
        new_data_x = np.delete(new_data_x, [i for i in range(len(not_category_idx), len(not_category_idx)+category_len)], axis=1)
        new_predicted_idx_data = ori_data_x[:, idx]
        for i in range(ori_data_x.shape[1]):
            if data_m[i][idx] == 0:
                predicted_data = np.argmax(category_data[i])
                new_predicted_idx_data[i] = category_data_inv_map[idx][predicted_data]
        new_predicted_data.append(new_predicted_idx_data)

    predicted_result = ori_data_x.copy()
    for i in range(len(category_idx)):
        predicted_result[:, category_idx[i]] = new_predicted_data[i].reshape(-1)
    for i in range(len(not_category_idx)):
        predicted_result[:, not_category_idx[i]] = new_data_x[:, i] 

    return predicted_result

def main(args):
    
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    ori_data_x, data_x, data_m = data_loader(args.data_name, args.miss_rate)

    # category_idx = [0, 5]
    # category_idx_all = {"wine":[0], "heart":[1, 2, 5, 6, 8, 10, 12], "breast":[9], "car":[0, 1, 2, 3, 4, 5, 6], 
    #                 "wireless":[7], "abalone":[7], "turkiye":[28, 29, 30, 31, 32], "letter":[12, 13, 14, 15], "chess":[5, 6, 7], "shuttle":[6, 7, 8, 9] }

    category_idx_all = {"wine":[0], "heart":[1, 2, 5, 6, 8, 10, 12], "breast":[9], "car":[1, 2, 3, 4, 5, 6], 
                    "wireless":[7], "abalone":[7], "turkiye":[28, 29, 30, 31, 32], "letter":[12, 13, 14, 15], "chess":[4, 5, 6], "shuttle":[6, 7, 8, 9] }

    category_idx = category_idx_all[args.data_name]
    data_x, data_m_imputed, category_len_map, category_data_inv_map = to_onehot(ori_data_x, data_x, data_m, category_idx)
    

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
        M_mb = data_m_imputed[batch_idx, :]  
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
        M_mb = data_m_imputed[batch_idx, :]  
        Z_mb = uniform_sampler(0, 0.01, len(batch_idx), dim)
        X_mb = M_mb * X_mb + (1-M_mb) * Z_mb
    
        imputed_data, mu, logvar = generator.forward(torch.Tensor(X_mb).to(device), torch.Tensor(M_mb).to(device))
        imputed_data = X_mb * M_mb + (1-M_mb) * imputed_data.cpu().detach().numpy()
        imputed_data_all = torch.cat((imputed_data_all, torch.Tensor(imputed_data)),dim=0)


    # Renormalization
    imputed_data = renormalization(imputed_data_all.cpu().detach().numpy(), norm_parameters)  
    
    end = time.time()
    print('Test Time taken: %0.6f' % (end - start))

    # Rounding
    # imputed_data = rounding(imputed_data, data_x)  

    imputed_data = back_from_onehot(imputed_data, data_m, category_idx, ori_data_x, category_len_map, category_data_inv_map)


    print('\n##### RMSE Performance: ' + str(np.round(RMSE(imputed_data, ori_data_x, data_m), 4)))
    print('\n##### MAE Performance: ' + str(np.round(MAE(imputed_data, ori_data_x, data_m), 4)))
    norm_data_x, norm_parameters = normalization(imputed_data)
    # plt.plot(epoch_idx, train_loss)
    # plt.plot(epoch_idx, test_loss, color='red')
    # plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)

