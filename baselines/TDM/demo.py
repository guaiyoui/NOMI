
import numpy as np
import os
from utils import generate_mask
from run_tdm import run_TDM
import argparse

def main(args):
    print(args)
    data_dir = './datasets/'
    # dataset = 'seeds_complete'
    # dataset = 'wine'
    # dataset = 'heart'
    # dataset = 'breast'
    # dataset = 'car'
    # dataset = 'wireless'
    # dataset = 'abalone'
    # dataset = 'turkiye'
    # # dataset = 'letter'
    # dataset = 'chess'
    # dataset = 'shuttle'
    dataset = args.data_name
    missing_prop = args.missing_rate
    # missing_type = 'MCAR' # Choosing from MAR, MNARL, MNARQ, MCAR
    missing_type = args.missing_mechanism
    # data = np.load(os.path.join(data_dir,  '{}.npy'.format(dataset)), allow_pickle=True).item()
    # X_true = data['X_true']
    file_name = data_dir+dataset+'.csv'
    X_true = np.loadtxt(file_name, delimiter=",", skiprows=1)
    
    feature_dim = int(X_true.shape[1]*args.feature_dim)
    X_true = X_true[:, :feature_dim]

    training_size = int(X_true.shape[0]*args.training_size)
    X_true = X_true[:training_size, :]
    
    if missing_type != "MCAR":
        missing_prop = 2*missing_prop
    # missing_prop = 1-missing_prop
    mask = generate_mask(X_true, missing_prop, missing_type)
    X_missing = np.copy(X_true)
    X_missing[mask.astype(bool)] = np.nan

    niter = 10000
    batchsize = 64
    lr = 1e-2
    report_interval = 100
    network_depth = 3
    network_width = 2
    args = {'out_dir': f'./demo1_{dataset}', 'niter': niter, 'data_name': dataset,
    'batchsize': batchsize, 'lr': lr, 'network_width': network_width, 'network_depth': network_depth, 'report_interval': report_interval}


    run_TDM(X_missing, args, X_true)

def parse_args():
    parser = argparse.ArgumentParser(description='Neural Network Gaussian Process Imputation Network')
    parser.add_argument('--data_name', choices=['wine', 'letter','spam', 'heart', 'breast', 'phishing', 'wireless', 'turkiye', 'credit', 'connect', 'car', 'chess', 'news', 'shuttle', 'poker', 'abalone', 'yeast', 'retail', 'wisdm', 'higgs'], default='wine', type=str)
    parser.add_argument('--missing_mechanism', choices=['MAR', 'MNARL','MCAR'], default='MCAR', type=str)
    parser.add_argument('--missing_rate', type=float, default=0.2, help='the rate of missing ')
    parser.add_argument('--feature_dim', type=float, default=1.0, help='the rate of feature dimension ')
    parser.add_argument('--training_size', type=float, default=1.0, help='the rate of training size ')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(args)