import argparse
import time
import random
import torch
import numpy as np
from data_loader import data_loader
from utils import normalization, renormalization, rounding, MAE, RMSE, sample_batch_index, dist2sim
from sklearn.neighbors import NearestNeighbors
from model import NNGP_Imputation, MLP, MLP_Imputation
from tqdm import tqdm
from nngp import NNGP
from neural_tangents import stax
import neural_tangents as nt
import jax
import hnswlib

def prediction(pred_fn, X_test, kernel_type="nngp", compute_cov = True):

		pred_mean, pred_cov = pred_fn(x_test=X_test, get=kernel_type,
									 compute_cov= compute_cov)
		return pred_mean, pred_cov

def normalization_std(data):
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)

    normalized_data = (data - min_vals) / (max_vals - min_vals)
    return normalized_data+1


def main(args):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    ori_data_x, data_x, data_m = data_loader(args.data_name, args.miss_rate, args.missing_mechanism, args)
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)

    num, dims = norm_data_x.shape
    imputed_X = norm_data_x.copy()
    data_m_imputed = data_m.copy()

    init_fn, apply_fn, kernel_fn = stax.serial(
    stax.Dense(2*dims), stax.Relu(),
    stax.Dense(dims), stax.Relu(),
    stax.Dense(1), stax.Sigmoid_like()
    )

    start = time.time()

    time_index = 0 
    
    for iteration in range(args.max_iter):
        for dim in tqdm(range(dims)):
            
            X_wo_dim = np.delete(imputed_X, dim, 1)
            i_not_nan_index = data_m_imputed[:, dim].astype(bool)
            
            X_train = X_wo_dim[i_not_nan_index]
            Y_train = imputed_X[i_not_nan_index, dim]

            X_test = X_wo_dim[~i_not_nan_index]
            true_indices = np.where(~i_not_nan_index)[0]
            # print(~i_not_nan_index, true_indices)
            
            if X_test.shape[0] == 0:
                continue
            
            start_index = time.time()
            no, d = X_train.shape
            index = hnswlib.Index(space=args.metric, dim=d)
            index.init_index(max_elements=no, ef_construction=200, M=16)
            index.add_items(X_train)
            index.set_ef(int(args.k_candidate * 1.2))

            if X_train.shape[0]>300:
                batch_idx = sample_batch_index(X_train.shape[0], 300)
            else:
                batch_idx = sample_batch_index(X_train.shape[0], X_train.shape[0])
            
            X_batch = X_train[batch_idx,:]
            Y_batch = Y_train[batch_idx]

            neigh_ind, neigh_dist = index.knn_query(X_batch, k=args.k_candidate)
            neigh_dist = np.sqrt(neigh_dist)

            weights = dist2sim(neigh_dist[:,1:])
            
            y_neighbors = Y_train[neigh_ind[:,1:]]
            train_input = weights*y_neighbors
            
            neigh_ind_test, neigh_dist_test = index.knn_query(X_test, k=args.k_candidate)
            neigh_dist_test = np.sqrt(neigh_dist_test)

            weights_test = dist2sim(neigh_dist_test[:, :-1])
            y_neighbors_test = Y_train[neigh_ind_test[:, :-1]]
            test_input = weights_test*y_neighbors_test

            time_index += time.time()-start_index
            
            # print("start nngp training")
            predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, 
                    train_input, Y_batch.reshape(-1, 1), diag_reg=1e-4)
            
            y_pred, pred_cov = prediction(predict_fn, test_input, kernel_type="nngp")
            # print(y_pred, pred_cov)
            # pred_cov = np.nan_to_num(pred_cov, nan=1.0)
            

            if iteration == 0:
                imputed_X[~i_not_nan_index, dim] = y_pred.reshape(-1)
            elif iteration <= 3:
                pred_std = np.sqrt(np.diag(pred_cov))
                pred_std = np.ravel(np.array(pred_std))
                pred_std = normalization_std(pred_std)
                
                pred_std = np.nan_to_num(pred_std, nan=1.0)
                
                greater_than_threshold_indices = np.where(pred_std <= args.tau)[0]
                
                for i in range(greater_than_threshold_indices.shape[0]):
                    data_m_imputed[true_indices[greater_than_threshold_indices[i]]:, dim] = 1
                
                imputed_X[~i_not_nan_index, dim] = (1-args.beta/pred_std)*imputed_X[~i_not_nan_index, dim] + args.beta/pred_std*y_pred.reshape(-1)
            else:
                imputed_X[~i_not_nan_index, dim] = y_pred.reshape(-1)
           
            # imputed_X[~i_not_nan_index, dim] = y_pred.reshape(-1)
        print("training using time: ", (time.time()-start))
    imputed_data = renormalization(imputed_X, norm_parameters)  
    imputed_data = rounding(imputed_data, data_x)
    print(ori_data_x[0], imputed_data[0], data_x[0])
    
    print("the input construction time: ", time_index)
    
    print('\n##### RMSE Performance: ' + str(np.round(RMSE(imputed_data, ori_data_x, data_m), 4)))
    print('\n##### MAE Performance: ' + str(np.round(MAE(imputed_data, ori_data_x, data_m), 4)))


def parse_args():
    parser = argparse.ArgumentParser(description='Neural Network Gaussian Process Imputation Network')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--k_candidate', help='how many candidates to save', type=int, default=10)
    parser.add_argument('--batch_size', help='the number of samples in mini-batch', default=16, type=int)
    parser.add_argument('--miss_rate', help='missing data probability', default=0.2, type=float)
    parser.add_argument('--data_name', choices=['wine', 'letter','spam', 'heart', 'breast', 'phishing', 'wireless', 'turkiye', 'credit', 'connect', 'car', 'chess', 'news', 'shuttle', 'poker', 'abalone', 'yeast'], default='wine', type=str)
    parser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    parser.add_argument('--alpha', type=float, default=0.05, help='balance the loss for true and knn')
    parser.add_argument('--beta', type=float, default=1.0, help='balance the imputation for now and previous')
    parser.add_argument('--tau', type=float, default=1.0, help='terminate the iteration')
    parser.add_argument('--max_iter', type=int, default=3, help='The maximum number of iterations.')
    
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0100, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--metric', choices=['cityblock', 'cosine','euclidean', 'haversine', 'ip', 'l1', 'l2', 'manhattan', 'nan_euclidean'], default='l2', type=str)
    parser.add_argument('--device', type=int, default=2, help='Device cuda id')
    parser.add_argument('--missing_mechanism', choices=['MAR', 'MNAR','MCAR'], default='MCAR', type=str)
    # parser.add_argument('--missing_rate', type=float, default=0.2, help='the rate of missing ')
    parser.add_argument('--feature_dim', type=float, default=1.0, help='the rate of feature dimension ')
    parser.add_argument('--training_size', type=float, default=1.0, help='the rate of training size ')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
    