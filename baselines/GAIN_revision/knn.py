import numpy as np
import math
import torch
from sklearn.neighbors import NearestNeighbors

def COS(x, y, mx, my):
    x = x*my
    y = y*mx
    return np.dot(x,y)/math.sqrt(np.dot(x,x)*np.dot(y,y))

def knn_search(X, M, K):
    no, dim = np.shape(X)

    X_KNN = []
    X_KSimilarity = []

    for i in range(no):
        if i % 100 == 0:
            print(i)
        topk_similarity = [0 for m in range(K)]
        topk_index = [-1 for m in range(K)]
        for j in range(no):
            if i == j: continue
            similarity = COS(X[i], X[j], M[i], M[j])
            for m in range(K):
                if similarity > topk_similarity[m]:
                    topk_similarity[m] = similarity
                    topk_index[m] = j
                    break
        
        KNN_i = []
        for idx in topk_index:
            KNN_i.append(X[idx])

        X_KNN.append(np.array(KNN_i))
        X_KSimilarity.append(np.array(topk_similarity))
        
    return np.array(X_KNN), np.array(X_KSimilarity)
        
def knn_search_matrix(X, M, K):

    X_zero = X*M
    no, dim = np.shape(X_zero)
    x_dot = X_zero.dot(X_zero.T)
    diagonal = np.diagonal(x_dot).reshape(-1,1)
    xy = diagonal.dot(diagonal.T).astype(np.float32)
    similarity = torch.from_numpy(x_dot / np.sqrt(xy))
    a,_ = similarity.topk(k=K, dim=1)
    a_min = torch.min(a,dim=-1).values
    a_min = a_min.unsqueeze(-1).repeat(1,similarity.shape[-1])
    ge = torch.ge(similarity, a_min).int()
    X_KNN = []
    X_KSimilarity = []
    for i in range(no):
        index = torch.nonzero(ge[i]==1).squeeze()
        X_KNN.append(X[index])
        X_KSimilarity.append(similarity[i][index].detach().numpy())
    
    return np.array(X_KNN), np.array(X_KSimilarity)

def knn_search_sklearn(X, M, K):

    X_zero = X*M
    no, dim = np.shape(X_zero)
    nbrs = NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(X_zero)
    distances, indices = nbrs.kneighbors(X)
    no, dim = np.shape(X)
    X_KNN = []
    for i in range(no):
        X_KNN.append(X[indices[i]])
    return np.array(X_KNN), np.array(1-distances)
    # return np.array(X_KNN), np.array(X_KSimilarity)


# def ann_search_sklearn(X, M, K):

