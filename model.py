import torch.nn as nn
import torch.nn.functional as F
import torch
from layers import re_rank

class NNGP_Imputation(nn.Module):
    def __init__(self, input_dim, hidden_dim, k_neighbors):
        super(NNGP_Imputation, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors

    def forward(self, x):
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, column_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.column_dim = column_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.BN1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

        self.linear4 = nn.Linear(column_dim, hidden_dim)
        self.relu4 = nn.ReLU()
        self.BN2 = nn.BatchNorm1d(hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.relu5 = nn.ReLU()
        self.linear6 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

        self.linear7 = nn.Linear(2, 1)
    
    def forward(self, x, column_input, epoch):
        x = self.linear1(x)
        x = self.relu1(x)
        x = F.dropout(x, 0.40, training=self.training)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.sigmoid(x)

        column_input = self.linear4(column_input)
        column_input = self.relu4(column_input)
        column_input = F.dropout(column_input, 0.40, training=self.training)
        column_input = self.linear5(column_input)
        column_input = self.relu5(column_input)
        column_input = self.linear6(column_input)
        column_input = self.sigmoid(column_input)

        x = torch.cat((x, column_input), 1)

        return self.sigmoid(self.linear7(x))

class MLP_Imputation(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, column_dim):
        super(MLP_Imputation, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.column_dim = column_dim

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.BN1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

        self.linear4 = nn.Linear(column_dim, hidden_dim)
        self.relu4 = nn.ReLU()
        self.BN2 = nn.BatchNorm1d(hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.relu5 = nn.ReLU()
        self.linear6 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

        self.linear7 = nn.Linear(2, 1)

        self.re_rank = re_rank(column_dim, hidden_dim, output_dim)
    
    def forward(self, x, column_input, Neigh_X, Neigh_M, dim):
        
        a = self.re_rank(column_input, Neigh_X, Neigh_M)
        # # print("=====a: ======: ", a)
        x = a*x

        x = self.linear1(x)
        x = self.relu1(x)
        x = F.dropout(x, 0.40, training=self.training)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.sigmoid(x)

        column_input = self.linear4(column_input)
        column_input = self.relu4(column_input)
        column_input = F.dropout(column_input, 0.40, training=self.training)
        column_input = self.linear5(column_input)
        column_input = self.relu5(column_input)
        column_input = self.linear6(column_input)
        column_input = self.sigmoid(column_input)

        x = torch.cat((x, column_input), 1)

        return self.sigmoid(self.linear7(x))
