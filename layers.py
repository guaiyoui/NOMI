import torch
import torch.nn as nn
import torch.nn.functional as F


# adaptive select the number of neighbors.
# rerank module to select neighbors

class re_rank(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(re_rank, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc_x1 = nn.Linear(input_dim, hidden_dim)
        self.fc_n1 = nn.Linear(2*input_dim, hidden_dim)

        self.fc_3 = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_5 = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, neigh_x, neigh_m):
        x = x.repeat(1, 1, neigh_x.shape[1]).reshape(neigh_x.shape[0], neigh_x.shape[1], neigh_x.shape[2])      

        neigh = torch.concatenate((neigh_x, neigh_m), dim=2)
        # neigh = neigh_x

        x = self.fc_x1(x)
        x = self.relu(x)
        x = F.dropout(x, 0.50, training=self.training)

        neigh = self.fc_n1(neigh)
        neigh = self.relu(neigh)
        neigh = F.dropout(neigh, 0.50, training=self.training)

        x_all = torch.concatenate((x, neigh), dim=2)
        x_all = self.fc_3(x_all)
        x_all = self.relu(x_all)
        x_all = F.dropout(x_all, 0.50, training=self.training)
        x_all = self.fc_5(x_all)
        x_all = self.sigmoid(x_all)
        x_all = x_all.reshape((x_all.shape[0], x_all.shape[1]))

        # return normalization(x_all)
        return x_all

def normalization(d, a=1.0, b=1.2):
    max_, _ =torch.max(d, dim=1)
    min_, _ =torch.min(d, dim=1)
    k = (b-a)/(max_-min_+0.01)
    min_ = min_.reshape((-1, 1))
    min_ = min_.repeat(1, d.shape[1]).reshape((d.shape[0], d.shape[1]))
    k = k.reshape((-1, 1))
    k = k.repeat(1, d.shape[1]).reshape((d.shape[0], d.shape[1]))
    # print(k.shape, d.shape, min_.shape)
    d = a+k*(d-min_)
    return d