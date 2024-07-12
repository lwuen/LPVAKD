import torch
import torch.nn as nn
import torch.nn.functional as F
# import dgl.function as fn

class logits_D(nn.Module):
    def __init__(self, in_channels, n_hidden):
        super(logits_D, self).__init__()
        self.in_channels = in_channels
        self.n_hidden = n_hidden
        self.lin = nn.Linear(self.in_channels, self.n_hidden)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(self.n_hidden, self.n_hidden, bias=False)

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.lin2.reset_parameters()


    def forward(self, logits, temperature=1.0):
        out = self.lin(logits / temperature)
        out = logits + out
        out = self.relu(out)
        dist = self.lin2(out)
        return dist

class logits_pairs(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(logits_pairs, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)

        return x
