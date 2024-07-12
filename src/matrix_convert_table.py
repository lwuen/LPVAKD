import copy
import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from torch_geometric.transforms import RandomNodeSplit

def aggregate_node_feature(x, edge_index):
    row, col = edge_index
    out = torch.zeros_like(x)
    out.scatter_add_(0, col.unsqueeze(1).repeat(1, x.size(1)), x[row])
    out.scatter_add_(0, row.unsqueeze(1).repeat(1, x.size(1)), x[col])
    out= out + x
    return out