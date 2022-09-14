import torch
import torch.nn as nn
from torch_geometric.nn import AttentionalAggregation
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_dense_adj
from torch_sparse import SparseTensor


class SelfNodeLabelAggregation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        node_labels = x
        return node_labels


class AttentionNodeLabelAggregation(nn.Module):
    def __init__(self, n_input: int):
        super().__init__()
        self.aggregator = AttentionalAggregation(gate_nn=nn.Linear(n_input, 1))

    def forward(self, x, edge_index):
        n_nodes = x.size(0)
        adj = to_dense_adj(edge_index=edge_index, max_num_nodes=n_nodes).squeeze(0)
        print(adj.shape)
        print(x.shape)
        x_neighbors_att = self.aggregator(x=x, index=adj)
        print(x_neighbors_att.shape)
        node_labels = torch.cat((x, x_neighbors_att), dim=-1)
        return node_labels


class GCNNormNodeLabelAggregation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        # SparseTensor does not support CUDA
        x = x.to("cpu")
        edge_index = edge_index.to("cpu")
        adj = SparseTensor.from_edge_index(edge_index)
        adj_norm = gcn_norm(adj, add_self_loops=False)
        x_neighbors_norm = adj_norm.matmul(x)
        node_labels = torch.cat((x, x_neighbors_norm), dim=-1)
        return node_labels


class SumNodeLabelAggregation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        # SparseTensor does not support CUDA
        x = x.to("cpu")
        edge_index = edge_index.to("cpu")
        adj = SparseTensor.from_edge_index(edge_index)
        x_neighbors_sum = adj.matmul(x)
        node_labels = torch.cat((x, x_neighbors_sum), dim=-1)
        return node_labels

