import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import to_dense_adj, softmax
from torch_sparse import SparseTensor


class SelfNodeLabelAggregation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        node_labels = x
        return node_labels


class AttentionNodeLabelAggregation(MessagePassing):
    def __init__(self,
                 n_input: int,
                 negative_slope: float=0.2):
        super().__init__(node_dim=0)

        self.n_input = n_input
        self.negative_slope = negative_slope

        self.att = nn.Parameter(torch.Tensor(1, n_input))
        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.att)

    def forward(self, x, edge_index, return_attention_weights: bool=False):
        x = x.to("cpu")
        edge_index = edge_index.to("cpu")

        x_i = x_j = x

        x_neighbors_att = self.propagate(edge_index, x=(x_j, x_i))
        node_labels = torch.cat((x, x_neighbors_att), dim=-1)

        alpha = self._alpha
        self._alpha = None

        if return_attention_weights:
            assert alpha is not None
            if isinstance(edge_index, torch.Tensor):
                return node_labels, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return node_labels, edge_index.set_value(alpha, layout="coo")
        else:
            return node_labels

    def message(self,
                x_j: torch.Tensor,
                x_i: torch.Tensor,
                index: torch.Tensor) -> torch.Tensor:
        x = x_j + x_i
        x = F.leaky_relu(x, self.negative_slope)
        x = x.to("cuda:0")
        alpha = (x * self.att).sum(dim=-1)
        alpha = alpha.to("cpu")
        alpha = softmax(alpha, index)
        self._alpha = alpha
        return x_j * alpha.unsqueeze(-1)


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

