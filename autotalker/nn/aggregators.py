from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import to_dense_adj, softmax
from torch_sparse import SparseTensor


class AttentionNodeLabelAggregator(MessagePassing):
    def __init__(self,
                 n_input: int,
                 n_heads: int=4,
                 leaky_relu_negative_slope: float=0.2):
        super().__init__(node_dim=0)

        self.n_input = n_input
        self.n_heads = n_heads
        self.leaky_relu_negative_slope = leaky_relu_negative_slope

        self.linear_l_l = Linear(n_input,
                                 n_input * n_heads,
                                 bias=False,
                                 weight_initializer="glorot")
        self.linear_r_l = Linear(n_input,
                                 n_input * n_heads,
                                 bias=False,
                                 weight_initializer="glorot")
        self.attn = nn.Parameter(torch.Tensor(1, n_heads, n_input))
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        self.linear_l_l.reset_parameters()
        self.linear_r_l.reset_parameters()
        glorot(self.attn)

    def forward(self, x, edge_index, return_attention_weights: bool=False):
        x_l = x_r = x
        g_l = self.linear_l_l(x_l).view(-1, self.n_heads, self.n_input)
        g_r = self.linear_r_l(x_r).view(-1, self.n_heads, self.n_input)
        x_l = x_l.repeat(1, self.n_heads).view(-1, self.n_heads, self.n_input)
        output = self.propagate(edge_index, x=(x_l, x_r), g=(g_l, g_r))
        x_neighbors_att = output.mean(dim=1)
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
                g_j: torch.Tensor,
                g_i: torch.Tensor,
                index: torch.Tensor) -> torch.Tensor:
        g = g_i + g_j
        g = self.activation(g)
        alpha = (g * self.attn).sum(dim=-1)
        alpha = softmax(alpha, index)
        self._alpha = alpha
        return x_j * alpha.unsqueeze(-1)


class GCNNormNodeLabelAggregator(nn.Module):
    """
    cover case in which no edge in sampled batch
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        adj = SparseTensor.from_edge_index(edge_index)
        adj_norm = gcn_norm(adj, add_self_loops=False)
        x_neighbors_norm = adj_norm.matmul(x)
        node_labels = torch.cat((x, x_neighbors_norm), dim=-1)
        return node_labels


class SelfNodeLabelPseudoAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        node_labels = x
        return node_labels


class SumNodeLabelAggregator(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_index):
        adj = SparseTensor.from_edge_index(edge_index)
        x_neighbors_sum = adj.matmul(x)
        node_labels = torch.cat((x, x_neighbors_sum), dim=-1)
        return node_labels

