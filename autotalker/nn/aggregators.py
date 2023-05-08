"""
This module contains gene expression aggregators used by the Autotalker model.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
from torch_sparse import SparseTensor


class OneHopAttentionNodeLabelAggregator(MessagePassing):
    def __init__(self,
                 n_input: int,
                 features_idx: torch.Tensor,
                 n_heads: int=4,
                 leaky_relu_negative_slope: float=0.2,
                 dropout_rate: float=0.,
                 self_loops: bool=True):
        """
        One-hop Attention Node Label Aggregator class that uses a weighted sum
        of the gene expression of a node's 1-hop neighbors to build an
        aggregated neighbor gene expression vector for a node. The weights are
        determined by an additivite attention mechanism with learnable weights.
        It returns a concatenation of the node's own gene expression and the 
        attention-aggregated neighbor gene expression vector as node labels for
        the gene expression reconstruction task. 
        
        Parts of the implementation are inspired by
        https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gatv2_conv.py#L16
        (01.10.2022).

        Parameters
        ----------
        n_input:
            Number of omics features used for the Node Label Aggregation.
        features_idx:
            Index of omics features that are in the gp and ca masks.
        n_heads:
            Number of attention heads for multi-head attention.
        leaky_relu_negative_slope:
            Slope of the leaky relu activation function.
        dropout_rate:
            Dropout probability of the normalized attention coefficients which
            exposes each node to a stochastically sampled neighborhood during 
            training.
        self_loops:
            If ´True´, use self loops to model autocrine communication.
        """
        super().__init__(node_dim=0)
        self.n_input = n_input
        self.features_idx = features_idx
        self.n_heads = n_heads
        self.leaky_relu_negative_slope = leaky_relu_negative_slope
        self.self_loops = self_loops
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
        self.dropout = nn.Dropout(p=dropout_rate)
        self._alpha = None
        self.reset_parameters()

        print(f"ONE HOP ATTENTION NODE LABEL AGGREGATOR -> n_input: {n_input}, "
              f"n_heads: {n_heads}, self_loops: {self.self_loops}")

    def reset_parameters(self):
        """
        Reset weight parameters.
        """
        self.linear_l_l.reset_parameters()
        self.linear_r_l.reset_parameters()
        glorot(self.attn)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                return_attention_weights: bool=False) -> torch.Tensor:
        """
        Forward pass of the One-hop Attention Node Label Aggregator.
        
        Parameters
        ----------
        x:
            Tensor containing the gene expression of the nodes in the current 
            node batch including sampled neighbors. 
            (Size: n_nodes_batch_and_sampled_neighbors x n_node_features)
        edge_index:
            Tensor containing the node indices of edges in the current node 
            batch including sampled neighbors.
            (Size: 2 x n_edges_batch_and_sampled_neighbors)
        return_attention_weights:
            If ´True´, also return the attention weights with the corresponding
            edge index.

        Returns
        ----------
        node_labels:
            Tensor containing the node labels of the nodes in the current node 
            batch excluding sampled neighbors. These labels are used for the 
            gene expression reconstruction task.
            (Size: n_nodes_batch x (2 x n_node_features))
        alpha:
            Attention weights for edges in ´edge_index´.
        """
        x_l = x_r = x
        g_l = self.linear_l_l(x_l).view(-1, self.n_heads, self.n_input)
        g_r = self.linear_r_l(x_r).view(-1, self.n_heads, self.n_input)
        x_l = x_l.repeat(1, self.n_heads).view(-1, self.n_heads, self.n_input)

        if self.self_loops:
            # Add self loops to account for autocrine communication
            n_nodes = x.size(0)
            edge_index, _ = remove_self_loops(edge_index) # in case there are
                                                          # already self loops
            edge_index, _ = add_self_loops(edge_index,
                                           num_nodes=n_nodes)

        output = self.propagate(edge_index, x=(x_l, x_r), g=(g_l, g_r))
        x_neighbors_att = output.mean(dim=1)
        node_labels = torch.cat(
            (x, x_neighbors_att), dim=-1)[:, self.features_idx]
        alpha = self._alpha
        self._alpha = None
        if return_attention_weights:
            return node_labels, alpha
        return node_labels, None

    def message(self,
                x_j: torch.Tensor,
                x_i: torch.Tensor,
                g_j: torch.Tensor,
                g_i: torch.Tensor,
                index: torch.Tensor) -> torch.Tensor:
        """
        Message method of the MessagePassing parent class. Variables with "_i" 
        suffix refer to the central nodes that aggregate information. Variables
        with "_j" suffix refer to the neigboring nodes.

        Parameters
        ----------
        x_j:
            Gene expression of neighboring nodes (dim: n_index x n_heads x
            n_node_features).
        g_i:
            Key vector of central nodes (dim: n_index x n_heads x
            n_node_features).
        g_j:
            Query vector of neighboring nodes (dim: n_index x n_heads x
            n_node_features).     
        """
        g = g_i + g_j
        g = self.activation(g)
        alpha = (g * self.attn).sum(dim=-1)
        alpha = softmax(alpha, index) # index is 2nd dim of edge_index (index of
                                      # central node over which softmax should
                                      # be applied)
        self._alpha = alpha
        alpha = self.dropout(alpha)
        return x_j * alpha.unsqueeze(-1)


class OneHopGCNNormNodeLabelAggregator(nn.Module):
    """
    One-hop GCN Norm Node Label Aggregator class that uses a symmetrically
    normalized sum (as introduced in Kipf, T. N. & Welling, M. Semi-Supervised 
    Classification with Graph Convolutional Networks. arXiv [cs.LG] (2016)) of 
    the omics feature vector of a node's 1-hop neighbors to build
    an aggregated neighbor gene expression vector for a node. It returns a 
    concatenation of the node's own gene expression and the gcn-norm aggregated
    neighbor gene expression vector as node labels for the gene expression
    reconstruction task.
    """
    def __init__(self,
                 features_idx: torch.Tensor,
                 self_loops: bool=True):
        """
        Parameters
        ----------
        features_idx:
            Index of omics features that are in the gp and ca masks.
        self_loops:
            If ´True´, use self loops to model autocrine communication.
        """
        super().__init__()
        self.features_idx = features_idx
        self.self_loops = self_loops

        print("ONE HOP GCN NORM NODE LABEL AGGREGATOR -> self_loops: "
              f"{self_loops}")

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                return_attention_weights: bool=False) -> torch.Tensor:
        """
        Forward pass of the One-hop GCN Norm Node Label Aggregator.
        
        Parameters
        ----------
        x:
            Tensor containing the gene expression of the nodes in the current 
            node batch including sampled neighbors. 
            (Size: n_nodes_batch_and_sampled_neighbors x n_node_features)
        edge_index:
            Tensor containing the node indices of edges in the current node 
            batch including sampled neighbors.
            (Size: 2 x n_edges_batch_and_sampled_neighbors)

        Returns
        ----------
        node_labels:
            Tensor containing the node labels of the nodes in the current node 
            batch. These labels are used for the gene expression reconstruction
            task. (Size: n_nodes_batch x (2 x n_node_features))
        """
        adj = SparseTensor.from_edge_index(edge_index,
                                           sparse_sizes=(x.shape[0],
                                                         x.shape[0]))
        adj_norm = gcn_norm(adj, add_self_loops=self.self_loops)
        x_neighbors_norm = adj_norm.t().matmul(x)
        node_labels = torch.cat((x, x_neighbors_norm),
                                dim=-1)[:, self.features_idx]
        return node_labels


class OneHopSumNodeLabelAggregator(nn.Module):
    """
    One-hop Sum Node Label Aggregator class that sums up the gene expression of
    a node's 1-hop neighbors to build an aggregated neighbor gene expression 
    vector for a node. It returns a concatenation of the node's own gene 
    expression and the sum-aggregated neighbor gene expression vector as node 
    labels for the gene expression reconstruction task.
    """
    def __init__(self):
        super().__init__()

    def forward(self,
                x: torch.Tensor,
                edge_index:torch.Tensor,
                batch_size:int) -> torch.Tensor:
        """
        Forward pass of the One-hop Sum Node Label Aggregator.
        
        Parameters
        ----------
        x:
            Tensor containing the gene expression of the nodes in the current 
            node batch including sampled neighbors. 
            (Size: n_nodes_batch_and_sampled_neighbors x n_node_features)
        edge_index:
            Tensor containing the node indices of edges in the current node 
            batch including sampled neighbors.
            (Size: 2 x n_edges_batch_and_sampled_neighbors)
        batch_size:
            Node batch size. Is used to return only node labels for the nodes
            in the current node batch.

        Returns
        ----------
        node_labels:
            Tensor containing the node labels of the nodes in the current node 
            batch excluding sampled neighbors. These labels are used for the 
            gene expression reconstruction task.
            (Size: n_nodes_batch x (2 x n_node_features))
        """
        adj = SparseTensor.from_edge_index(edge_index,
                                           sparse_sizes=(x.shape[0],
                                                         x.shape[0]))
        x_neighbors_sum = adj.t().matmul(x)
        node_labels = torch.cat((x, x_neighbors_sum), dim=-1)
        return node_labels[:batch_size]


class SelfNodeLabelNoneAggregator(nn.Module):
    """
    Self Node Label None Aggregator class that provides an API to pass ´x´ and
    ´edge_index´ to the forward pass (for consistency with other aggregators) 
    but does no neighborhood gene expression aggregation. Instead, it just 
    returns the gene expression of the nodes themselves as labels for the gene 
    expression reconstruction task (hence 'none aggregator'). Note that this 
    is a capability for benchmarking but is not compatible with the inference of
    communication gene programs that require an aggregation of the neighborhood
    gene expression.
    """
    def __init__(self):
        super().__init__()

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                batch_size: int) -> torch.Tensor:
        """
        Forward pass of the One-hop Sum Node Label Aggregator.
        
        Parameters
        ----------
        x:
            Tensor containing the gene expression of the nodes in the current 
            node batch including sampled neighbors. 
            (Size: n_nodes_batch_and_sampled_neighbors x n_node_features)
        edge_index:
            Tensor containing the node indices of edges in the current node 
            batch including sampled neighbors.
            (Size: 2 x n_edges_batch_and_sampled_neighbors)
        batch_size:
            Node batch size. Is used to return only node labels for the nodes
            in the current node batch.

        Returns
        ----------
        node_labels:
            Tensor containing the node labels of the nodes in the current node 
            batch excluding sampled neighbors. These labels are used for the 
            gene expression reconstruction task.
            (Size: n_nodes_batch x (2 x n_node_features))
        """
        node_labels = x
        return node_labels[:batch_size]