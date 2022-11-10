import torch
import torch.nn as nn
from torch_geometric.nn.inits import glorot
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.utils import softmax
from torch_sparse import SparseTensor


class OneHopAttentionNodeLabelAggregator(MessagePassing):
    def __init__(self,
                 n_input: int,
                 n_heads: int=4,
                 leaky_relu_negative_slope: float=0.2,
                 dropout_rate: float=0.0):
        """
        One-hop Attention Node Label Aggregator class that uses a weighted sum
        of the gene expression of a node's 1-hop neighbors to build an
        aggregated neighbor gene expression vector for a node. The weights are
        determined by an attention mechanism with learnable weights. It returns 
        a concatenation of the node's own gene expression and the 
        attention-aggregated neighbor gene expression vector as node labels for
        the gene expression reconstruction task. Implementation inspired by
        https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gatv2_conv.py#L16.

        Parameters
        ----------
        n_input:
            Number of input nodes to the Node Label Aggregator (corresponds to
            number of genes).
        n_heads:
            Number of attention heads for multi-head attention.
        leaky_relu_negative_slope:
            Slope of the leaky relu activation function.
        dropout_rate:
            Dropout probability of the normalized attention coefficients which
            exposes each node to a stochastically sampled neighborhood during 
            training.
        """
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
        self.dropout = nn.Dropout(p=dropout_rate)
        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset weight parameters.
        """
        self.linear_l_l.reset_parameters()
        self.linear_r_l.reset_parameters()
        glorot(self.attn)

    def forward(self,
                x: torch.tensor,
                edge_index: torch.tensor,
                batch_size: int,
                return_attention_weights: bool=False):
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
        batch_size:
            Node batch size. Is used to return only node labels for the nodes
            in the current node batch.
        return_attention_weights:
            If ´True´, in addition return the learned attention weights.

        Returns
        ----------
        node_labels:
            Tensor containing the node labels of the nodes in the current node 
            batch excluding sampled neighbors. These labels are used for the 
            gene expression reconstruction task.
            (Size: n_nodes_batch x (2 x n_node_features))
        """
        x_l = x_r = x
        g_l = self.linear_l_l(x_l).view(-1, self.n_heads, self.n_input)
        g_r = self.linear_r_l(x_r).view(-1, self.n_heads, self.n_input)
        x_l = x_l.repeat(1, self.n_heads).view(-1, self.n_heads, self.n_input)
        output = self.propagate(edge_index, x=(x_l, x_r), g=(g_l, g_r))
        x_neighbors_att = output.mean(dim=1)
        node_labels = torch.cat((x, x_neighbors_att), dim=-1)[:batch_size]

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
        """
        Message method of the MessagePassing parent class. Variables with "_i" 
        suffix refer to the central nodes that aggregate information. Variables
        with "_j" suffix refer to the neigboring nodes.
        """
        g = g_i + g_j
        g = self.activation(g)
        alpha = (g * self.attn).sum(dim=-1)
        alpha = softmax(alpha, index)
        self._alpha = alpha
        alpha = self.dropout(alpha)
        return x_j * alpha.unsqueeze(-1)


class OneHopGCNNormNodeLabelAggregator(nn.Module):
    """
    One-hop GCN Norm Node Label Aggregator class that uses a symmetrically
    normalized sum (as introduced in Kipf, T. N. & Welling, M. Semi-Supervised 
    Classification with Graph Convolutional Networks. arXiv [cs.LG] (2016)) of 
    the gene expression of a node's 1-hop neighbors to build
    an aggregated neighbor gene expression vector for a node. It returns a 
    concatenation of the node's own gene expression and the gcn-norm aggregated
    neighbor gene expression vector as node labels for the gene expression
    reconstruction task.
    """
    def __init__(self):
        super().__init__()

    def forward(self,
                x: torch.tensor,
                edge_index: torch.tensor,
                batch_size: int) -> torch.tensor:
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
        adj_norm = gcn_norm(adj, add_self_loops=False)
        x_neighbors_norm = adj_norm.t().matmul(x)
        node_labels = torch.cat((x, x_neighbors_norm), dim=-1)
        return node_labels[:batch_size]


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
        # Get adjacency matrix with edges of nodes in the current batch with 
        # sampled neighbors only (exluding edges between sampled neighbors)
        #batch_nodes_edge_index = edge_index[:, (edge_index[1] < batch_size)]
        #adj = SparseTensor.from_edge_index(
        #    batch_nodes_edge_index,
        #    sparse_sizes=(max(batch_nodes_edge_index[0]+1), batch_size))
        adj = SparseTensor.from_edge_index(edge_index,
                                           sparse_sizes=(x.shape[0],
                                                         x.shape[0]))
        x_neighbors_sum = adj.t().matmul(x)
        node_labels = torch.cat((x[:batch_size,:], x_neighbors_sum), dim=-1)
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
                x: torch.tensor,
                edge_index: torch.tensor,
                batch_size: int):
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