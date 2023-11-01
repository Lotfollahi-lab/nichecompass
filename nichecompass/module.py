import torch
import torch_sparse
import torch_geometric


class StaticMaskedLinear(torch.nn.Linear):

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor, bias=False):

        if in_features != mask.shape[0] or out_features != mask.shape[1]:
            raise ValueError(f"Invalid shape of the mask. Required shape: ({in_features} x {out_features})")

        super().__init__(in_features, out_features, bias)

        self.register_buffer("mask", mask.t())
        self.weight.data *= self.mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        masked_weights = self.weight * self.mask
        return torch.nn.functional.linear(x, masked_weights, self.bias)


class DynamicMaskedLinear(torch.nn.Linear):
    #FIXME this shouldn't be a child of torch.nn.Linear :(

    def __init__(self, in_features: int, out_features: int, mask: torch.Tensor, bias=False):
        if in_features != mask.shape[0] or out_features != mask.shape[1]:
            raise ValueError(f"Invalid shape of the mask. Required shape: ({in_features} x {out_features})")

        super().__init__(in_features, out_features, bias)

        self.register_buffer("mask", mask.t())
        self.weight.data *= self.mask

    def forward(self, x: torch.Tensor, dynamic_mask: torch.Tensor) -> torch.Tensor:
        dynamic_mask = dynamic_mask.t().to(self.mask.device)
        self.weight.data *= dynamic_mask
        masked_weights = self.weight * self.mask * dynamic_mask
        return torch.nn.functional.linear(x, masked_weights, self.bias)


class GCNSimpleConv(torch_geometric.nn.conv.MessagePassing):
    #FIXME isn't this just a re-implementation of torch_geometric.nn.conv.GCNConv ? or nn.conv.SimpleConv?
    #FIXME find a way to import this rather than recreating it! (perhaps with a transform or aggregation fun)

    def __init__(self):
        super().__init__()

    def forward(self, node_attributes: torch.Tensor, edge_list: torch.Tensor) -> torch.Tensor:
        adjacency_matrix = torch_sparse.SparseTensor.from_edge_index(edge_list,
                                           sparse_sizes=(node_attributes.shape[0],
                                                         node_attributes.shape[0]))
        normalised_adjacency_matrix = torch_geometric.nn.conv.gcn_conv.gcn_norm(adjacency_matrix)
        aggregated_node_attributes = normalised_adjacency_matrix.t().matmul(node_attributes)
        return aggregated_node_attributes


class GATSimpleConv(torch_geometric.nn.conv.MessagePassing):
    #FIXME this should be refactored to mirror the torch_geometric gatv2_conv function with docs added
    def __init__(self,
                 n_input: int,
                 n_heads: int=4,
                 leaky_relu_negative_slope: float=0.2,
                 dropout_rate: float=0.):
        """
        One-hop Attention Node Label Aggregator class that uses a weighted sum
        of the gene expression of a node's 1-hop neighbors to build an
        aggregated neighbor omics feature vector for a node. The weights are
        determined by an additivite attention mechanism with learnable weights.
        It returns a concatenation of the node's own omics feature vector and
        the attention-aggregated neighbor omics feature vector as node labels
        for the omics reconstruction task.

        Parts of the implementation are inspired by
        https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/nn/conv/gatv2_conv.py#L16
        (01.10.2022).

        Parameters
        ----------
        modality:
            Omics modality that is aggregated. Can be either `rna` or `atac`.
        n_input:
            Number of omics features used for the Node Label Aggregation.
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
        self.linear_l_l = torch_geometric.nn.dense.linear.Linear(n_input,
                                 n_input * n_heads,
                                 bias=False,
                                 weight_initializer="glorot")
        self.linear_r_l = torch_geometric.nn.dense.linear.Linear(n_input,
                                 n_input * n_heads,
                                 bias=False,
                                 weight_initializer="glorot")
        self.attn = torch.nn.Parameter(torch.Tensor(1, n_heads, n_input))
        self.activation = torch.nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self._alpha = None
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset weight parameters.
        """
        self.linear_l_l.reset_parameters()
        self.linear_r_l.reset_parameters()
        torch_geometric.nn.inits.glorot(self.attn)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                return_agg_weights: bool=False) -> torch.Tensor:
        """
        Forward pass of the One-hop Attention Node Label Aggregator.

        Parameters
        ----------
        x:
            Tensor containing the omics features of the nodes in the current
            node batch including sampled neighbors.
            (Size: n_nodes_batch_and_sampled_neighbors x n_node_features)
        edge_index:
            Tensor containing the node indices of edges in the current node
            batch including sampled neighbors.
            (Size: 2 x n_edges_batch_and_sampled_neighbors)
        return_agg_weights:
            If ´True´, also return the aggregation weights (attention weights).

        Returns
        ----------
        x_neighbors:
            Tensor containing the node labels of the nodes in the current node
            batch excluding sampled neighbors. These labels are used for the
            omics feature reconstruction task.
            (Size: n_nodes_batch x (2 x n_node_features))
        alpha:
            Aggregation weights for edges in ´edge_index´.
        """
        x_l = x_r = x
        g_l = self.linear_l_l(x_l).view(-1, self.n_heads, self.n_input)
        g_r = self.linear_r_l(x_r).view(-1, self.n_heads, self.n_input)
        x_l = x_l.repeat(1, self.n_heads).view(-1, self.n_heads, self.n_input)

        output = self.propagate(edge_index, x=(x_l, x_r), g=(g_l, g_r))
        x_neighbors = output.mean(dim=1)
        alpha = self._alpha
        self._alpha = None
        if return_agg_weights:
            return x_neighbors, alpha
        return x_neighbors, None

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
        alpha = torch_geometric.utils.softmax(alpha, index) # index is 2nd dim of edge_index (index of
                                      # central node over which softmax should
                                      # be applied)
        self._alpha = alpha
        alpha = self.dropout(alpha)
        return x_j * alpha.unsqueeze(-1)
