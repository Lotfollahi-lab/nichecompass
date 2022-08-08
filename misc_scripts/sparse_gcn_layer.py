class SparseGCNLayer(nn.Module):
    """
    ### WIP ###
    Graph convolutional network layer class as per Kipf, T. N. & Welling, M.
    Semi-Supervised Classification with Graph Convolutional Networks. arXiv
    [cs.LG] (2016).

    Parameters
    ----------
    n_input
        Number of input nodes to the GCN Layer.
    n_output
        Number of output nodes from the GCN layer.
    dropout
        Probability of nodes to be dropped during training.
    activation
        Activation function used in the GCN layer.
    """
    def __init__(self,
                 n_input: int,
                 n_output: int,
                 dropout: float = 0.0,
                 activation = torch.relu):
        self.dropout = dropout
        self.activation = activation
        self.weights = nn.Parameter(torch.FloatTensor(n_input, n_output))
        self.initialize_weights()

    def initialize_weights(self):
        # Glorot weight initialization
        torch.nn.init.xavier_uniform_(self.weights)

    def forward(self,
                X: torch.sparse_coo_tensor,
                A: torch.sparse_coo_tensor):
        output = F.dropout(X, self.dropout, self.training)
        output = torch.sparse.mm(output, self.weights)
        output = torch.sparse.mm(A, output)
        return self.activation(output)