import torch
import torch.nn.functional as F
from torch import nn as nn


class DotProductDecoder(nn.Module):
    """
    Dot product decoder class as per Kipf, T. N. & Welling, M. Variational Graph
    Auto-Encoders. arXiv [stat.ML] (2016).

    Takes the latent space features Z as input, calculates their dot product
    and returns the predicted adjacency matrix A_pred.

    Parameters
    ----------
    dropout
        Probability of nodes to be dropped during training.
    activation
        Activation function used for predicting the adjacency matrix. Defaults
        to sigmoid activation.
    """

    def __init__(self, dropout: float = 0.0, activation=torch.sigmoid):
        super(DotProductDecoder, self).__init__()
        self.dropout = dropout
        self.activation = activation

    def forward(self, Z):
        Z_dropout = F.dropout(Z, self.dropout, self.training)
        dot_product = torch.mm(Z_dropout, Z_dropout.t())
        A_pred = self.activation(dot_product)
        return A_pred


class FCLayer(nn.Module):
    """
    Fully connected layer class.

    Parameters
    ----------
    n_input
        Number of input nodes to the FC Layer.
    n_output
        Number of output nodes from the FC layer.
    activation
        Activation function used in the FC layer.
    """

    def __init__(self, n_input: int, n_output: int, activation=F.relu):
        self.activation = activation
        self.linear = nn.Linear(n_input, n_output)

    def forward(self, input: torch.Tensor):
        output = self.linear(input)
        if self.activation is not None:
            return self.activation(output)
        else:
            return output


class GCNLayer(nn.Module):
    """
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
                 activation=F.relu):
        super(GCNLayer, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.weights = nn.Parameter(
            torch.FloatTensor(n_input, n_output)
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Glorot weight initialization
        torch.nn.init.xavier_uniform_(self.weights)

    def forward(self, input: torch.Tensor, adj_mtx: torch.Tensor):
        output = F.dropout(input, self.dropout, self.training)
        output = torch.mm(output, self.weights)
        output = torch.mm(adj_mtx, output)
        return self.activation(output)


class SparseGCNLayer(nn.Module):
    """
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
                 activation=F.relu):
        self.dropout = dropout
        self.activation = activation
        self.weights = nn.Parameter(
            torch.Tensor(n_input, n_output, dtype=torch.float32)
        )
        self.initialize_weights()

    def initialize_weights(self):
        # Glorot weight initialization
        torch.nn.init.xavier_uniform_(self.weights)

    def forward(self,
                input: torch.sparse_coo_tensor,
                adj_mtx: torch.sparse_coo_tensor):
        output = F.dropout(input, self.dropout, self.training)
        output = torch.sparse.mm(output, self.weights)
        output = torch.sparse.mm(adj_mtx, output)
        return self.activation(output)


class GCNEncoder(nn.Module):
    """
    Graph convolutional network encoder class as per Kipf, T. N. & Welling, M.
    Variational Graph Auto-Encoders. arXiv [stat.ML] (2016).

    Takes the input space features X and the adjacency matrix A as input,
    computes one shared GCN layer and two separate GCN layers to output mu and
    logstd of the latent space distribution.

    Parameters
    ----------
    n_input
        Number of input nodes to the GCN encoder.
    n_hidden
        Number of hidden nodes after the first GCN layer.
    n_latent
        Number of output nodes from the GCN encoder, making up the latent space.
    dropout
        Probability of nodes in the first GCN layer to be dropped during
        training.
    activation
        Activation function used in the first GCN layer. Defaults to relu
        activation.
    """

    def __init__(
        self,
        n_input: int,
        n_hidden: int,
        n_latent: int,
        dropout: float = 0.0,
        activation=F.relu,
    ):
        super(GCNEncoder, self).__init__()
        self.gcn_l1 = GCNLayer(n_input, n_hidden, dropout, activation=activation)
        self.gcn_mu = GCNLayer(n_hidden, n_latent, activation=lambda x: x)
        self.gcn_logstd = GCNLayer(n_hidden, n_latent, activation=lambda x: x)

    def forward(self, X, A):
        hidden = self.gcn_l1(X, A)
        mu = self.gcn_mu(hidden, A)
        logstd = self.gcn_logstd(hidden, A)
        return mu, logstd
