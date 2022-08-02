import torch
import torch.nn.functional as F
from torch import nn as nn


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network encoder class as per Kipf, T. N. & Welling, M.
    Variational Graph Auto-Encoders. arXiv [stat.ML] (2016).

    Takes the input space features X and the adjacency matrix A as input,
    computes one shared GCN layer and two separate GCN layers to output mu and
    logstd of the latent space distribution.

    Parameters
    ----------
    n_input
        Number of input nodes to the GCN encoder.
    n_hidden
        Number of hidden nodes outputted by the first GCN layer.
    n_latent
        Number of output nodes from the GCN encoder, making up the latent 
        features.
    dropout
        Probability of nodes to be dropped in the first GCN layer during
        training.
    activation
        Activation function used in the first GCN layer.
    """
    def __init__(
            self,
            n_input: int,
            n_hidden: int,
            n_latent: int,
            dropout: float = 0.0,
            activation = torch.relu):
        super(GCNEncoder, self).__init__()
        self.gcn_l1 = GCNLayer(n_input, n_hidden, dropout, activation)
        self.gcn_mu = GCNLayer(n_hidden, n_latent, activation = lambda x: x)
        self.gcn_logstd = GCNLayer(n_hidden, n_latent, activation = lambda x: x)

    def forward(self, X, A):
        hidden = self.gcn_l1(X, A)
        mu = self.gcn_mu(hidden, A)
        logstd = self.gcn_logstd(hidden, A)
        return mu, logstd


class DotProductDecoder(nn.Module):
    """
    Dot product decoder class as per Kipf, T. N. & Welling, M. Variational Graph
    Auto-Encoders. arXiv [stat.ML] (2016).

    Takes the latent space features Z as input, calculates their dot product
    to return the reconstructed adjacency matrix with logits A_rec_logits.
    Sigmoid activation function is skipped as it is integrated into the binary 
    cross entropy loss for computational efficiency.

    Parameters
    ----------
    dropout
        Probability of nodes to be dropped during training.
    """
    def __init__(self, dropout: float = 0.0):
        super(DotProductDecoder, self).__init__()
        self.dropout = dropout

    def forward(self, Z):
        Z_dropout = F.dropout(Z, self.dropout, self.training)
        A_rec_logits = torch.mm(Z_dropout, Z_dropout.t())
        return A_rec_logits


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
    def __init__(self, n_input: int, n_output: int, activation = F.relu):
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
    Graph Convolutional Network layer class as per Kipf, T. N. & Welling, M.
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
    def __init__(
            self,
            n_input: int,
            n_output: int,
            dropout: float = 0.0,
            activation = torch.relu):
        super(GCNLayer, self).__init__()
        self.dropout = dropout
        self.activation = activation
        self.weights = nn.Parameter(torch.FloatTensor(n_input, n_output))
        self.initialize_weights()

    def initialize_weights(self):
        # Glorot weight initialization
        torch.nn.init.xavier_uniform_(self.weights)

    def forward(self, input: torch.Tensor, adj_mx: torch.Tensor):
        output = F.dropout(input, self.dropout, self.training)
        output = torch.mm(output, self.weights)
        output = torch.mm(adj_mx, output)
        return self.activation(output)


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