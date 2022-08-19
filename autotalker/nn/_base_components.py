from typing import Literal, Optional

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


class GCNEncoder(nn.Module):
    """
    Graph Convolutional Network encoder class as per Kipf, T. N. & Welling, M.
    Variational Graph Auto-Encoders. arXiv [stat.ML] (2016).

    Takes the input space features x and the edge indices as input, computes one
    shared GCN layer and two separate GCN layers to output mu and logstd of the 
    latent space distribution.

    Parameters
    ----------
    n_input:
        Number of input nodes to the GCN encoder.
    n_hidden:
        Number of hidden nodes outputted by the first GCN layer.
    n_latent:
        Number of output nodes from the GCN encoder, making up the latent 
        features.
    dropout_rate:
        Probability of nodes to be dropped during training.
    activation:
        Activation function used in the first GCN layer.
    """
    def __init__(
            self,
            n_input: int,
            n_hidden: int,
            n_latent: int,
            dropout_rate: float=0.0,
            activation=torch.relu):
        super(GCNEncoder, self).__init__()
        self.gcn_l1 = GCNConv(n_input, n_hidden)
        self.gcn_mu = GCNConv(n_hidden, n_latent)
        self.gcn_logstd = GCNConv(n_hidden, n_latent)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x, edge_index):
        hidden = self.dropout(self.activation(self.gcn_l1(x, edge_index)))
        mu = self.gcn_mu(hidden, edge_index)
        logstd = self.gcn_logstd(hidden, edge_index)
        return mu, logstd


class DotProductDecoder(nn.Module):
    """
    Dot product decoder class as per Kipf, T. N. & Welling, M. Variational Graph
    Auto-Encoders. arXiv [stat.ML] (2016).

    Takes the latent space features z as input, calculates their dot product
    to return the reconstructed adjacency matrix with logits adj_rec_logits.
    Sigmoid activation function is skipped as it is integrated into the binary 
    cross entropy loss for computational efficiency.

    Parameters
    ----------
    dropout_rate:
        Probability of nodes to be dropped during training.
    """
    def __init__(self, dropout_rate: float=0.0):
        super(DotProductDecoder, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, z):
        z = self.dropout(z)
        adj_rec_logits = torch.mm(z, z.t())
        return adj_rec_logits


class GCNLayer(nn.Module):
    """
    Graph Convolutional Network layer class as per Kipf, T. N. & Welling, M.
    Semi-Supervised Classification with Graph Convolutional Networks. arXiv
    [cs.LG] (2016).

    Parameters
    ----------
    n_input:
        Number of input nodes to the GCN Layer.
    n_output:
        Number of output nodes from the GCN layer.
    dropout:
        Probability of nodes to be dropped during training.
    activation:
        Activation function used in the GCN layer.
    """
    def __init__(
            self,
            n_input: int,
            n_output: int,
            dropout_rate: float=0.0,
            activation=torch.relu):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation
        self.weights = nn.Parameter(torch.FloatTensor(n_input, n_output))
        self.initialize_weights()

    def initialize_weights(self):
        # Glorot weight initialization
        torch.nn.init.xavier_uniform_(self.weights)

    def forward(self, input, adj):
        output = self.dropout(input)
        output = torch.mm(output, self.weights)
        output = torch.sparse.mm(adj, output)
        return self.activation(output)


class FCLayer(nn.Module):
    """
    Fully connected layer class.

    Parameters
    ----------
    n_input:
        Number of input nodes to the FC Layer.
    n_output:
        Number of output nodes from the FC layer.
    activation:
        Activation function used in the FC layer.
    """
    def __init__(self, n_input: int, n_output: int, activation=torch.relu):
        super(FCLayer, self).__init__()
        self.activation = activation
        self.linear = nn.Linear(n_input, n_output)

    def forward(self, input: torch.Tensor):
        output = self.linear(input)
        if self.activation is not None:
            return self.activation(output)
        else:
            return output


class MaskedFCLayer(nn.Linear):
    """
    Adapted from https://github.com/theislab/scarches. 
    """
    def __init__(self, n_input: int, n_output: int, mask, bias=True):
        # Mask should have dim n_input x n_output
        if n_input != mask.shape[0] or n_output != mask.shape[1]:
            raise ValueError("Incorrect shape of the mask. Mask should have dim"
                             "n_input x n_output")
        super().__init__(n_input, n_output, bias)
        
        self.register_buffer("mask", mask.t())

        # Zero out weights with the mask
        # Gradient descent does not consider these zero weights
        self.weight.data *= self.mask

    def forward(self, x):
        return nn.functional.linear(input, self.weight * self.mask, self.bias)


class MaskedLinearDecoder(nn.Module):
    """
    Adapted from https://github.com/theislab/scarches. 
    """
    def __init__(
            self,
            n_input: int,
            n_output: int,
            mask,
            recon_loss: Literal["mse", "nb"],
            last_layer_activation: Optional[Literal["softmax", "relu"]]=None):
        super().__init__()

        if recon_loss == "mse":
            if last_layer_activation == "softmax":
                raise ValueError("Can't specify softmax last layer activation "
                                 "with mse loss.")
            last_layer_activation = ("identity" if last_layer_activation is None
                                     else last_layer_activation) 
        elif recon_loss == "nb":
            last_layer_activation = ("softmax" if last_layer_activation is None 
                                     else last_layer_activation)
        else:
            raise ValueError("Unrecognized loss. Specify either 'mse' or 'nb' "
                             "as reconstruction loss")

        self.mfc_l1 = MaskedFCLayer(n_input, n_output, mask, bias=False)
        if last_layer_activation == "softmax":
            self.act_l1 = nn.Softmax(dim=-1)
        elif last_layer_activation == "softplus":
            self.act_l1 = nn.Softplus()
        elif last_layer_activation == "exp":
            self.act_l1 = torch.exp
        elif last_layer_activation == "relu":
            self.act_l1 = nn.ReLU()
        elif last_layer_activation == "identity":
            self.act_l1 = lambda a: a
        else:
            raise ValueError("Unrecognized last layer activation function.")

    def forward(self, z):
        x_recon = self.act_l1(self.mfc_l1(z))
        return x_recon
