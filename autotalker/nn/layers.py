from typing import Optional

import torch
import torch.nn as nn

from .layercomponents import MaskedLinear


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
    def __init__(self,
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
        """Initialize weights with Glorot weight initialization"""
        torch.nn.init.xavier_uniform_(self.weights)

    def forward(self, input: torch.Tensor, adj :torch.Tensor):
        """
        Forward pass of the GCN layer.

        Parameters
        ----------
        input:
            Tensor containing gene expression.
        adj:
            Sparse tensor containing adjacency matrix.

        Returns
        ----------
        output:
            Output of the GCN layer.
        """
        output = self.dropout(input)
        output = torch.mm(output, self.weights)
        output = torch.sparse.mm(adj, output)
        output = self.activation(output)
        return output


class MaskedLayer(nn.Module):
    """
    Masked layer class.

    Parameters
    ----------
    n_input:
        Number of input nodes to the masked layer.
    n_output:
        Number of output nodes from the masked layer.
    bias:
        If ´True´, use a bias.
    mask:
        Mask that is used to mask the node connections from the input layer to
        the output layer.
    activation:
        Activation function used at the end of the masked layer.
    """
    def __init__(self,
                 n_input: int,
                 n_output: int,
                 bias: bool=False,
                 mask: Optional[torch.Tensor]=None,
                 activation: nn.Module=nn.ReLU):
        super().__init__()

        if mask is None:
            self.mfc_l = nn.Linear(n_input, n_output, bias=bias)
        else:
            self.mfc_l = MaskedLinear(n_input, n_output, mask, bias=bias)

        self.activation = activation

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the masked layer.

        Parameters
        ----------
        input:
            Input features to the masked layer (latent space features).

        Returns
        ----------
        output:
            Output of the masked layer.
        """
        output = self.activation(self.mfc_l(input))
        return output