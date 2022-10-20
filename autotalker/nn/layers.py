from re import A
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


class AddOnMaskedLayer(nn.Module):
    """
    Add-on masked layer class. Adapted from 
    https://github.com/theislab/scarches/blob/7980a187294204b5fb5d61364bb76c0b809eb945/scarches/models/expimap/modules.py#L28.

    Parameters
    ----------
    n_input:
        Number of maskable input nodes to the masked layer.
    n_output:
        Number of output nodes from the masked layer.
    bias:
        If ´True´, use a bias for the maskable input nodes.
    mask:
        Mask that is used to mask the node connections from the input layer to
        the output layer.
    n_addon_input:
        Number of non-maskable add-on input nodes to the masked layer.
    activation:
        Activation function used at the end of the masked layer.
    """
    def __init__(self,
                 n_input: int,
                 n_output: int,
                 bias: bool=False,
                 mask: Optional[torch.Tensor]=None,
                 n_addon_input: int=0,
                 activation: nn.Module=nn.ReLU):
        super().__init__()
        self.n_addon_input = n_addon_input

        # Masked layer
        if mask is None:
            self.masked_l = nn.Linear(n_input, n_output, bias=bias)
        else:
            self.masked_l = MaskedLinear(n_input, n_output, mask, bias=bias)

        # Add-on layer
        if n_addon_input != 0:
            self.addon_l = nn.Linear(n_addon_input, n_output, bias=False)

        self.activation = activation

    def forward(self, input: torch.Tensor):
        """
        Forward pass of the add-on masked layer.

        Parameters
        ----------
        input:
            Input features to the add-on masked layer. Includes add-on input
            nodes if specified.

        Returns
        ----------
        output:
            Output of the add-on masked layer.
        """
        if self.n_addon_input == 0:
            maskable_input = input
        else:
            maskable_input, addon_input = torch.split(
                input,
                [input.shape[1] - self.n_addon_input, self.n_addon_input],
                dim=1)

        output = self.masked_l(maskable_input)
        if self.n_addon_input != 0:
            output = output + self.addon_l(addon_input)
        output = self.activation(output)

        return output