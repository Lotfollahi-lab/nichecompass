from typing import Optional

import torch
import torch.nn as nn

from ._layercomponents import MaskedLinear


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


class MaskedCondExtLayer(nn.Module):
    """
    Masked conditional extension layer adapted from 
    https://github.com/theislab/scarches. Takes input nodes plus optionally
    condition nodes, unmasked extension nodes and masked extension nodes and
    computes a linear transformation with masks for the masked parts.

    Parameters
    ----------
    n_input:
    n_output:
    n_condition:
    n_extension_unmasked:
    n_extension_masked:
    mask:
    extension_mask:
    """
    def __init__(self,
                 n_input: int,
                 n_output: int,
                 n_condition: int,
                 n_extension_unmasked: int,
                 n_extension_masked: int,
                 mask: Optional[torch.Tensor]=None,
                 extension_mask: Optional[torch.Tensor]=None):
        super().__init__()
        self.n_condition = n_condition
        self.n_extension_unmasked = n_extension_unmasked
        self.n_extension_masked = n_extension_masked
        
        # Creating layer components
        if mask is None:
            self.input_l = nn.Linear(n_input, n_output, bias=False)
        else:
            self.input_l = MaskedLinear(n_input,
                                             n_output,
                                             mask,
                                             bias=False)

        if self.n_condition != 0:
            self.condition_l = nn.Linear(self.n_condition,
                                          n_output,
                                          bias=False)

        if self.n_extension_unmasked != 0:
            self.extension_unmasked_l = nn.Linear(self.n_extension_unmasked,
                                                   n_output,
                                                   bias=False)

        if self.n_extension_masked != 0:
            if extension_mask is None:
                self.extension_masked_l = nn.Linear(self.n_extension_masked,
                                                     n_output,
                                                     bias=False)
            else:
                self.extension_masked_l = MaskedLinear(
                    self.n_extension_masked,
                    n_output,
                    bias=False)

    def forward(self, input: torch.Tensor):
        # Split input into its components to be fed separately into different
        # layer components
        if self.n_condition == 0:
            input, condition = input, None
        else:
            input, condition = torch.split(
                input, [input.shape[1] - self.n_condition, self.n_condition],
                dim=1)

        if self.n_extension_unmasked == 0:
            extension_unmasked = None
        else:
            input, extension_unmasked = torch.split(
                input, [input.shape[1] - self.n_extension_unmasked,
                        self.n_extension_unmasked],
                dim=1)

        if self.n_extension_masked == 0:
            extension_masked = None
        else:
            input, extension_masked = torch.split(
                input, [input.shape[1] - self.extension_masked,
                        self.extension_masked],
                dim=1)

        # Forward pass with different layer components
        output = self.input_l(input)
        if extension_unmasked is not None:
            output = output + self.extension_unmasked_l(extension_unmasked)
        if extension_masked is not None:
            output = output + self.extension_masked_l(extension_masked)
        if condition is not None:
            output = output + self.condition_l(condition)
        return output