"""
This module contains (full) neural network layers used by the NicheCompass model.
"""


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
                 dropout_rate: float=0.,
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

    def forward(self, input: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
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
    Add-on masked layer class. 
    
    Parts of the implementation are adapted from 
    https://github.com/theislab/scarches/blob/7980a187294204b5fb5d61364bb76c0b809eb945/scarches/models/expimap/modules.py#L28
    (01.10.2022).

    Parameters
    ----------
    n_input:
        Number of mask input nodes to the masked layer.
    n_output:
        Number of output nodes from the masked layer.
    bias:
        If ´True´, use a bias for the mask input nodes.
    mask:
        Mask that is used to mask the node connections for mask inputs from the
        input layer to the output layer.
    n_addon_input:
        Number of add-on input nodes to the masked layer.
    n_cat_covariates_embed_input:
        Number of categorical covariates embedding input nodes to the masked
        layer.
    activation:
        Activation function used at the end of the masked layer.
    """
    def __init__(self,
                 n_input: int,
                 n_output: int,
                 unmasked_features_idx: list,
                 bias: bool=False,
                 mask: Optional[torch.Tensor]=None,
                 n_addon_input: int=0,
                 n_cat_covariates_embed_input: int=0,
                 activation: nn.Module=nn.Softmax(dim=-1)):
        super().__init__()
        self.n_input = n_input
        self.n_addon_input = n_addon_input
        self.n_cat_covariates_embed_input = n_cat_covariates_embed_input
        self.unmasked_features_idx = unmasked_features_idx

        # Masked layer
        if mask is None:
            self.masked_l = nn.Linear(n_input, n_output, bias=bias)
        else:
            self.masked_l = MaskedLinear(n_input, n_output, mask, bias=bias)

        # Add-on layer
        if n_addon_input != 0:
            self.addon_l = nn.Linear(n_addon_input, n_output, bias=False)
        
        # Categorical covariates embedding layer
        if n_cat_covariates_embed_input != 0:
            self.cat_covariates_embed_l = nn.Linear(
                n_cat_covariates_embed_input,
                n_output,
                bias=False)

        self.activation = activation

    def forward(self,
                input: torch.Tensor,
                dynamic_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Forward pass of the add-on masked layer.

        Parameters
        ----------
        input:
            Input features to the add-on masked layer. Includes add-on input
            nodes and categorical covariates embedding input nodes if specified.
        dynamic_mask:
            Additional optional dynamic mask for the masked layer.

        Returns
        ----------
        output:
            Output of the add-on masked layer.
        """
        if (self.n_addon_input == 0) & (self.n_cat_covariates_embed_input == 0):
            mask_input = input
        elif ((self.n_addon_input != 0) &
              (self.n_cat_covariates_embed_input == 0)):
            mask_input, addon_input = torch.split(
                input,
                [self.n_input, self.n_addon_input],
                dim=1)
        elif ((self.n_addon_input == 0) &
              (self.n_cat_covariates_embed_input != 0)):
            mask_input, cat_covariates_embed_input = torch.split(
                input,
                [self.n_input, self.n_cat_covariates_embed_input],
                dim=1)          
        elif ((self.n_addon_input != 0) &
              (self.n_cat_covariates_embed_input != 0)):
            mask_input, addon_input, cat_covariates_embed_input = torch.split(
                input,
                [self.n_input, self.n_addon_input, self.n_cat_covariates_embed_input],
                dim=1)

        output = self.masked_l(input=mask_input,
                               dynamic_mask=dynamic_mask)
        if self.n_addon_input != 0:
            # Only add addon layer output to unmasked features
            output[:self.unmasked_features_idx] += self.addon_l(addon_input)[
                :self.unmasked_features_idx]
        if self.n_cat_covariates_embed_input != 0:
            if self.n_addon_input != 0:
                output += self.cat_covariates_embed_l(
                    cat_covariates_embed_input)
            else:
                output[:self.unmasked_features_idx] += self.addon_l(addon_input)[
                    :self.unmasked_features_idx]                
        output = self.activation(output)
        return output