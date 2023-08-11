"""
This module contains neural network layers used by the NicheCompass model.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from .layercomponents import MaskedLinear


class AddOnMaskedLayer(nn.Module):
    """
    Add-on masked layer class. 
    
    Parts of the implementation are adapted from 
    https://github.com/theislab/scarches/blob/7980a187294204b5fb5d61364bb76c0b809eb945/scarches/models/expimap/modules.py#L28;
    01.10.2022.

    Parameters
    ----------
    n_input:
        Number of mask input nodes to the add-on masked layer.
    n_output:
        Number of output nodes from the add-on masked layer.
    mask:
        Mask that is used to mask the node connections for mask inputs from the
        input layer to the output layer.
    addon_mask:
        Mask that is used to mask the node connections for add-on inputs from
        the input layer to the output layer.
    masked_features_idx:
        Index of input features that are included in the mask.
    bias:
        If ´True´, use a bias for the mask input nodes.
    n_addon_input:
        Number of add-on input nodes to the add-on masked layer.
    n_cat_covariates_embed_input:
        Number of categorical covariates embedding input nodes to the addon
        masked layer.
    activation:
        Activation function used at the end of the ad-on masked layer.
    """
    def __init__(self,
                 n_input: int,
                 n_output: int,
                 mask: torch.Tensor,
                 addon_mask: torch.Tensor,
                 masked_features_idx: List,
                 bias: bool=False,
                 n_addon_input: int=0,
                 n_cat_covariates_embed_input: int=0,
                 activation: nn.Module=nn.Softmax(dim=-1)):
        super().__init__()
        self.n_input = n_input
        self.n_addon_input = n_addon_input
        self.n_cat_covariates_embed_input = n_cat_covariates_embed_input
        self.masked_features_idx = masked_features_idx

        # Masked layer
        self.masked_l = MaskedLinear(n_input=n_input,
                                     n_output=n_output,
                                     mask=mask,
                                     bias=bias)

        # Add-on layer
        if n_addon_input != 0:
            self.addon_l = MaskedLinear(n_input=n_addon_input,
                                        n_output=n_output,
                                        mask=addon_mask,
                                        bias=False)
        
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
            
        output = self.masked_l(
            input=mask_input,
            dynamic_mask=(dynamic_mask[:self.n_input, :] if
                          dynamic_mask is not None else None)) 
            # Dynamic mask also has entries for add-on gps
        if self.n_addon_input != 0:
            # Only unmasked features will have weights != 0.
            output += self.addon_l(
                input=addon_input,
                dynamic_mask=(dynamic_mask[self.n_input:, :] if
                              dynamic_mask is not None else None))
        if self.n_cat_covariates_embed_input != 0:
            if self.n_addon_input != 0:
                output += self.cat_covariates_embed_l(
                    cat_covariates_embed_input)
            else:
                # Only add categorical covariates embedding layer output to
                # masked features
                output[:, self.masked_features_idx] += self.cat_covariates_embed_l(
                    cat_covariates_embed_input)[:, self.masked_features_idx]                
        output = self.activation(output)
        return output
    