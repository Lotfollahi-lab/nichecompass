"""
This module contains neural network layer components used by the NicheCompass 
model.
"""

from typing import Optional

import torch
import torch.nn as nn


class MaskedLinear(nn.Linear):
    """
    Masked linear class.
    
    Parts of the implementation are adapted from
    https://github.com/theislab/scarches/blob/master/scarches/models/expimap/modules.py#L9;
    01.10.2022.

    Uses static and dynamic binary masks to mask connections from the input
    layer to the output layer so that only unmasked connections can be used.

    Parameters
    ----------
    n_input:
        Number of input nodes to the masked layer.
    n_output:
        Number of output nodes from the masked layer.
    mask:
        Static mask that is used to mask the node connections from the input
        layer to the output layer.
    bias:
        If ´True´, use a bias.
    """
    def __init__(self,
                 n_input: int,
                 n_output: int,
                 mask: torch.Tensor,
                 bias=False):
        # Mask should have dim n_input x n_output
        if n_input != mask.shape[0] or n_output != mask.shape[1]:
            raise ValueError("Incorrect shape of the mask. Mask should have dim"
                            " (n_input x n_output). Please provide a mask with"
                            f"  dimensions ({n_input} x {n_output}).")
        super().__init__(n_input, n_output, bias)

        self.register_buffer("mask", mask.t())

        # Zero out weights with the mask so that the optimizer does not
        # consider them
        self.weight.data *= self.mask

    def forward(self,
                input: torch.Tensor,
                dynamic_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Forward pass of the masked linear class.

        Parameters
        ----------
        input:
            Tensor containing the input features to the masked linear class.
        dynamic_mask:
            Additional optional Tensor containing a mask that changes
            during training.

        Returns
        ----------
        output:
            Tensor containing the output of the masked linear class (linear 
            transformation of the input by only considering unmasked 
            connections).
        """
        if dynamic_mask is not None:
            dynamic_mask = dynamic_mask.t().to(self.mask.device)
            self.weight.data *= dynamic_mask
            masked_weights = self.weight * self.mask * dynamic_mask
        else:
            masked_weights = self.weight * self.mask
        output = nn.functional.linear(input, masked_weights, self.bias)
        return output
    