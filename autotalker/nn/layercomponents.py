"""
This module contains neural network layer components used by the Autotalker 
model.
"""

import torch
import torch.nn as nn


class MaskedLinear(nn.Linear):
    """
    Masked linear class.
    
    Parts of the implementation are adapted from
    https://github.com/theislab/scarches/blob/master/scarches/models/expimap/modules.py#L9
    (01.10.2022).

    Uses a binary mask to mask connections from the input layer to the output 
    layer so that only unmasked connections can be used.

    Parameters
    ----------
    n_input:
        Number of input nodes to the masked layer.
    n_output:
        Number of output nodes from the masked layer.
    mask:
        Mask that is used to mask the node connections from the input layer to
        the output layer.
    bias:
        If ´True´, use a bias.
    """
    def __init__(self,
                 n_input: int,
                 n_output: int,
                 mask: torch.Tensor,
                 bias=True):
        # Mask should have dim n_input x n_output
        if n_input != mask.shape[0] or n_output != mask.shape[1]:
            raise ValueError("Incorrect shape of the mask. Mask should have dim"
                             " (n_input x n_output). Please provide a mask with"
                             f" dimensions ({n_input} x {n_output}).")
        super().__init__(n_input, n_output, bias)
        
        self.register_buffer("mask", mask.t())

        # Zero out weights with the mask so that the optimizer does not consider
        # them
        if self.mask.is_sparse: # this is the case for the ca decoder
            # Convert mask to dense; in the future check whether weights could
            # be sparse to use a multiplication (sparse, sparse) instead
            self.mask = self.mask.to_dense()

        self.weight.data *= self.mask

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the masked linear class.

        Parameters
        ----------
        input:
            Tensor containing the input features to the masked linear class.

        Returns
        ----------
        output:
            Tensor containing the output of the masked linear class (linear 
            transformation of the input by only considering unmasked 
            connections).
        """
        output = nn.functional.linear(input, self.weight * self.mask, self.bias)
        return output