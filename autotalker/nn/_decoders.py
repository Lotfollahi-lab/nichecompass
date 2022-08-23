from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._layers import MaskedCondExtLayer


class DotProductGraphDecoder(nn.Module):
    """
    Dot product decoder class as per Kipf, T. N. & Welling, M. Variational Graph
    Auto-Encoders. arXiv [stat.ML] (2016).

    Takes the latent space features z as input, calculates their dot product
    to return the reconstructed adjacency matrix with logits `adj_rec_logits`.
    Sigmoid activation function is skipped as it is integrated into the binary 
    cross entropy loss for computational efficiency.

    Parameters
    ----------
    dropout_rate:
        Probability of nodes to be dropped during training.
    """
    def __init__(self, dropout_rate: float=0.0):
        super().__init__()

        print(f"DOT PRODUCT GRAPH DECODER - dropout_rate: {dropout_rate}")

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, z):
        z = self.dropout(z)
        adj_rec_logits = torch.mm(z, z.t())
        return adj_rec_logits


class MaskedLinearExprDecoder(nn.Module):
    """
    Masked linear expression decoder class adapted from 
    https://github.com/theislab/scarches. 

    Takes the latent space features z as input, transforms them with a masked
    FC layer and returns the reconstructed input space features x.

    Parameters
    ----------
    n_inputs:
        Number of input nodes to the decoder (latent space dimensionality).
    n_outputs:
        Number of output nodes from the decoder (feature space x dimensionality).
    mask:
        Mask that determines which input nodes / latent features can contribute
        to the reconstruction of which genes.
    recon_loss:
        Loss used for the reconstruction.
    """
    def __init__(self,
                 n_inputs: int,
                 n_outputs: int,
                 mask: torch.Tensor,
                 extension_mask: Optional[torch.Tensor]=None,
                 n_conditions: int=0,
                 n_extensions_unmasked: int=0,
                 n_extensions_masked: int=0,
                 recon_loss: Literal["nb", "mse"]="nb"):
        super().__init__()
        self.n_extensions_unmasked = n_extensions_unmasked
        self.n_extensions_masked = n_extensions_masked
        self.n_conditions = n_conditions

        print(f"MASKED LINEAR EXPRESSION DECODER - n_inputs: {n_inputs}, n_outputs"
              f": {n_outputs}, n_conditions: {n_conditions}, "
              f"n_extensions_unmasked: {n_extensions_unmasked}, "
              f"n_extensions_masked: {n_extensions_masked}, recon_loss: "
              f"{recon_loss}")

        self.mce_l0 = MaskedCondExtLayer(
            n_inputs=n_inputs,
            n_outputs=n_outputs,
            n_conditions=n_conditions,
            n_extensions_unmasked=n_extensions_unmasked,
            n_extensions_masked=n_extensions_masked,
            mask=mask,
            extension_mask=extension_mask)

        if recon_loss == "nb":
            self.activation_l0 = nn.Softmax(dim=-1) # softmax activation
        elif recon_loss == "mse":
            self.activation_l0 = lambda a: a # identity activation

    def forward(self, inputs, conditions=None):
        if conditions is not None:
            # Add one-hot-encoded condition nodes to inputs
            conditions_ohe = F.one_hot(conditions, self.n_conditions)
            inputs = torch.cat((inputs, conditions_ohe), dim=-1)
        decoder_latents = self.mce_l0(inputs)    
        outputs = self.activation_l0(decoder_latents)
        return outputs, decoder_latents

    def non_zero_gene_program_node_mask(self):
        gp_nodes_weights = self.mce_l0.input_l.weight.data
        # Check if sum of absolute values of gp node weights > 0 (per node)
        non_zero_mask = (gp_nodes_weights.norm(p=1, dim=0) > 0).cpu().numpy()
        non_zero_mask = np.append(non_zero_mask,
                                  np.full(self.n_extensions_unmasked, True))
        non_zero_mask = np.append(non_zero_mask,
                                  np.full(self.n_extensions_masked, True))
        return non_zero_mask

    def n_inactive_gene_program_nodes(self):
        n = (~self.non_zero_gene_program_node_mask()).sum()
        return int(n)