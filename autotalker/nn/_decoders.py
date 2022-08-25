from typing import Literal, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._layers import MaskedFCLayer


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

        print(f"DOT PRODUCT GRAPH DECODER -> dropout_rate: {dropout_rate}")

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, z):
        z = self.dropout(z)
        adj_rec_logits = torch.mm(z, z.t())
        return adj_rec_logits


class MaskedFCExprDecoder(nn.Module):
    """
    Masked fully connected expression decoder class.

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
    """
    def __init__(self,
                 n_input: int,
                 n_output: int,
                 mask: torch.Tensor,
                 dropout_rate: float=0.0):
        super().__init__()

        print(f"MASKED FULLY CONNECTED EXPRESSION DECODER -> n_input: {n_input}"
              f", n_output: {n_output}")

        self.mfc_l0 = MaskedFCLayer(
            n_input=n_input,
            n_output=n_output,
            dropout_rate=dropout_rate,
            use_batch_norm=True,
            use_layer_norm=False,
            bias=True,
            activation=nn.Softmax(dim=-1),
            mask=mask)

    def forward(self, z: torch.Tensor):
        mean_gamma = self.mfc_l0(z)
        return mean_gamma

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