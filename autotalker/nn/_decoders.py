from typing import Literal

import torch
import torch.nn as nn

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
    n_input:
        Number of input nodes to the decoder (latent space dimensionality).
    n_output:
        Number of output nodes from the decoder (feature space x dimensionality).
    mask:
        Mask that determines which input nodes / latent features can contribute
        to the reconstruction of which genes.
    recon_loss:
        Loss used for the reconstruction.
    """
    def __init__(self,
                 n_input: int,
                 n_output: int,
                 mask: torch.Tensor,
                 recon_loss: Literal["nb", "mse"]):
        super().__init__()

        print(f"MASKED LINEAR EXPRESSION DECODER - n_input: {n_input}, n_output"
              f": {n_output}, recon_loss: {recon_loss}")

        self.mfc_l1 = MaskedFCLayer(n_input, n_output, mask, bias=False)

        if recon_loss == "nb":
            self.act_l1 = nn.Softmax(dim=-1) # softmax activation
        elif recon_loss == "mse":
            self.act_l1 = lambda a: a # identity activation

    def forward(self, z):
        x_recon = self.act_l1(self.mfc_l1(z))
        return x_recon