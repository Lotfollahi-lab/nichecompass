import torch
import torch.nn as nn
import torch.nn.functional as F

from ._layers import MaskedLayer


class DotProductGraphDecoder(nn.Module):
    """
    Dot product graph decoder class as per Kipf, T. N. & Welling, M. Variational 
    Graph Auto-Encoders. arXiv [stat.ML] (2016).

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

    def forward(self, z: torch.Tensor):
        """
        Forward pass of the dot product graph decoder.

        Parameters
        ----------
        z:
            Tensor containing the latent space features.

        Returns
        ----------
        adj_rec_logits:
            Tensor containing the reconstructed adjacency matrix with logits.
        """
        z = self.dropout(z)
        adj_rec_logits = torch.mm(z, z.t())
        return adj_rec_logits


class MaskedGeneExprDecoder(nn.Module):
    """
    Masked gene expression decoder class.

    Takes the latent space features z as input, and has two separate masked
    layers to decode the parameters of the ZINB distribution.

    Parameters
    ----------
    n_input:
        Number of input nodes to the decoder (latent space dimensionality).
    n_output:
        Number of output nodes from the decoder (number of genes).
    mask:
        Mask that determines which input nodes / latent features can contribute
        to the reconstruction of which genes.
    """
    def __init__(self,
                 n_input: int,
                 n_output: int,
                 mask: torch.Tensor):
        super().__init__()

        print(f"MASKED GENE EXPRESSION DECODER -> n_input: {n_input}"
              f", n_output: {n_output}")

        self.nb_means_normalized_decoder = MaskedLayer(
            n_input=n_input,
            n_output=n_output,
            bias=False,
            activation=nn.Softmax(dim=-1),
            mask=mask)

        self.zi_prob_logits_decoder = MaskedLayer(
            n_input=n_input,
            n_output=n_output,
            bias=False,
            activation=nn.ReLU(),
            mask=mask)

    def forward(self, z: torch.Tensor, log_library_size: torch.Tensor):
        """
        Forward pass of the masked gene expression decoder.

        Parameters
        ----------
        z:
            Tensor containing the latent space features.
        log_library_size:
            Tensor containing the log library size of the nodes.

        Returns
        ----------
        zinb_parameters:
            Parameters for the ZINB distribution to model gene expression.
        """
        nb_means_normalized = self.nb_means_normalized_decoder(z)
        nb_means = torch.exp(log_library_size) * nb_means_normalized # log?
        zi_prob_logits = self.zi_prob_logits_decoder(z)
        zinb_parameters = (nb_means, zi_prob_logits)
        return zinb_parameters