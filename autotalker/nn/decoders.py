import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import AddOnMaskedLayer


class FCGraphDecoder(nn.Module):
    """
    Fully connected graph decoder class.
    """
    def __init__(self,
                 n_input: int,
                 n_output: int,
                 bias: bool=False,
                 dropout_rate: float=0.0,
                 activation: nn.Module=nn.Identity):
        super().__init__()

        print(f"FULLY CONNECTED GRAPH DECODER -> dropout_rate: {dropout_rate}")

        self.linear = nn.Linear(n_input * 2, n_output, bias=bias)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, z: torch.Tensor):
        """
        Forward pass of the fully connected graph decoder.

        Parameters
        ----------
        z:
            Tensor containing the latent space features.

        Returns
        ----------
        adj_rec_logits:
            Tensor containing the reconstructed adjacency matrix with logits.
        """
        z_cat = torch.cat((z, z.t()), dim=-1)
        print(z_cat.shape)
        z_cat = self.linear(self.dropout(z_cat))
        adj_rec_logits = self.activation(z_cat)
        return adj_rec_logits


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
        Number of maskable input nodes to the decoder (maskable latent space 
        dimensionality).
    n_output:
        Number of output nodes from the decoder (number of genes).
    mask:
        Mask that determines which input nodes / latent features can contribute
        to the reconstruction of which genes.
    n_addon_input:
        Number of non-maskable add-on input nodes to the decoder (non-maskable
        latent space dimensionality)
    """
    def __init__(self,
                 n_input: int,
                 n_output: int,
                 mask: torch.Tensor,
                 n_addon_input: int):
        super().__init__()

        print(f"MASKED GENE EXPRESSION DECODER -> n_input: {n_input}, "
              f"n_addon_input: {n_addon_input}, n_output: {n_output}")

        """
        self.shared_decoder = AddOnMaskedLayer(
            n_input=n_input,
            n_output=n_output,
            bias=False,
            mask=mask,
            n_addon_input=n_addon_input,
            activation=nn.Identity())

        self.nb_means_normalized_decoder = nn.Sequential(
            self.shared_decoder,
            nn.Softmax(dim=-1))

        self.zi_prob_logits_decoder = nn.Sequential(
            self.shared_decoder,
            nn.Identity())
        """

        self.nb_means_normalized_decoder = AddOnMaskedLayer(
            n_input=n_input,
            n_output=n_output,
            bias=False,
            mask=mask,
            n_addon_input=n_addon_input,
            activation=nn.Softmax(dim=-1))

        self.zi_prob_logits_decoder = AddOnMaskedLayer(
            n_input=n_input,
            n_output=n_output,
            bias=False,
            mask=mask,
            n_addon_input=n_addon_input,
            activation=nn.Identity())

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
        nb_means = torch.exp(log_library_size) * nb_means_normalized
        zi_prob_logits = self.zi_prob_logits_decoder(z)
        zinb_parameters = (nb_means, zi_prob_logits)
        return zinb_parameters