"""
This module contains decoders used by the Autotalker model.
"""

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from .layers import AddOnMaskedLayer


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
    def __init__(self,
                 n_input: int,
                 n_addon_input: int,
                 n_cond_embed_input: int,
                 n_output: int,
                 include_linear_l: bool=True,
                 dropout_rate: float=0.):
        super().__init__()

        self.include_linear_l = include_linear_l

        print(f"DOT PRODUCT GRAPH DECODER -> include_linear_l: "
              f"{include_linear_l}, n_input: {n_input}, n_addon_input:"
              f" {n_addon_input}, n_cond_embed_input: {n_cond_embed_input}, "
              f"n_output: {n_output}, dropout_rate: {dropout_rate}")

        if include_linear_l:
            if n_addon_input != 0:
                n_input += n_addon_input
            if n_cond_embed_input != 0:
                n_input += n_cond_embed_input
            self.linear_l = nn.Linear(n_input, n_output, bias=False)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                z: torch.Tensor,
                cond_embed: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the dot product graph decoder.

        Parameters
        ----------
        z:
            Tensor containing the latent space features.
        cond_embed:
            Tensor containing the conditional embedding.

        Returns
        ----------
        adj_rec_logits:
            Tensor containing the reconstructed adjacency matrix with logits.
        """
        # Add conditional embedding to latent feature vector
        if cond_embed is not None:
            z = torch.cat((z, cond_embed), dim=-1)
        
        if self.include_linear_l:
            z = self.linear_l(z)
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
    n_addon_input:
        Number of non-maskable add-on input nodes to the decoder (non-maskable
        latent space dimensionality).
    n_cond_embed_input:
        Number of conditional embedding input nodes to the decoder (conditional
        embedding dimensionality).
    n_output:
        Number of output nodes from the decoder (number of genes).
    mask:
        Mask that determines which input nodes / latent features can contribute
        to the reconstruction of which genes.
    genes_idx:
        Index of genes that are in the gp mask.
    gene_expr_recon_dist:
        The distribution used for gene expression reconstruction. If `nb`, uses
        a Negative Binomial distribution. If `zinb`, uses a Zero-inflated
        Negative Binomial distribution.
    """
    def __init__(self,
                 n_input: int,
                 n_addon_input: int,
                 n_cond_embed_input: int,
                 n_output: int,
                 mask: torch.Tensor,
                 genes_idx: torch.Tensor,
                 recon_dist: Literal["nb", "zinb"]):
        super().__init__()

        print(f"MASKED GENE EXPRESSION DECODER -> n_input: {n_input}, "
              f"n_cond_embed_input: {n_cond_embed_input}, n_addon_input: "
              f"{n_addon_input}, n_output: {n_output}")

        self.genes_idx = genes_idx
        self.recon_dist = recon_dist

        self.nb_means_normalized_decoder = AddOnMaskedLayer(
            n_input=n_input,
            n_output=n_output,
            bias=False,
            mask=mask,
            n_addon_input=n_addon_input,
            n_cond_embed_input=n_cond_embed_input,
            activation=nn.Softmax(dim=-1))

        if recon_dist == "zinb":
            self.zi_prob_logits_decoder = AddOnMaskedLayer(
                n_input=n_input,
                n_output=n_output,
                bias=False,
                mask=mask,
                n_addon_input=n_addon_input,
                n_cond_embed_input=n_cond_embed_input,
                activation=nn.Identity())

    def forward(self,
                z: torch.Tensor,
                log_library_size: torch.Tensor,
                cond_embed: Optional[torch.Tensor]=None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the masked gene expression decoder.

        Parameters
        ----------
        z:
            Tensor containing the latent space features.
        log_library_size:
            Tensor containing the log library size of the nodes.
        cond_embed:
            Tensor containing the conditional embedding.

        Returns
        ----------
        zinb_parameters:
            Parameters for the ZINB distribution to model gene expression.
        """
        # Add conditional embedding to latent feature vector
        if cond_embed is not None:
            z = torch.cat((z, cond_embed), dim=-1)
        
        nb_means_normalized = self.nb_means_normalized_decoder(z)
        
        nb_means = torch.exp(log_library_size) * nb_means_normalized
        nb_means = nb_means[:, self.genes_idx]
        if self.recon_dist == "nb":
            gene_expr_decoder_params = nb_means
        elif self.recon_dist == "zinb":
            zi_prob_logits = self.zi_prob_logits_decoder(z)
            zi_prob_logits = zi_prob_logits[:, self.genes_idx]
            gene_expr_decoder_params = (nb_means, zi_prob_logits)
        return gene_expr_decoder_params