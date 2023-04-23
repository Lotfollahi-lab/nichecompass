"""
This module contains decoders used by the Autotalker model.
"""

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from .layers import AddOnMaskedLayer
from .utils import compute_cosine_similarity


class DotProductGraphDecoder(nn.Module):
    """
    Dot product graph decoder class.

    Takes the concatenated latent feature vectors z of the source and
    destination nodes as input, and calculates the element-wise dot product
    between source and destination nodes to return the reconstructed edge
    logits. Sigmoid activation function to compute reconstructed edge
    probabilities is integrated into the binary cross entropy loss for
    computational efficiency. Optionally, takes a conditional embedding as input
    and adds it to the concatenated latent feature vectors z.

    Parameters
    ----------
    n_cond_embed_input:
        Dimensionality of the conditional embedding.
    n_cond_embed_output:
        Dimensionality of the latent feature vectors.
    dropout_rate:
        Probability of nodes to be dropped during training.
    """
    def __init__(self,
                 n_cond_embed_input: int,
                 n_cond_embed_output: int,
                 dropout_rate: float=0.):
        super().__init__()

        print(f"DOT PRODUCT GRAPH DECODER -> n_cond_embed_input: "
              f"{n_cond_embed_input}, n_cond_embed_output: "
              f"{n_cond_embed_output}, dropout_rate: {dropout_rate}")

        # Conditional embedding layer
        if n_cond_embed_input != 0:
            self.cond_embed_l = nn.Linear(n_cond_embed_input,
                                          n_cond_embed_output,
                                          bias=False)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                z: torch.Tensor,
                cond_embed: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the dot product graph decoder.

        Parameters
        ----------
        z:
            Concatenated latent feature vector of the source and destination
            nodes (dim: 4 * edge_batch_size x n_gps due to negative edges).
        cond_embed:
            Concatenated conditional embedding vector of the source and
            destination nodes (dim: 4 * edge_batch_size x n_cond_embed).

        Returns
        ----------
            Reconstructed edge logits (dim: 2 * edge_batch_size due to negative
            edges).
        """
        # Add conditional embedding to latent feature vector
        if cond_embed is not None:
            z += self.cond_embed_l(cond_embed)
        
        z = self.dropout(z)

        # Compute element-wise cosine similarity
        edge_recon_logits = compute_cosine_similarity(
            z[:int(z.shape[0]/2)], # ´edge_label_index[0]´
            z[int(z.shape[0]/2):]) # ´edge_label_index[1]´
        return edge_recon_logits
    

class CosineSimGraphDecoder(nn.Module):
    """
    Cosine similarity graph decoder class.

    Takes the concatenated latent feature vectors z of the source and
    destination nodes as input, and calculates the element-wise cosine
    similarity between source and destination nodes to return the reconstructed
    edge logits. Sigmoid activation function to compute reconstructed edge
    probabilities is integrated into the binary cross entropy loss for
    computational efficiency. Optionally, takes a conditional embedding as input
    and adds it to the concatenated latent feature vectors z.

    Parameters
    ----------
    n_cond_embed_input:
        Dimensionality of the conditional embedding.
    n_cond_embed_output:
        Dimensionality of the latent feature vectors.
    dropout_rate:
        Probability of nodes to be dropped during training.
    """
    def __init__(self,
                 n_cond_embed_input: int,
                 n_cond_embed_output: int,
                 dropout_rate: float=0.):
        super().__init__()

        print(f"COSINE SIM GRAPH DECODER -> n_cond_embed_input: "
              f"{n_cond_embed_input}, n_cond_embed_output: "
              f"{n_cond_embed_output}, dropout_rate: {dropout_rate}")

        # Conditional embedding layer
        if n_cond_embed_input != 0:
            self.cond_embed_l = nn.Linear(n_cond_embed_input,
                                          n_cond_embed_output,
                                          bias=False)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                z: torch.Tensor,
                cond_embed: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass of the cosine similarity graph decoder.

        Parameters
        ----------
        z:
            Concatenated latent feature vector of the source and destination
            nodes (dim: 4 * edge_batch_size x n_gps due to negative edges).
        cond_embed:
            Concatenated conditional embedding vector of the source and
            destination nodes (dim: 4 * edge_batch_size x n_cond_embed).

        Returns
        ----------
        edge_recon_logits:
            Reconstructed edge logits (dim: 2 * edge_batch_size due to negative
            edges).
        """
        # Add conditional embedding to latent feature vectors
        if cond_embed is not None:
            z += self.cond_embed_l(cond_embed)
        
        z = self.dropout(z)

        # Compute element-wise cosine similarity
        edge_recon_logits = compute_cosine_similarity(
            z[:int(z.shape[0]/2)], # ´edge_label_index[0]´
            z[int(z.shape[0]/2):]) # ´edge_label_index[1]´
        return edge_recon_logits


class MaskedGeneExprDecoder(nn.Module):
    """
    Masked gene expression decoder class.

    Takes the latent space features z as input, and has two separate masked
    layers to decode the parameters of the gene expression distribution.

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
                 n_cond_embed_input: int,
                 n_addon_input: int,
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
    

class MaskedChromAccessDecoder(nn.Module):
    """
    Masked chromatin accessibility decoder class.

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

        print("MASKED CHROMATIN ACCESSIBILITY DECODER -> n_input: "
                f"{n_input}, n_cond_embed_input: {n_cond_embed_input}, "
                f"n_addon_input: {n_addon_input}, n_output: {n_output}")

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
        Forward pass of the masked chromatin accessibility decoder.

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
            chrom_access_decoder_params = nb_means
        elif self.recon_dist == "zinb":
            zi_prob_logits = self.zi_prob_logits_decoder(z)
            zi_prob_logits = zi_prob_logits[:, self.genes_idx]
            chrom_access_decoder_params = (nb_means, zi_prob_logits)
        return chrom_access_decoder_params