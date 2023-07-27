"""
This module contains decoders used by the NicheCompass model.
"""

from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn

from .layers import AddOnMaskedLayer
from .utils import compute_cosine_similarity
    

class CosineSimGraphDecoder(nn.Module):
    """
    Cosine similarity graph decoder class.

    Takes the concatenated latent feature vectors z of the source and
    destination nodes as input, and calculates the element-wise cosine
    similarity between source and destination nodes to return the reconstructed
    edge logits. Sigmoid activation function to compute reconstructed edge
    probabilities is integrated into the binary cross entropy loss for
    computational efficiency.

    Parameters
    ----------
    dropout_rate:
        Probability of nodes to be dropped during training.
    """
    def __init__(self,
                 dropout_rate: float=0.):
        super().__init__()

        print(f"COSINE SIM GRAPH DECODER -> dropout_rate: {dropout_rate}")

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the cosine similarity graph decoder.

        Parameters
        ----------
        z:
            Concatenated latent feature vector of the source and destination
            nodes (dim: 4 * edge_batch_size x n_gps due to negative edges).

        Returns
        ----------
        edge_recon_logits:
            Reconstructed edge logits (dim: 2 * edge_batch_size due to negative
            edges).
        """
        z = self.dropout(z)

        # Compute element-wise cosine similarity
        edge_recon_logits = compute_cosine_similarity(
            z[:int(z.shape[0]/2)], # ´edge_label_index[0]´
            z[int(z.shape[0]/2):]) # ´edge_label_index[1]´
        return edge_recon_logits
    

class MaskedOmicsFeatureDecoder(nn.Module):
    """
    Masked omics feature decoder class.

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
    n_cat_covariates_embed_input:
        Number of categorical covariates embedding input nodes to the decoder
        (categorical covariates embedding dimensionality).
    n_output:
        Number of output nodes from the decoder (number of genes).
    mask:
        Mask that determines which input nodes / latent features can contribute
        to the reconstruction of which genes.
    mask_idx:
        Index of genes that are in the gp mask.
    gene_expr_recon_dist:
        The distribution used for gene expression reconstruction. If `nb`, uses
        a Negative Binomial distribution. If `zinb`, uses a Zero-inflated
        Negative Binomial distribution.
    """
    def __init__(self,
                 mod: Literal["rna", "atac"],
                 n_prior_gp_input: int,
                 n_addon_gp_input: int,
                 n_cat_covariates_embed_input: int,
                 n_output: int,
                 mask: torch.Tensor,
                 masked_features_idx: torch.Tensor,
                 unmasked_features_idx: torch.Tensor,
                 recon_loss: Literal["nb"]):
        super().__init__()

        if mod == "rna":
            print(f"MASKED RNA DECODER -> n_prior_gp_input: {n_prior_gp_input}, "
                  f"n_cat_covariates_embed_input: {n_cat_covariates_embed_input}, "
                  f"n_addon_gp_input: {n_addon_gp_input}, n_output: {n_output}")
        elif mod == "atac":
            print(f"MASKED ATAC DECODER -> n_prior_gp_input: {n_prior_gp_input}, "
                  f"n_cat_covariates_embed_input: {n_cat_covariates_embed_input}, "
                  f"n_addon_gp_input: {n_addon_gp_input}, n_output: {n_output}")
            
        self.masked_features_idx = masked_features_idx
        self.unmasked_features_idx = unmasked_features_idx
        self.recon_loss = recon_loss

        self.nb_means_normalized_decoder = AddOnMaskedLayer(
            n_input=n_prior_gp_input,
            n_addon_input=n_addon_gp_input,
            n_cat_covariates_embed_input=n_cat_covariates_embed_input,
            n_output=n_output,
            bias=False,
            mask=mask,
            unmasked_features_idx=unmasked_features_idx,
            activation=nn.Softmax(dim=-1))

    def forward(self,
                z: torch.Tensor,
                log_library_size: torch.Tensor,
                dynamic_mask: Optional[torch.Tensor]=None,
                cat_covariates_embed: Optional[torch.Tensor]=None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the masked gene expression decoder.

        Parameters
        ----------
        z:
            Tensor containing the latent space features.
        log_library_size:
            Tensor containing the log library size of the nodes.
        dynamic_mask:
            For atac modality, a tensor containing a dynamic peak mask based on gene
            weights. If a gene is removed by regularization (its weight is 0),
            the corresponding peaks will be marked as 0 in the
            `gene_weight_peak_mask`.        
        cat_covariates_embed:
            Tensor containing the categorical covariates embedding (all
            categorical covariates embeddings concatenated into one embedding).

        Returns
        ----------
        nb_means:
        """            
        # Add categorical covariates embedding to latent feature vector
        if cat_covariates_embed is not None:
            z = torch.cat((z, cat_covariates_embed), dim=-1)
        
        nb_means_normalized = self.nb_means_normalized_decoder(
            input=z,
            dynamic_mask=dynamic_mask)
        nb_means = torch.exp(log_library_size) * nb_means_normalized
        return nb_means