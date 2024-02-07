"""
This module contains decoders used by the NicheCompass model.
"""

from typing import Literal, List, Optional

import torch
import torch.nn as nn

from .layers import AddOnMaskedLayer
from .utils import compute_cosine_similarity
    

class CosineSimGraphDecoder(nn.Module):
    """
    Cosine similarity graph decoder class.

    Takes the concatenated latent feature vectors z of the source and
    target nodes as input, and calculates the element-wise cosine similarity
    between source and target nodes to return the reconstructed edge logits.
    
    The sigmoid activation function to compute reconstructed edge probabilities
    is integrated into the binary cross entropy loss for computational
    efficiency.

    Parameters
    ----------
    dropout_rate:
        Probability of nodes to be dropped during training.
    """
    def __init__(self,
                 dropout_rate: float=0.):
        super().__init__()
        print("COSINE SIM GRAPH DECODER -> "
              f"dropout_rate: {dropout_rate}")

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self,
                z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the cosine similarity graph decoder.

        Parameters
        ----------
        z:
            Concatenated latent feature vector of the source and target nodes
            (dim: 4 * edge_batch_size x n_gps due to negative edges).

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

    Takes the latent space features z (gp scores) as input, and has a masked
    layer to decode the parameters of the underlying omics feature distributions.

    Parameters
    ----------
    modality:
        Omics modality that is decoded. Can be either `rna` or `atac`.
    entity:
        Entity that is decoded. Can be either `target` or `source`.
    n_prior_gp_input:
        Number of maskable prior gp input nodes to the decoder (maskable latent
        space dimensionality).
    n_addon_gp_input:
        Number of non-maskable add-on gp input nodes to the decoder (
        non-maskable latent space dimensionality).
    n_cat_covariates_embed_input:
        Number of categorical covariates embedding input nodes to the decoder
        (categorical covariates embedding dimensionality).
    n_output:
        Number of output nodes from the decoder (number of omics features).
    mask:
        Mask that determines which masked input nodes / prior gp latent features
        z can contribute to the reconstruction of which omics features.
    addon_mask:
        Mask that determines which add-on input nodes / add-on gp latent
        features z can contribute to the reconstruction of which omics features.
    masked_features_idx:
        Index of omics features that are included in the mask.
    recon_loss:
        The loss used for omics reconstruction. If `nb`, uses a negative
        binomial loss.
    """
    def __init__(self,
                 modality: Literal["rna", "atac"],
                 entity: Literal["target", "source"],
                 n_prior_gp_input: int,
                 n_addon_gp_input: int,
                 n_cat_covariates_embed_input: int,
                 n_output: int,
                 mask: torch.Tensor,
                 addon_mask: torch.Tensor,
                 masked_features_idx: List,
                 recon_loss: Literal["nb"]):
        super().__init__()
        print(f"MASKED {entity.upper()} {modality.upper()} DECODER -> "
              f"n_prior_gp_input: {n_prior_gp_input}, "
              f"n_addon_gp_input: {n_addon_gp_input}, "
              f"n_cat_covariates_embed_input: {n_cat_covariates_embed_input}, "
              f"n_output: {n_output}")

        self.masked_features_idx = masked_features_idx
        self.recon_loss = recon_loss

        self.nb_means_normalized_decoder = AddOnMaskedLayer(
            n_input=n_prior_gp_input,
            n_addon_input=n_addon_gp_input,
            n_cat_covariates_embed_input=n_cat_covariates_embed_input,
            n_output=n_output,
            bias=False,
            mask=mask,
            addon_mask=addon_mask,
            masked_features_idx=masked_features_idx,
            activation=nn.Softmax(dim=-1))

    def forward(self,
                z: torch.Tensor,
                log_library_size: torch.Tensor,
                cat_covariates_embed: Optional[torch.Tensor]=None,
                dynamic_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Forward pass of the masked omics feature decoder.

        Parameters
        ----------
        z:
            Tensor containing the latent space features.
        log_library_size:
            Tensor containing the omics feature log library size of the nodes.
        dynamic_mask:
           Dynamic mask that can change in each forward pass. Is used for atac
           modality: if a gene is removed by regularization in the rna decoder
           (its weight is set to 0), the corresponding peaks will be marked as 0
           in the `dynamic_mask`.        
        cat_covariates_embed:
            Tensor containing the categorical covariates embedding (all
            categorical covariates embeddings concatenated into one embedding).

        Returns
        ----------
        nb_means:
            The mean parameters of the negative binomial distribution.
        """            
        # Add categorical covariates embedding to latent feature vector
        if cat_covariates_embed is not None:
            z = torch.cat((z, cat_covariates_embed), dim=-1)
        
        nb_means_normalized = self.nb_means_normalized_decoder(
            input=z,
            dynamic_mask=dynamic_mask)
        nb_means = torch.exp(log_library_size) * nb_means_normalized
        return nb_means
    

class FCOmicsFeatureDecoder(nn.Module):
    """
    Fully connected omics feature decoder class.

    Takes the latent space features z as input, and has a fully connected layer
    to decode the parameters of the underlying omics feature distributions.

    Parameters
    ----------
    modality:
        Omics modality that is decoded. Can be either `rna` or `atac`.
    entity:
        Entity that is decoded. Can be either `target` or `source`.
    n_prior_gp_input:
        Number of maskable prior gp input nodes to the decoder (maskable latent
        space  dimensionality).
    n_addon_gp_input:
        Number of non-maskable add-on gp input nodes to the decoder (
        non-maskable latent space dimensionality).
    n_cat_covariates_embed_input:
        Number of categorical covariates embedding input nodes to the decoder
        (categorical covariates embedding dimensionality).
    n_output:
        Number of output nodes from the decoder (number of omics features).
    n_layers:
        Number of fully connected layers used for decoding.
    recon_loss:
        The loss used for omics reconstruction. If `nb`, uses a negative
        binomial loss.
    """
    def __init__(self,
                 modality: Literal["rna", "atac"],
                 entity: Literal["target", "source"],
                 n_prior_gp_input: int,
                 n_addon_gp_input: int,
                 n_cat_covariates_embed_input: int,
                 n_output: int,
                 n_layers: int,
                 recon_loss: Literal["nb"]):
        super().__init__()
        print(f"FC {entity.upper()} {modality.upper()} DECODER -> "
              f"n_prior_gp_input: {n_prior_gp_input}, "
              f"n_addon_gp_input: {n_addon_gp_input}, "
              f"n_cat_covariates_embed_input: {n_cat_covariates_embed_input}, "
              f"n_output: {n_output}")

        self.n_input = (n_prior_gp_input
                        + n_addon_gp_input
                        + n_cat_covariates_embed_input)
        self.recon_loss = recon_loss

        if n_layers == 1:
            self.nb_means_normalized_decoder = nn.Sequential(
                nn.Linear(self.n_input, n_output, bias=False),
                nn.Softmax(dim=-1))
        elif n_layers == 2:
            self.nb_means_normalized_decoder = nn.Sequential(
                nn.Linear(self.n_input, self.n_input, bias=False),
                nn.ReLU(),
                nn.Linear(self.n_input, n_output, bias=False),
                nn.Softmax(dim=-1))

    def forward(self,
                z: torch.Tensor,
                log_library_size: torch.Tensor,
                cat_covariates_embed: Optional[torch.Tensor]=None,
                **kwargs) -> torch.Tensor:
        """
        Forward pass of the fully connected omics feature decoder.

        Parameters
        ----------
        z:
            Tensor containing the latent space features.
        log_library_size:
            Tensor containing the log library size of the nodes.
        dynamic_mask:
           Dynamic mask that can change in each forward pass. Is used for atac
           modality. If a gene is removed by regularization in the rna decoder
           (its weight is set to 0), the corresponding peaks will be marked as 0
           in the `dynamic_mask`.        
        cat_covariates_embed:
            Tensor containing the categorical covariates embedding (all
            categorical covariates embeddings concatenated into one embedding).

        Returns
        ----------
        nb_means:
            The mean parameters of the negative binomial distribution.
        """            
        # Add categorical covariates embedding to latent feature vector
        if cat_covariates_embed is not None:
            z = torch.cat((z, cat_covariates_embed), dim=-1)
        
        nb_means_normalized = self.nb_means_normalized_decoder(input=z)
        nb_means = torch.exp(log_library_size) * nb_means_normalized
        return nb_means
    