"""
This module contains decoders used by the NicheCompass model.
"""

from typing import Literal, List, Optional

import torch
import torch.nn as nn
from torch import nn as nn


class CosineSimGraphDecoder(nn.Module):
    #FIXME check against CosineSimilarity in torch
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


class AddOnMaskedLayer(nn.Module):
    """
    Add-on masked layer class.

    Parts of the implementation are adapted from
    https://github.com/theislab/scarches/blob/7980a187294204b5fb5d61364bb76c0b809eb945/scarches/models/expimap/modules.py#L28;
    01.10.2022.

    Parameters
    ----------
    n_input:
        Number of mask input nodes to the add-on masked layer.
    n_output:
        Number of output nodes from the add-on masked layer.
    mask:
        Mask that is used to mask the node connections for mask inputs from the
        input layer to the output layer.
    addon_mask:
        Mask that is used to mask the node connections for add-on inputs from
        the input layer to the output layer.
    masked_features_idx:
        Index of input features that are included in the mask.
    bias:
        If ´True´, use a bias for the mask input nodes.
    n_addon_input:
        Number of add-on input nodes to the add-on masked layer.
    n_cat_covariates_embed_input:
        Number of categorical covariates embedding input nodes to the addon
        masked layer.
    activation:
        Activation function used at the end of the ad-on masked layer.
    """
    def __init__(self,
                 n_input: int,
                 n_output: int,
                 mask: torch.Tensor,
                 addon_mask: torch.Tensor,
                 masked_features_idx: List,
                 bias: bool=False,
                 n_addon_input: int=0,
                 n_cat_covariates_embed_input: int=0,
                 activation: nn.Module=nn.Softmax(dim=-1)):
        super().__init__()
        self.n_input = n_input
        self.n_addon_input = n_addon_input
        self.n_cat_covariates_embed_input = n_cat_covariates_embed_input
        self.masked_features_idx = masked_features_idx

        # Masked layer
        self.masked_l = MaskedLinear(n_input=n_input,
                                     n_output=n_output,
                                     mask=mask,
                                     bias=bias)

        # Add-on layer
        if n_addon_input != 0:
            self.addon_l = MaskedLinear(n_input=n_addon_input,
                                        n_output=n_output,
                                        mask=addon_mask,
                                        bias=False)

        # Categorical covariates embedding layer
        if n_cat_covariates_embed_input != 0:
            self.cat_covariates_embed_l = nn.Linear(
                n_cat_covariates_embed_input,
                n_output,
                bias=False)

        self.activation = activation

    def forward(self,
                input: torch.Tensor,
                dynamic_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
        """
        Forward pass of the add-on masked layer.

        Parameters
        ----------
        input:
            Input features to the add-on masked layer. Includes add-on input
            nodes and categorical covariates embedding input nodes if specified.
        dynamic_mask:
            Additional optional dynamic mask for the masked layer.

        Returns
        ----------
        output:
            Output of the add-on masked layer.
        """
        if (self.n_addon_input == 0) & (self.n_cat_covariates_embed_input == 0):
            mask_input = input
        elif ((self.n_addon_input != 0) &
              (self.n_cat_covariates_embed_input == 0)):
            mask_input, addon_input = torch.split(
                input,
                [self.n_input, self.n_addon_input],
                dim=1)
        elif ((self.n_addon_input == 0) &
              (self.n_cat_covariates_embed_input != 0)):
            mask_input, cat_covariates_embed_input = torch.split(
                input,
                [self.n_input, self.n_cat_covariates_embed_input],
                dim=1)
        elif ((self.n_addon_input != 0) &
              (self.n_cat_covariates_embed_input != 0)):
            mask_input, addon_input, cat_covariates_embed_input = torch.split(
                input,
                [self.n_input, self.n_addon_input, self.n_cat_covariates_embed_input],
                dim=1)

        output = self.masked_l(
            input=mask_input,
            dynamic_mask=(dynamic_mask[:self.n_input, :] if
                          dynamic_mask is not None else None))
            # Dynamic mask also has entries for add-on gps
        if self.n_addon_input != 0:
            # Only unmasked features will have weights != 0.
            output += self.addon_l(
                input=addon_input,
                dynamic_mask=(dynamic_mask[self.n_input:, :] if
                              dynamic_mask is not None else None))
        if self.n_cat_covariates_embed_input != 0:
            if self.n_addon_input != 0:
                output += self.cat_covariates_embed_l(
                    cat_covariates_embed_input)
            else:
                # Only add categorical covariates embedding layer output to
                # masked features
                output[:, self.masked_features_idx] += self.cat_covariates_embed_l(
                    cat_covariates_embed_input)[:, self.masked_features_idx]
        output = self.activation(output)
        return output


def compute_cosine_similarity(tensor1: torch.Tensor,
                              tensor2: torch.Tensor,
                              eps: float=1e-8) -> torch.Tensor:
    """
    Compute the element-wise cosine similarity between two 2D tensors.

    Parameters
    ----------
    tensor1:
        First tensor for element-wise cosine similarity computation (dim: n_obs
        x n_features).
    tensor2:
        Second tensor for element-wise cosine similarity computation (dim: n_obs
        x n_features).

    Returns
    ----------
    cosine_sim:
        Result tensor that contains the computed element-wise cosine
        similarities (dim: n_obs).
    """
    tensor1_norm = tensor1.norm(dim=1)[:, None]
    tensor2_norm = tensor2.norm(dim=1)[:, None]
    tensor1_normalized = tensor1 / torch.max(
            tensor1_norm, eps * torch.ones_like(tensor1_norm))
    tensor2_normalized = tensor2 / torch.max(
            tensor2_norm, eps * torch.ones_like(tensor2_norm))
    cosine_sim = torch.mul(tensor1_normalized, tensor2_normalized).sum(1)
    return cosine_sim
