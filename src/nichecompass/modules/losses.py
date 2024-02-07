"""
This module contains all loss functions used by the Variational Gene Program 
Graph Autoencoder module.
"""

from typing import List, Literal, Optional

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_cat_covariates_contrastive_loss(
        edge_recon_logits: torch.Tensor,
        edge_recon_labels: torch.Tensor,
        edge_same_cat_covariates_cat: Optional[List[torch.Tensor]]=None,
        contrastive_logits_pos_ratio: float=0.,
        contrastive_logits_neg_ratio: float=0.) -> torch.Tensor:
    """
    Compute categorical covariates contrastive weighted binary cross entropy
    loss with logits. The loss is computed for each categorical covariate
    separately and added up. Sampled edges with nodes from different categories
    whose edge reconstruction logits are among the top
    (´contrastive_logits_pos_ratio´ * 100)% logits are considered positive
    examples for a specific categorical covariate. Sampled edges with
    nodes from different categories whose edge reconstruction logits are among
    the bottom (´contrastive_logits_neg_ratio´ * 100)% logits are considered
    negative examples.

    Parameters
    ----------
    edge_recon_logits:
        Predicted edge reconstruction logits for both positive and negative
        sampled edges (dim: 2 * edge_batch_size).
    edge_recon_labels:
        Edge ground truth labels for both positive and negative sampled edges
        (dim: 2 * edge_batch_size).
    edge_same_cat_covariates_cat:
        List of boolean tensors indicating whether the edge node pair has the
        same categorical covariate category for each categorical covariate
        respectively, and for both positive and negative sampled edges (dim of
        tensors: 2 * edge_batch_size).
    contrastive_logits_pos_ratio:
        Ratio for determining the logits threshold of positive contrastive
        examples of node pairs from different categories. The top
        (´contrastive_logits_pos_ratio´ * 100)% logits of node pairs from
        different categories serve as positive labels for the contrastive
        loss.
    contrastive_logits_neg_ratio:
        Ratio for determining the logits threshold of negative contrastive
        examples of node pairs from different categories. The bottom
        (´contrastive_logits_neg_ratio´ * 100)% logits of node pairs from
        different categories serve as negative labels for the contrastive
        loss.

    Returns
    ----------
    cat_covariates_contrastive_loss:
        Categorical covariates contrastive binary cross entropy loss (calculated
        from logits for numerical stability in backpropagation, and summed up
        over all categorical covariates).
    """
    if edge_same_cat_covariates_cat is None or (
        (contrastive_logits_pos_ratio == 0) &
        (contrastive_logits_neg_ratio == 0)):
        return torch.tensor(0.)
    
    cat_covariates_contrastive_loss = torch.tensor(0.).to(
                edge_recon_logits.device)
    
    # Compute categorical covariate contrastive loss for each categorical
    # covariate and add to accumulated loss
    for edge_same_cat_covariate_cat in edge_same_cat_covariates_cat:
        # Determine logit thresholds for positive and negative contrastive
        # examples of node pairs from different categorical covariate categories
        edge_recon_logits_diff_cat_covariate_cat = edge_recon_logits[
            ~edge_same_cat_covariate_cat]
        pos_n_top = math.ceil(contrastive_logits_pos_ratio *
                              len(edge_recon_logits_diff_cat_covariate_cat))
        if pos_n_top == 0:
            pos_thresh = torch.tensor(float("inf")).to(
                edge_recon_logits.device)
        else:
            pos_thresh = torch.topk(
                edge_recon_logits_diff_cat_covariate_cat.detach().clone(),
                pos_n_top).values[-1]            
        neg_n_top = math.ceil(contrastive_logits_neg_ratio *
                              len(edge_recon_logits_diff_cat_covariate_cat))
        if neg_n_top == 0:
            neg_thresh = torch.tensor(float("-inf")).to(
                edge_recon_logits.device)
        else:
            neg_thresh = torch.topk(
                edge_recon_logits_diff_cat_covariate_cat.detach().clone(),
                neg_n_top,
                largest=False).values[-1]

        # Set labels of different category node pairs with logits above
        # ´pos_thresh´ to 1, labels of different category node pairs with logits
        # below ´neg_thresh´ to 0, and exclude other examples from the loss
        diff_cat_covariate_pos_examples = (
            (~edge_same_cat_covariate_cat) & (edge_recon_logits >= pos_thresh))
        diff_cat_covariate_neg_examples = (
            (~edge_same_cat_covariate_cat) & (edge_recon_logits <= neg_thresh))

        edge_recon_labels[diff_cat_covariate_pos_examples] = 1
        edge_recon_labels[diff_cat_covariate_neg_examples] = 0
        edge_recon_logits_contrastive_examples = edge_recon_logits[
            diff_cat_covariate_pos_examples | diff_cat_covariate_neg_examples]
        edge_recon_labels_contrastive_examples = edge_recon_labels[
            diff_cat_covariate_pos_examples | diff_cat_covariate_neg_examples]

        # Compute bce loss from logits for numerical stability
        cat_covariate_contrastive_loss = F.binary_cross_entropy_with_logits(
            edge_recon_logits_contrastive_examples,
            edge_recon_labels_contrastive_examples)
        cat_covariates_contrastive_loss += cat_covariate_contrastive_loss
    return cat_covariates_contrastive_loss


def compute_edge_recon_loss(
        edge_recon_logits: torch.Tensor,
        edge_recon_labels: torch.Tensor,
        edge_incl: Optional[torch.Tensor]=None
        ) -> torch.Tensor:
    """
    Compute edge reconstruction weighted binary cross entropy loss with logits 
    using ground truth edge labels and predicted edge logits.

    Parameters
    ----------
    edge_recon_logits:
        Predicted edge reconstruction logits for both positive and negative
        sampled edges (dim: 2 * ´edge_batch_size´).
    edge_recon_labels:
        Edge ground truth labels for both positive and negative sampled edges
        (dim: 2 * ´edge_batch_size´).
    edge_incl:
        Boolean mask which indicates edges to be included in the edge recon loss
        (dim: 2 * ´edge_batch_size´). If ´None´, includes all edges.

    Returns
    ----------
    edge_recon_loss:
        Weighted binary cross entropy loss between edge labels and predicted
        edge probabilities (calculated from logits for numerical stability in
        backpropagation).
    """
    if edge_incl is not None:
        # Remove edges whose node pair has different categories in categorical
        # covariates for which no cross-category edges are present
        edge_recon_logits = edge_recon_logits[edge_incl]
        edge_recon_labels = edge_recon_labels[edge_incl]

    # Determine weighting of positive examples
    pos_labels = (edge_recon_labels == 1.).sum(dim=0)
    neg_labels = (edge_recon_labels == 0.).sum(dim=0)
    pos_weight = neg_labels / pos_labels

    # Compute weighted bce loss from logits for numerical stability
    edge_recon_loss = F.binary_cross_entropy_with_logits(edge_recon_logits,
                                                         edge_recon_labels,
                                                         pos_weight=pos_weight)
    return edge_recon_loss


def compute_omics_recon_nb_loss(x: torch.Tensor,
                                mu: torch.Tensor,
                                theta: torch.Tensor,
                                eps: float=1e-8) -> torch.Tensor:
    """
    Compute omics reconstruction loss according to a negative binomial model,
    which is often used to model omics count data such as scRNA-seq or
    scATAC-seq data.

    Parts of the implementation are adapted from Lopez, R., Regier, J., Cole, M.
    B., Jordan, M. I. & Yosef, N. Deep generative modeling for single-cell
    transcriptomics. Nat. Methods 15, 1053–1058 (2018) ->
    https://github.com/scverse/scvi-tools/blob/main/scvi/distributions/_negative_binomial.py#L75;
    29.11.2022.

    Parameters
    ----------
    x:
        Reconstructed feature vector (dim: batch_size, n_genes; nodes that
        are in current batch beyond originally sampled batch_size for message
        passing reasons are not considered).
    mu:
        Mean of the negative binomial with positive support.
        (dim: batch_size x n_genes)
    theta:
        Inverse dispersion parameter with positive support.
        (dim: n_genes)
    eps:
        Numerical stability constant.

    Returns
    ----------
    nb_loss:
        Omics reconstruction loss using a negative binomial model.
    """
    log_theta_mu_eps = torch.log(theta + mu + eps)
    log_likelihood_nb = (
        theta * (torch.log(theta + eps) - log_theta_mu_eps)
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1))
    nb_loss = torch.mean(-log_likelihood_nb.sum(-1))
    return nb_loss 


def compute_gp_group_lasso_reg_loss(model: nn.Module) -> torch.Tensor:
    """
    Compute group lasso regularization loss for the masked decoder layer weights
    to enforce gene program sparsity (each gene program is a group; groups are
    normalized by the number of non-masked weights per group).
    
    Check https://leimao.github.io/blog/Group-Lasso/ for more information about
    group lasso regularization.

    Parameters
    ----------
    model:
        The VGPGAE module.

    Returns
    ----------
    group_lasso_reg_loss:
        Group lasso regularization loss for the decoder layer weights.    
    """
    # Compute L2 norm per group / gene program, normalize by number of weights
    # and sum across all gene programs
    decoder_layerwise_param_gpgroupnorm_sum = torch.stack(
        [torch.mul(
            torch.sqrt(torch.count_nonzero(
                getattr(
                    getattr(
                        getattr(
                            getattr(model, param_name.split(".")[0]),
                            param_name.split(".")[1]),
                        param_name.split(".")[2]),
                    "mask"),
                dim=0)),
            torch.linalg.vector_norm(param, ord=2, dim=0)).sum()
         for param_name, param in model.named_parameters() if
         "rna_decoder.nb_means_normalized_decoder.masked_l" in param_name],
         dim=0)
    group_lasso_reg_loss = torch.sum(decoder_layerwise_param_gpgroupnorm_sum)
    return group_lasso_reg_loss


def compute_kl_reg_loss(mu: torch.Tensor,
                        logstd: torch.Tensor) -> torch.Tensor:
    """
    Compute Kullback-Leibler divergence as per Kingma, D. P. & Welling, M. 
    Auto-Encoding Variational Bayes. arXiv [stat.ML] (2013). Equation (10).
    This will encourage encodings to distribute evenly around the center of
    a continuous and complete latent space, producing similar (for points close
    in latent space) and meaningful content after decoding.
    
    For detailed derivation, see
    https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes.

    Parameters
    ----------
    mu:
        Expected values of the normal latent distribution of each node (dim:
        n_nodes_current_batch, n_gps).
    logstd:
        Log standard deviations of the normal latent distribution of each node
        (dim: n_nodes_current_batch, n_gps).

    Returns
    ----------
    kl_reg_loss:
        Kullback-Leibler divergence.
    """
    # Sum over n_gps and mean over n_nodes_current_batch
    kl_reg_loss = -0.5 * torch.mean(
    torch.sum(1 + 2 * logstd - mu ** 2 - torch.exp(logstd) ** 2, 1))
    return kl_reg_loss


def compute_gp_l1_reg_loss(
        model: nn.Module,
        gp_type: Literal["prior", "addon"],
        l1_targets_mask: Optional[torch.Tensor]=None,
        l1_sources_mask: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Compute L1 regularization loss for the rna decoder weights of gene programs
    of the type ´gp_type´ to encourage gene sparsity of those gene programs.

    Parameters
    ----------
    model:
        The VGPGAE module.
    gp_type:
        Type of gene programs to which the L1 regularization loss should be
        applied.
    l1_targets_mask:
        Boolean gene program gene mask that is True for all gene program target
        genes to which the L1 regularization loss should be applied (dim:
        n_genes, n_gps).
    l1_sources_mask:
        Boolean gene program gene mask that is True for all gene program source
        genes to which the L1 regularization loss should be applied (dim:
        n_genes, n_gps).

    Returns
    ----------
    gp_l1_reg_loss:
        L1 regularization loss for the rna decoder weights.
    """
    if gp_type == "prior":
        layer_name = "masked_l"
    elif gp_type == "addon":
        layer_name = "addon_l"

    # First compute layer-wise sum of absolute weights over target and source
    # rna decoder layers, then sum across layers. Use l1 masks to determine
    # which weights are included in the sum.
    # NOTE:
    # - the absolute weights and thus the L1 loss will be higher for highly
    #   expressed genes
    # - the model will keep weights non-zero for gps with very high scores and
    #   turn off weights for gps with low scores 
    decoder_layerwise_param_sum = torch.stack(
        [torch.linalg.vector_norm(param[l1_targets_mask if 
                                        "target" in param_name else
                                        l1_sources_mask],
                                  ord=1) for
         param_name, param in model.named_parameters() if
         f"rna_decoder.nb_means_normalized_decoder.{layer_name}" in param_name],
        dim=0)
    gp_l1_reg_loss = torch.sum(decoder_layerwise_param_sum)
    return gp_l1_reg_loss
