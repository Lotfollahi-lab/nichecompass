"""
This module contains all loss functions used by the Variational Gene Program 
Graph Autoencoder module.
"""

from typing import Iterable, Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_addon_l1_reg_loss(model: nn.Module) -> torch.Tensor:
    """
    Compute L1 regularization loss for the add-on decoder layer weights to 
    enforce gene sparsity of add-on gene programs.

    Parameters
    ----------
    model:
        The VGPGAE module.

    Returns
    ----------
    addon_l1_reg_loss:
        L1 regularization loss for the add-on decoder layer weights.
    """
    addon_decoder_layerwise_param_sum = torch.stack(
        [torch.linalg.vector_norm(param, ord=1) for param_name, param in
         model.named_parameters() if "nb_means_normalized_decoder.addon_l" in
         param_name],
         dim=0)
    addon_l1_reg_loss = torch.sum(addon_decoder_layerwise_param_sum)
    return addon_l1_reg_loss


def compute_cond_contrastive_loss(
        edge_recon_logits: torch.Tensor,
        edge_recon_labels: torch.Tensor,
        edge_same_condition_labels: Optional[torch.Tensor]=None,
        contrastive_logits_pos_ratio: float=0.,
        contrastive_logits_neg_ratio: float=0.,
        include_same_cond_neg_edges_as_neg_examples: bool=False) -> torch.Tensor:
    """
    Compute conditional contrastive weighted binary cross entropy loss with
    logits. Sampled negative edges with nodes from different conditions whose
    edge reconstruction logits are among the top (´contrastive_logits_ratio´ *
    100)% logits are considered positive examples. Sampled negative edges with
    nodes from different conditions whose edge reconstruction logits are among
    the bottom (´contrastive_logits_ratio´ * 100)% logits are considered
    negative examples.

    Parameters
    ----------
    edge_recon_logits:
        Predicted edge reconstruction logits for both positive and negative
        sampled edges (dim: 2 * edge_batch_size).
    edge_recon_labels:
        Edge ground truth labels for both positive and negative sampled edges
        (dim: 2 * edge_batch_size).
    edge_same_condition_labels:
        Edge same condition labels for both positive and negative sampled edges
        (dim: 2 * edge_batch_size).
    contrastive_logits_pos_ratio:
        Ratio for determining the logits threshold of positive contrastive
        examples of node pairs from different conditions. The top
        (´contrastive_logits_pos_ratio´ * 100)% logits of node pairs from
        different conditions serve as positive labels for the contrastive
        loss.
    contrastive_logits_neg_ratio:
        Ratio for determining the logits threshold of negative contrastive
        examples of node pairs from different conditions. The bottom
        (´contrastive_logits_neg_ratio´ * 100)% logits of node pairs from
        different conditions serve as negative labels for the contrastive
        loss.
    include_same_cond_neg_edges_as_neg_examples:
        If ´True´, in addition include negative edges of node pairs from
        the same condition as negative contrastive examples.

    Returns
    ----------
    cond_contrastive_loss:
        Conditional contrastive binary cross entropy loss (calculated from
        logits for numerical stability in backpropagation).
    """
    if edge_same_condition_labels is None or (
        (contrastive_logits_pos_ratio == 0) & (contrastive_logits_neg_ratio == 0)):
        return torch.tensor(0.)
                
    # Determine logit thresholds for positive and negative contrastive examples
    # of node pairs from different conditions
    edge_recon_logits_same_condition = edge_recon_logits[
        ~edge_same_condition_labels]
    edge_recon_labels_same_condition = edge_recon_labels[
        ~edge_same_condition_labels]
    pos_n_top = math.ceil(contrastive_logits_pos_ratio *
                          len(edge_recon_logits_same_condition))
    if pos_n_top == 0:
        pos_thresh = torch.tensor(float("inf")).to(
            edge_recon_logits_same_condition.device)
    else:
        pos_thresh = torch.topk(
            edge_recon_logits_same_condition.detach().clone(),
            pos_n_top).values[-1]            
    neg_n_top = math.ceil(contrastive_logits_neg_ratio *
                          len(edge_recon_logits_same_condition))
    if neg_n_top == 0:
        neg_thresh = torch.tensor(float("-inf")).to(
            edge_recon_logits_same_condition.device)
    else:
        neg_thresh = torch.topk(
            edge_recon_logits_same_condition.detach().clone(),
            neg_n_top,
            largest=False).values[-1]

    # Set labels of different condition node pairs with logits above ´pos_thresh´
    # to 1, labels of different condition node pairs with logits below ´neg_thresh´
    # to 0, labels of same condition node pairs with neg edges to 0 (if specified),
    # and exclude other examples from the loss
    diff_cond_pos_examples = (
        (~edge_same_condition_labels) & (edge_recon_logits >= pos_thresh))
    diff_cond_neg_examples = (
        (~edge_same_condition_labels) & (edge_recon_logits <= neg_thresh))
    if include_same_cond_neg_edges_as_neg_examples:
        same_cond_neg_examples = (
            edge_same_condition_labels & (edge_recon_labels == 0))
    else:
        same_cond_neg_examples = torch.full((edge_recon_labels.size(0),),
                                            False,
                                            dtype=torch.bool).to(
            edge_recon_labels.device)
    
    edge_recon_labels[diff_cond_pos_examples] = 1
    edge_recon_labels[diff_cond_neg_examples] = 0
    edge_recon_labels[same_cond_neg_examples] = 0
    edge_recon_logits = edge_recon_logits[
        diff_cond_pos_examples | diff_cond_neg_examples | same_cond_neg_examples]
    edge_recon_labels = edge_recon_labels[
        diff_cond_pos_examples | diff_cond_neg_examples | same_cond_neg_examples]

    # Compute bce loss from logits for numerical stability
    cond_contrastive_loss = F.binary_cross_entropy_with_logits(
        edge_recon_logits,
        edge_recon_labels)
    return cond_contrastive_loss


def compute_edge_recon_loss(
        edge_recon_logits: torch.Tensor,
        edge_recon_labels: torch.Tensor,
        edge_same_condition_labels: Optional[torch.Tensor]=None
        ) -> torch.Tensor:
    """
    Compute edge reconstruction weighted binary cross entropy loss with logits 
    using ground truth edge labels and predicted edge logits.

    Parameters
    ----------
    edge_recon_logits:
        Predicted edge reconstruction logits for both positive and negative
        sampled edges (dim: 2 * edge_batch_size).
    edge_recon_labels:
        Edge ground truth labels for both positive and negative sampled edges
        (dim: 2 * edge_batch_size).
    edge_same_condition_labels:
        Edge same condition labels for both positive and negative sampled edges
        (dim: 2 * edge_batch_size).

    Returns
    ----------
    edge_recon_loss:
        Weighted binary cross entropy loss between edge labels and predicted
        edge probabilities (calculated from logits for numerical stability in
        backpropagation).
    """
    if edge_same_condition_labels is not None:
        # Remove examples that have nodes from different conditions
        edge_recon_logits = edge_recon_logits[edge_same_condition_labels]
        edge_recon_labels = edge_recon_labels[edge_same_condition_labels]

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
    Compute gene expression reconstruction loss according to a negative binomial
    gene expression model, which is often used to model omics count data such as
    scRNA-seq or scATAC-seq data.

    Parts of the implementation are adapted from Lopez, R., Regier, J., Cole, M.
    B., Jordan, M. I. & Yosef, N. Deep generative modeling for single-cell
    transcriptomics. Nat. Methods 15, 1053–1058 (2018):
    https://github.com/scverse/scvi-tools/blob/main/scvi/distributions/_negative_binomial.py#L75
    (29.11.2022).

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
        (dim: batch_size x n_genes)
    eps:
        Numerical stability constant.

    Returns
    ----------
    nb_loss:
        Gene expression reconstruction loss using a NB gene expression model.
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


def compute_gene_expr_recon_zinb_loss(x: torch.Tensor,
                                      mu: torch.Tensor,
                                      theta: torch.Tensor,
                                      zi_prob_logits: torch.Tensor,
                                      eps: float=1e-8) -> torch.Tensor:
    """
    Gene expression reconstruction loss according to a zero-inflated negative 
    binomial gene expression model, which is used to model scRNA-seq count data
    due to its capacity of modeling excess zeros and overdispersion. The 
    bernoulli distribution is parameterized using logits, hence the use of a 
    softplus function.

    Parts of the implementation are adapted from
    https://github.com/scverse/scvi-tools/blob/master/scvi/distributions/_negative_binomial.py#L22
    (01.10.2022).

    Parameters
    ----------
    x:
        Reconstructed feature matrix.
    mu:
        Mean of the negative binomial with positive support.
        (dim: batch_size x n_genes)
    theta:
        Inverse dispersion parameter with positive support.
        (dim: batch_size x n_genes)
    zi_prob_logits:
        Logits of the zero inflation probability with real support.
        (dim: batch_size x n_genes)
    eps:
        Numerical stability constant.

    Returns
    ----------
    zinb_loss:
        Gene expression reconstruction loss using a ZINB gene expression model.
    """
    # Reshape theta for broadcasting
    theta = theta.view(1, theta.size(0))

    # Uses log(sigmoid(x)) = -softplus(-x)
    softplus_zi_prob_logits = F.softplus(-zi_prob_logits)
    log_theta_eps = torch.log(theta + eps)
    log_theta_mu_eps = torch.log(theta + mu + eps)
    zi_prob_logits_theta_log = -zi_prob_logits + theta * (log_theta_eps - 
                                                          log_theta_mu_eps)

    case_zero = F.softplus(zi_prob_logits_theta_log) - softplus_zi_prob_logits
    mul_case_zero = torch.mul((x < eps).type(torch.float32), case_zero)

    case_non_zero = (
        -softplus_zi_prob_logits
        + zi_prob_logits_theta_log
        + x * (torch.log(mu + eps) - log_theta_mu_eps)
        + torch.lgamma(x + theta)
        - torch.lgamma(theta)
        - torch.lgamma(x + 1))
                     
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    log_likehood_zinb = mul_case_zero + mul_case_non_zero
    zinb_loss = torch.mean(-log_likehood_zinb.sum(-1))
    return zinb_loss


def compute_group_lasso_reg_loss(model: nn.Module) -> torch.Tensor:
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
                model.gene_expr_decoder.nb_means_normalized_decoder.masked_l.mask,
                dim=0)),
            torch.linalg.vector_norm(param, ord=2, dim=0)).sum()
         for param_name, param in model.named_parameters() if
         "gene_expr_decoder.nb_means_normalized_decoder.masked_l" in param_name],
         dim=0)

    # TO DO #
    # Sum over ´masked_l´ layer and ´addon_l´ layer if addon gene programs exist
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


def compute_masked_l1_reg_loss(model: nn.Module,
                               l1_masked_gp_idx,
                               only_target_genes: bool=True) -> torch.Tensor:
    """
    Compute L1 regularization loss for the masked decoder layer weights to 
    encourage gene sparsity of masked gene programs.

    Parameters
    ----------
    model:
        The VGPGAE module.
    only_target_genes:
        If ´True´, compute regularization loss only for target genes (not for
        source genes).
    min_genes_per_gp:
        Minimum number of genes that are in the gene program mask for a gene
        program to be included in the l1 reg loss.

    Returns
    ----------
    masked_l1_reg_loss:
        L1 regularization loss for the masked decoder layer weights.
    """
    if only_target_genes & (model.node_label_method_ != "self"):
        param_end_gene_idx = int(model.n_output_genes_ / 2)
    else:
        param_end_gene_idx = None

    # First compute layer-wise sum of absolute weights over all masked gene
    # expression decoder layers, then sum across layers
    masked_decoder_layerwise_param_sum = torch.stack(
        [torch.linalg.vector_norm(param[:param_end_gene_idx, l1_masked_gp_idx],
                                  ord=1) for
         param_name, param in model.named_parameters() if
         "nb_means_normalized_decoder.masked_l" in param_name],
        dim=0)
    masked_l1_reg_loss = torch.sum(masked_decoder_layerwise_param_sum)
    return masked_l1_reg_loss