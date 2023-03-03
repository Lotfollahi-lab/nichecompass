"""
This module contains all loss functions used by the Variational Gene Program 
Graph Autoencoder module.
"""

from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import edge_values_and_sorted_labels


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


def compute_edge_recon_loss(
        adj_recon_logits: torch.Tensor,
        edge_labels: torch.Tensor,
        edge_label_index: torch.Tensor,
        edge_label_conditions: Optional[torch.Tensor]=None) -> torch.Tensor:
    """
    Compute edge reconstruction weighted binary cross entropy loss with logits 
    using ground truth edge labels and predicted edge logits (retrieved from the
    reconstructed logits adjacency matrix).

    Parameters
    ----------
    adj_recon_logits:
        Adjacency matrix containing the predicted edge logits.
    edge_labels:
        Edge ground truth labels for both positive and negatively sampled edges.
    edge_label_index:
        Index with edge labels for both positive and negatively sampled edges.

    Returns
    ----------
    edge_recon_loss:
        Binary cross entropy loss between edge labels and predicted edge
        probabilities (calculated from logits for numerical stability in
        backpropagation).
    """
    if edge_label_conditions is not None:
        # Remove negative samples that are not within a condition
        # (This implementation is not super efficient and could be improved
        # in the future by avoiding to sample these edges in the first place)
        same_condition_edge = (edge_label_conditions[edge_label_index[0]] ==
                               edge_label_conditions[edge_label_index[1]])
        edge_labels = edge_labels[same_condition_edge]
        edge_label_index = edge_label_index[:, same_condition_edge]

    edge_recon_logits, edge_labels_sorted = edge_values_and_sorted_labels(
        adj=adj_recon_logits,
        edge_label_index=edge_label_index,
        edge_labels=edge_labels)

    # Compute weighted cross entropy loss
    edge_recon_loss = F.binary_cross_entropy_with_logits(edge_recon_logits,
                                                         edge_labels_sorted)
    return edge_recon_loss


def compute_gene_expr_recon_nb_loss(x: torch.Tensor,
                                    mu: torch.Tensor,
                                    theta: torch.Tensor,
                                    eps: float=1e-8) -> torch.Tensor:
    """
    Gene expression reconstruction loss according to a negative binomial gene
    expression model which is used to model scRNA-seq count data.

    Parts of the implementation are adapted from
    https://github.com/scverse/scvi-tools/blob/main/scvi/distributions/_negative_binomial.py#L75
    (29.11.2022).

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
    to enforce gene program sparsity (each gene program is a group; the number
    of weights per group normalization is omitted as each group / gene program
    has the same number of weights).

    Parameters
    ----------
    model:
        The VGPGAE module.

    Returns
    ----------
    group_lasso_reg_loss:
        Group lasso regularization loss for the decoder layer weights.    
    """
    # Compute L2 norm per group / gene program and sum across all gene programs
    decoder_layerwise_param_gpgroupnorm_sum = torch.stack(
        [torch.linalg.vector_norm(param, ord=2, dim=0).sum() for param_name,
         param in model.named_parameters() if
         "gene_expr_decoder.nb_means_normalized_decoder.masked_l" in param_name],
         dim=0)
    # Sum over ´masked_l´ layer and ´addon_l´ layer if addon gene programs exist
    group_lasso_reg_loss = torch.sum(decoder_layerwise_param_gpgroupnorm_sum)
    return group_lasso_reg_loss


def compute_kl_reg_loss(mu: torch.Tensor,
                        logstd: torch.Tensor,
                        n_nodes: int) -> torch.Tensor:
    """
    Compute Kullback-Leibler divergence as per Kingma, D. P. & Welling, M. 
    Auto-Encoding Variational Bayes. arXiv [stat.ML] (2013).

    Parameters
    ----------
    mu:
        Expected values of the latent distribution.
    logstd:
        Log of standard deviations of the latent distribution.
    n_nodes:
        Number of nodes in the graph.

    Returns
    ----------
    kl_reg_loss:
        Kullback-Leibler divergence.
    """
    kl_reg_loss = (-0.5 / n_nodes) * torch.mean(
    torch.sum(1 + 2 * logstd - mu ** 2 - torch.exp(logstd) ** 2, 1))
    return kl_reg_loss


def compute_masked_l1_reg_loss(model: nn.Module,
                               only_target_genes: bool=False) -> torch.Tensor:
    """
    Compute L1 regularization loss for the masked decoder layer weights to 
    enforce gene sparsity of masked gene programs.

    Parameters
    ----------
    model:
        The VGPGAE module.
    only_target_genes:
        If ´True´, compute regularization loss only for target genes.

    Returns
    ----------
    masked_l1_reg_loss:
        L1 regularization loss for the masked decoder layer weights.
    """
    if only_target_genes:
        param_end_gene_idx = model.n_input_
    else:
        param_end_gene_idx = None

    masked_decoder_layerwise_param_sum = torch.stack(
        [torch.linalg.vector_norm(param[:param_end_gene_idx, :], ord=1) for
         param_name, param in model.named_parameters() if
         "nb_means_normalized_decoder.masked_l" in param_name],
         dim=0)
    masked_l1_reg_loss = torch.sum(masked_decoder_layerwise_param_sum)
    return masked_l1_reg_loss