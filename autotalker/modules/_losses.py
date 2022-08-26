import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

from ._utils import unique_sorted_index


def compute_edge_recon_loss(adj_recon_logits: torch.Tensor,
                            edge_labels: torch.Tensor,
                            edge_label_index: torch.Tensor,
                            pos_weight: torch.Tensor):
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
    pos_weight:
        Weight with which positive examples are reweighted in the loss 
        calculation. Should be 1 if negative sampling ratio is 1.

    Returns
    ----------
    edge_recon_loss:
        Binary cross entropy loss between edge labels and predicted edge
        probabilities (calculated from logits for numerical stability in
        backpropagation).
    """
    # Create mask to retrieve values as given in ´edge_label_index´ from 
    # ´adj_recon_logits´
    n_nodes = adj_recon_logits.shape[0]
    adj_labels = to_dense_adj(edge_label_index, max_num_nodes=n_nodes)
    mask = torch.squeeze(adj_labels > 0)
    
    # Retrieve logits
    edge_recon_logits = torch.masked_select(adj_recon_logits, mask)
    
    # Sort ´edge_labels´ to align order with masked retrieval from adjacency 
    # matrix. In addition, remove entries in ´edge_labels´ that are due to 
    # duplicates in ´edge_label_index´, which can happen because of the 
    # approximate negative sampling implementation in PyG LinkNeighborLoader. 
    sort_index = unique_sorted_index(edge_label_index)
    edge_labels_sorted = edge_labels[sort_index]

    # Compute weighted cross entropy loss
    edge_recon_loss = F.binary_cross_entropy_with_logits(edge_recon_logits,
                                                         edge_labels_sorted,
                                                         pos_weight=pos_weight)
    return edge_recon_loss


def compute_kl_loss(mu: torch.Tensor,
                    logstd: torch.Tensor,
                    n_nodes: int):
    """
    Compute Kullback-Leibler divergence as per Kingma, D. P. & Welling, M. 
    Auto-Encoding Variational Bayes. arXiv [stat.ML] (2013).

    Parameters
    ----------
    mu:
        Expected values of the latent space distribution.
    logstd:
        Log of standard deviations of the latent space distribution.
    n_nodes:
        Number of nodes in the graph.

    Returns
    ----------
    kl_loss:
        Kullback-Leibler divergence.
    """
    kl_loss = (-0.5 / n_nodes) * torch.mean(
    torch.sum(1 + 2 * logstd - mu ** 2 - torch.exp(logstd) ** 2, 1))
    return kl_loss


def compute_vgae_loss(adj_recon_logits: torch.Tensor,
                      edge_labels: torch.Tensor,
                      edge_label_index: torch.Tensor,
                      edge_recon_loss_pos_weight: torch.Tensor,
                      edge_recon_loss_norm_factor: float,
                      mu: torch.Tensor,
                      logstd: torch.Tensor,
                      n_nodes: int):
    """
    Compute the Variational Graph Autoencoder loss which consists of the
    weighted binary cross entropy loss (edge reconstruction loss), where sparse 
    positive examples (Aij = 1) can be reweighted in case of imbalanced negative
    sampling, as well as the Kullback-Leibler divergence (regularization loss) 
    as per Kingma, D. P. & Welling, M. Auto-Encoding Variational Bayes. arXiv 
    [stat.ML] (2013). The edge reconstruction loss is weighted with a 
    normalization factor compared to the regularization loss.

    Parameters
    ----------
    adj_recon_logits:
        Adjacency matrix containing the predicted edge logits.
    edge_labels:
        Edge ground truth labels for both positive and negatively sampled edges.
    edge_label_index:
        Index with edge labels for both positive and negatively sampled edges.
    edge_recon_loss_pos_weight:
        Weight with which positive examples are reweighted in the reconstruction
        loss calculation. Should be 1 if negative sampling ratio is 1.
    edge_recon_loss_norm_factor:
        Factor with which edge reconstruction loss is weighted compared to 
        Kullback-Leibler divergence.
    mu:
        Expected values of the latent space distribution.
    logstd:
        Log of standard deviations of the latent space distribution.
    n_nodes:
        Number of nodes in the graph.
        
    Returns
    ----------
    vgae_loss:
        Variational Graph Autoencoder loss composed of edge reconstruction and 
        regularization loss.
    """
    edge_recon_loss = compute_edge_recon_loss(
        adj_recon_logits,
        edge_labels,
        edge_label_index,
        pos_weight=edge_recon_loss_pos_weight)

    kl_loss = compute_kl_loss(mu, logstd, n_nodes)

    vgae_loss = edge_recon_loss_norm_factor * edge_recon_loss + kl_loss
    return vgae_loss


def vgae_loss_parameters(data_batch, device):
    """
    Parameters
    ----------

    Returns
    ----------
    """
    n_possible_edges = data_batch.x.shape[0] ** 2
    n_neg_edges = (data_batch.edge_label == 0).sum()
    edge_recon_loss_norm_factor = n_possible_edges / n_neg_edges
    edge_recon_loss_pos_weight = torch.Tensor([1]).to(device)
    return edge_recon_loss_norm_factor, edge_recon_loss_pos_weight


def compute_gene_expr_recon_zinb_loss(x: torch.Tensor,
                                      mu: torch.Tensor,
                                      theta: torch.Tensor,
                                      zi_prob_logits: torch.Tensor,
                                      eps=1e-8):
    """
    Gene expression reconstruction loss according to a ZINB gene expression 
    model, which is used to model scRNA-seq count data due to its capacity of 
    modeling excess zeros and overdispersion. The source code is adapted from
    https://github.com/scverse/scvi-tools/blob/master/scvi/distributions/_negative_binomial.py#L22.
    The Bernoulli distribution is parameterized using logits, hence the use of a
    softplus function.

    Parameters
    ----------
    x:
        Reconstructed feature matrix.
    mu:
        Mean of the negative binomial with positive support.
        (shape: batch_size x n_genes)
    theta:
        Inverse dispersion parameter with positive support.
        (shape: batch_size x n_genes)
    zi_prob_logits:
        Logits of the zero inflation probability with real support.
        (shape: batch_size x n_genes)
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

    case_non_zero = (-softplus_zi_prob_logits
                     + zi_prob_logits_theta_log
                     + x * (torch.log(mu + eps) - log_theta_mu_eps)
                     + torch.lgamma(x + theta)
                     - torch.lgamma(theta)
                     - torch.lgamma(x + 1))
                     
    mul_case_non_zero = torch.mul((x > eps).type(torch.float32), case_non_zero)

    log_likehood_zinb = mul_case_zero + mul_case_non_zero
    zinb_loss = -log_likehood_zinb.sum(-1)
    return zinb_loss