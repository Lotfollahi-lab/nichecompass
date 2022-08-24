import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import to_dense_adj


def compute_edge_recon_loss(adj_recon_logits,
                            edge_labels,
                            edge_label_index):
    """


    Parameters
    ----------
    adj_recon_logits:
    edge_labels:
    edge_label_index:

    Returns
    ----------
    edge_recon_loss:
        aa
    """
    # Create mask to retrieve values at edge_label_index from adj_recon_logits 
    mask = torch.squeeze(to_dense_adj(edge_label_index)) > 0

    # Pad mask on right and bottom to have same dimension as adj_recon_logits
    pad_dim = (torch.tensor(adj_recon_logits.shape[0]) - 
               torch.tensor(mask.shape[0])).item()
    padded_mask = F.pad(mask, (0, pad_dim, 0, pad_dim), "constant", False)
    predicted_edge_logits = torch.masked_select(adj_recon_logits, padded_mask)

    # Compute cross entropy loss
    edge_recon_loss = F.binary_cross_entropy_with_logits(predicted_edge_logits,
                                                         edge_labels)
    return edge_recon_loss


def compute_adj_recon_loss_old(adj_recon_logits, edge_label_index, pos_weight):
    """
    Compute adjacency reconstruction loss from logits output by model and edge
    label indices from data.

    Parameters
    ----------
    adj_recon_logits:
        Tensor containing the reconstructed adjacency matrix with logits.
    edge_label_index:
        Tensor containing the edge label indices.
    pos_weight:
        Weight with which positive examples (Aij = 1) are reweighted. This
        reweighting can be benefical for very sparse adjacency matrices.
    """
    adj_labels = to_dense_adj(add_self_loops(edge_label_index)[0])[0]
    adj_recon_loss = F.binary_cross_entropy_with_logits(
        adj_recon_logits,
        adj_labels,
        pos_weight=pos_weight)
    return adj_recon_loss


def compute_kl_loss(mu, logstd, n_nodes):
    """
    Compute Kullback-Leibler divergence as per Kingma, D. P. & Welling, M. 
    Auto-Encoding Variational Bayes. arXiv [stat.ML] (2013).
    """
    kl_loss = (-0.5 / n_nodes) * torch.mean(
    torch.sum(1 + 2 * logstd - mu ** 2 - torch.exp(logstd) ** 2, 1))
    return kl_loss


def compute_vgae_loss(adj_recon_logits: torch.Tensor,
                      edge_labels: torch.Tensor,
                      edge_label_index: torch.Tensor,
                      mu: torch.Tensor,
                      logstd: torch.Tensor,
                      n_nodes: int,
                      norm_factor: float):
    """
    Compute the Variational Graph Autoencoder loss which consists of the
    weighted binary cross entropy loss (reconstruction loss), where sparse 
    positive examples (Aij = 1) are reweighted, as well as the Kullback-Leibler
    divergence (regularization loss) as per Kingma, D. P. & Welling, M.
    Auto-Encoding Variational Bayes. arXiv [stat.ML] (2013). The reconstruction
    loss is normalized to bring it on the same scale as the regularization loss.

    Parameters
    ----------
    adj_recon_logits:
        Tensor containing the reconstructed adjacency matrix with logits.
    edge_label_index:
        Tensor containing the edge label indices.
    mu:
        Expected values of the latent space distribution.
    logstd:
        Log standard deviations of the latent space distribution.
    n_nodes:
        Number of nodes.
    norm_factor:
        Factor with which reconstruction loss is weighted compared to Kullback-
        Leibler divergence.
        
    Returns
    ----------
    vgae_loss:
        Variational Graph Autoencoder loss composed of reconstruction and 
        regularization loss.
    """
    edge_recon_loss = compute_edge_recon_loss(adj_recon_logits,
                                              edge_labels,
                                              edge_label_index)
    kl_loss = compute_kl_loss(mu, logstd, n_nodes)
    vgae_loss = norm_factor * edge_recon_loss + kl_loss
    return vgae_loss


def compute_vgae_loss_parameters(edge_label_index):
    """
    Compute parameters for the vgae loss function as per 
    https://github.com/tkipf/gae.git. A small adjustment is that adjacency 
    matrix with 1s on the diagonal is used to reflect real labels used for
    training. 
    
    Parameters
    ----------
    edge_label_index:
        Tensor containing the edge label indices.

    Returns
    ----------
    vgae_loss_norm_factor:
        Weight of reconstruction loss compared to Kullback-Leibler divergence.
    vgae_loss_pos_weight:
        Weight with which loss for positive labels (Aij = 1) is reweighted.
    """
    adj_labels = to_dense_adj(add_self_loops(edge_label_index)[0])[0]
    n_all_labels = adj_labels.shape[0] ** 2
    n_pos_labels = adj_labels.sum()
    n_neg_labels = n_all_labels - n_pos_labels
    neg_to_pos_label_ratio = n_neg_labels / n_pos_labels

    vgae_loss_norm_factor = n_all_labels / float(n_neg_labels * 2)

    # Reweight positive examples of edges (Aij = 1) in loss calculation 
    # using the proportion of negative examples relative to positive ones to
    # achieve equal total weighting of negative and positive examples
    vgae_loss_pos_weight = torch.FloatTensor([neg_to_pos_label_ratio])

    return vgae_loss_norm_factor, vgae_loss_pos_weight


def compute_vgae_loss_old(adj_recon_logits: torch.Tensor,
                      edge_label_index: torch.Tensor,
                      pos_weight: torch.Tensor,
                      mu: torch.Tensor,
                      logstd: torch.Tensor,
                      n_nodes: int,
                      norm_factor: float):
    """
    Compute the Variational Graph Autoencoder loss which consists of the
    weighted binary cross entropy loss (reconstruction loss), where sparse 
    positive examples (Aij = 1) are reweighted, as well as the Kullback-Leibler
    divergence (regularization loss) as per Kingma, D. P. & Welling, M.
    Auto-Encoding Variational Bayes. arXiv [stat.ML] (2013). The reconstruction
    loss is normalized to bring it on the same scale as the regularization loss.

    Parameters
    ----------
    adj_recon_logits:
        Tensor containing the reconstructed adjacency matrix with logits.
    edge_label_index:
        Tensor containing the edge label indices.
    pos_weight:
        Weight with which positive examples (Aij = 1) are reweighted. This
        reweighting can be benefical for very sparse adjacency matrices.
    mu:
        Expected values of the latent space distribution.
    logstd:
        Log standard deviations of the latent space distribution.
    n_nodes:
        Number of nodes.
    norm_factor:
        Factor with which reconstruction loss is weighted compared to Kullback-
        Leibler divergence.
        
    Returns
    ----------
    vgae_loss:
        Variational Graph Autoencoder loss composed of reconstruction and 
        regularization loss.
    """
    adj_recon_loss = compute_adj_recon_loss(
        adj_recon_logits,
        edge_label_index,
        pos_weight)
    kl_loss = compute_kl_loss(mu, logstd, n_nodes)
    vgae_loss = norm_factor * adj_recon_loss + kl_loss
    return vgae_loss


def compute_vgae_loss_parameters(edge_label_index):
    """
    Compute parameters for the vgae loss function as per 
    https://github.com/tkipf/gae.git. A small adjustment is that adjacency 
    matrix with 1s on the diagonal is used to reflect real labels used for
    training. 
    
    Parameters
    ----------
    edge_label_index:
        Tensor containing the edge label indices.

    Returns
    ----------
    vgae_loss_norm_factor:
        Weight of reconstruction loss compared to Kullback-Leibler divergence.
    vgae_loss_pos_weight:
        Weight with which loss for positive labels (Aij = 1) is reweighted.
    """
    adj_labels = to_dense_adj(add_self_loops(edge_label_index)[0])[0]
    n_all_labels = adj_labels.shape[0] ** 2
    n_pos_labels = adj_labels.sum()
    n_neg_labels = n_all_labels - n_pos_labels
    neg_to_pos_label_ratio = n_neg_labels / n_pos_labels

    vgae_loss_norm_factor = n_all_labels / float(n_neg_labels * 2)

    # Reweight positive examples of edges (Aij = 1) in loss calculation 
    # using the proportion of negative examples relative to positive ones to
    # achieve equal total weighting of negative and positive examples
    vgae_loss_pos_weight = torch.FloatTensor([neg_to_pos_label_ratio])

    return vgae_loss_norm_factor, vgae_loss_pos_weight


def compute_x_recon_mse_loss(x_recon: torch.Tensor,
                             x: torch.Tensor):
    """
    Compute MSE loss between reconstructed and ground truth feature matrix.

    Parameters
    ----------
    recon_x:
        Tensor containing reconstructed feature matrix.
    x:
        Tensor containing ground truth feature matrix.

    Returns
    ----------
    mse_loss:
        Mean squared error loss.
    """
    mse_loss = torch.nn.functional.mse_loss(x_recon, x)
    return mse_loss


def compute_x_recon_nb_loss(x: torch.Tensor,
                            mu: torch.Tensor,
                            theta: torch.Tensor,
                            eps=1e-8):
    """
    Computes negative binomial loss. Adapted from 
    https://github.com/theislab/scarches.

    Parameters
    ----------
    x: torch.Tensor
         Torch Tensor of ground truth data.
    mu: torch.Tensor
         Torch Tensor of means of the negative binomial (has to be positive support).
    theta: torch.Tensor
         Torch Tensor of inverse dispersion parameter (has to be positive support).
    eps: Float
         numerical stability constant.
    Returns
    ----------
    If 'mean' is 'True' NB loss value gets returned, otherwise Torch tensor of losses gets returned.
    """
    if theta.ndimension() == 1:
        theta = theta.view(1, theta.size(0))

    log_theta_mu_eps = torch.log(theta + mu + eps)
    res = (theta * (torch.log(theta + eps) - log_theta_mu_eps)
           + x * (torch.log(mu + eps) - log_theta_mu_eps)
           + torch.lgamma(x + theta)
           - torch.lgamma(theta)
           - torch.lgamma(x + 1))

    return -(res.sum(dim=-1).mean())