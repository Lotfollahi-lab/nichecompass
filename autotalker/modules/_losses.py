import torch
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import to_dense_adj


def compute_edge_recon_loss(adj_recon_logits: torch.Tensor,
                            edge_labels: torch.Tensor,
                            edge_label_index: torch.Tensor):
    """
    Compute edge reconstruction binary cross entropy loss with logits using edge
    labels and predicted edge logits (retrieved from the reconstructed logits
    adjacency matrix).

    Parameters
    ----------
    adj_recon_logits:
        Adjacency matrix containing the predicted edge logits.
    edge_labels:
        Edge ground truth labels. Should contain equal number of 1s and
        negatively sampled 0s.
    edge_label_index:
        Index with edge labels for both positive and negatively sampled edges.

    Returns
    ----------
    edge_recon_loss:
        Binary cross entropy loss between edge labels and predicted edge
        probabilities (calculated from logits for numerical stability in
        backpropagation).
    """
    neg_sampling_double_edges = True
    pos_weight = torch.Tensor([1])
    
    if neg_sampling_double_edges:
        # this works even in the case of double edges from neg sampling
        for i, edge in enumerate(zip(edge_label_index[0], edge_label_index[1])):
            if i == 0:
                edge_recon_logits = torch.unsqueeze(
                    adj_recon_logits[edge[0], edge[1]], dim=-1)
            else:
                edge_recon_logits = torch.cat((
                    edge_recon_logits,
                    torch.unsqueeze(adj_recon_logits[edge[0], edge[1]], dim=-1)))
    
        # Compute cross entropy loss
        edge_recon_loss = F.binary_cross_entropy_with_logits(edge_recon_logits,
                                                             edge_labels,
                                                             pos_weight=pos_weight)
    
    else:
        # Create mask to retrieve values at ´edge_label_index´ from 
        # ´adj_recon_logits´
        n_nodes = adj_recon_logits.shape[0]
        adj_labels = to_dense_adj(edge_label_index, max_num_nodes=n_nodes)
        mask = torch.squeeze(adj_labels > 0)
    
        # Retrieve logits for edges from ´adj_recon_logits´
        edge_recon_logits = torch.masked_select(adj_recon_logits, mask)
    
        # Order edge labels based on sorted edge_label_index to align order with
        # masked retrieval from adjacency matrix
        sort_index = edge_label_index[0].sort(dim=-1).indices
        edge_labels_sorted = edge_labels[sort_index]
    
        # Compute cross entropy loss
        edge_recon_loss = F.binary_cross_entropy_with_logits(edge_recon_logits,
                                                             edge_labels_sorted,
                                                             pos_weight=pos_weight)
    return edge_recon_loss



def compute_kl_loss(mu, logstd, n_nodes):
    """
    Compute Kullback-Leibler divergence as per Kingma, D. P. & Welling, M. 
    Auto-Encoding Variational Bayes. arXiv [stat.ML] (2013).

    Parameters
    ----------
    mu:
    logstd:
    n_nodes:

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
                      mu: torch.Tensor,
                      logstd: torch.Tensor,
                      n_nodes: int,
                      edge_recon_loss_norm_factor: float):
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
    edge_recon_loss_norm_factor:
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

    vgae_loss = edge_recon_loss_norm_factor * edge_recon_loss + kl_loss
    return vgae_loss


def compute_feature_recon_mse_loss(x_recon: torch.Tensor,
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


def compute_feature_recon_nb_loss(x: torch.Tensor,
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