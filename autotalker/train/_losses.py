import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from matplotlib.ticker import MaxNLocator
from torch_geometric.utils import add_self_loops
from torch_geometric.utils import to_dense_adj


def compute_adj_recon_loss(adj_recon_logits, edge_label_index, pos_weight):
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


def compute_vgae_loss(
        adj_recon_logits: torch.Tensor,
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


def plot_loss_curves(loss_dict):
    """
    Plot loss curves.

    Parameters
    ----------
    loss_dict:
        Dictionary containing the training and validation losses.
    """
    # Plot epochs as integers
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Plot loss
    for loss_key, loss in loss_dict.items():
        plt.plot(loss, label = loss_key) 
    plt.title(f"Loss curves")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(loc = "upper right")

    # Retrieve figure
    fig = plt.gcf()
    return fig