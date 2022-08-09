import torch
import torch.nn.functional as F


def compute_adj_recon_loss(adj_recon_logits, adj_labels, pos_weight):
    F.binary_cross_entropy_with_logits(
        A_rec_logits,
        A_label,
        pos_weight=pos_weight)



def compute_vgae_loss(
        A_rec_logits: torch.Tensor,
        A_label: torch.Tensor,
        mu: torch.Tensor,
        logstd: torch.Tensor,
        n_nodes: int,
        norm_factor: float,
        pos_weight: torch.Tensor,
        debug: bool=False):
    """
    Compute the Variational Graph Autoencoder loss which consists of the
    weighted binary cross entropy loss (reconstruction loss), where sparse 
    positive examples (Aij = 1) are reweighted, as well as the Kullback-Leibler
    divergence (regularization loss) as per Kingma, D. P. & Welling, M.
    Auto-Encoding Variational Bayes. arXiv [stat.ML] (2013). The reconstruction
    loss is normalized to bring it on the same scale as the regularization loss.

    Parameters
    ----------
    A_rec_logits:
        Tensor containing the reconstructed adjacency matrix with logits.
    A_label:
        Tensor containing the adjacency matrix with labels.
    mu:
        Expected values of the latent space distribution.
    logstd:
        Log standard deviations of the latent space distribution.
    n_nodes:
        Number of nodes.
    norm_factor:
        Factor with which reconstruction loss is weighted compared to Kullback-
        Leibler divergence.
    pos_weight:
        Weight with which positive examples (Aij = 1) are reweighted. This
        reweighting can be benefical for very sparse A.
    Returns
    ----------
    vgae_loss:
        Variational Graph Autoencoder loss composed of reconstruction and 
        regularization loss.
    """

    weighted_bce_loss = norm_factor * F.binary_cross_entropy_with_logits(
        A_rec_logits,
        A_label,
        pos_weight=pos_weight)

    kl_divergence = (-0.5 / n_nodes) * torch.mean(
        torch.sum(1 + 2 * logstd - mu ** 2 - torch.exp(logstd) ** 2, 1))

    if debug:
        print(f"Weighted BCE: {weighted_bce_loss}")
        print(f"KLD: {kl_divergence}", "\n")

    vgae_loss = weighted_bce_loss + kl_divergence
    vgae_loss = weighted_bce_loss
    return vgae_loss


def compute_vgae_loss_parameters(A_label):
    """
    Compute parameters for the vgae loss function as per 
    https://github.com/tkipf/gae.git. A small adjustment is that adjacency 
    matrix with 1s on the diagonal is used to reflect real labels used for
    training. 
    
    Parameters
    ----------
    A_label:
        Adjacency matrix with labels (1s for training edges and diagonals)
    Returns
    ----------
    vgae_loss_norm_factor:
        Weight of reconstruction loss compared to Kullback-Leibler divergence.
    vgae_loss_pos_weight:
        Weight with which loss for positive labels (Aij = 1) is reweighted.
    """
    n_all_labels = A_label.shape[0] ** 2
    n_pos_labels = A_label.sum()
    n_neg_labels = n_all_labels - n_pos_labels
    neg_to_pos_label_ratio = n_neg_labels / n_pos_labels

    vgae_loss_norm_factor = n_all_labels / float(n_neg_labels * 2)

    # Reweight positive examples of edges (Aij = 1) in loss calculation 
    # using the proportion of negative examples relative to positive ones to
    # achieve equal total weighting of negative and positive examples
    vgae_loss_pos_weight = torch.FloatTensor([neg_to_pos_label_ratio])

    return vgae_loss_norm_factor, vgae_loss_pos_weight


def compute_adversarial_loss(preds_real,
                             preds_fake):
    """
    Compute the Discriminator loss of the adversarial module.

    Parameters
    ----------
    preds_real:

    preds_fake:

    Returns
    ----------
    dc_bce:
        Binary cross entropy loss of the discriminator module.
    gen_bce:
        Binary cross entropy loss of the generator module.
    """

    # Adversarial ground truths
    dc_labels_real = torch.ones_like(preds_real)
    dc_labels_fake = torch.zeros_like(preds_fake)
    gen_labels = torch.ones_like(preds_fake)

    # Discriminator loss
    dc_bce_real = F.binary_cross_entropy_with_logits(preds_real, dc_labels_real)
    dc_bce_fake = F.binary_cross_entropy_with_logits(preds_fake, dc_labels_fake)
    dc_bce = dc_bce_real + dc_bce_fake

    # Generator loss
    gen_bce = F.binary_cross_entropy_with_logits(preds_fake, gen_labels)

    return dc_bce, gen_bce