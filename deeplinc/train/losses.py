import torch
import torch.nn.functional as F
import torch.nn.modules.loss


def compute_vgae_loss(
        A_rec_logits: torch.tensor,
        A_label: torch.tensor,
        mu: torch.tensor,
        logstd: torch.tensor,
        n_nodes: int,
        norm_factor: float,
        pos_weight: torch.tensor,
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


def compute_adversarial_loss(preds,
                             preds_real,
                             preds_fake):
    """
    Compute the Discriminator loss of the adversarial module.

    Parameters
    ----------
    preds
        Tensor tha
    """

    # Adversarial ground truths
    dc_labels_real = torch.ones_like(preds_real)
    dc_labels_fake = torch.zeros_like(preds_fake)
    gen_labels = torch.ones_like(preds_fake)

    # Discriminator loss
    dc_bce_real = F.binary_cross_entropy_with_logits(preds_real, dc_labels_real)
    dc_bce_fake = F.binary_cross_entropy_with_logits(preds_fake, dc_labels_fake)

    # Generator loss
    gen_bce = F.binary_cross_entropy_with_logits(preds, gen_labels)
    
    return dc_bce_real + dc_bce_fake + gen_bce


def compute_combined_loss(preds, labels, mu, logstd, n_nodes, norm_factor, pos_weight):
    """
    Compute combined loss consisting of the Graph Variational Autoencoder loss
    and the loss of the adversarial module

    Parameters
    ----------

    """
    gvae_loss = compute_gvae_loss(
        preds, labels, mu, logstd, n_nodes, norm_factor, pos_weight
    )

    adversarial_loss = compute_adversarial_loss()

    return gvae_loss + adversarial_loss
