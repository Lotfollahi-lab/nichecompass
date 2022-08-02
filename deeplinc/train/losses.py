import torch
import torch.nn.functional as F
import torch.nn.modules.loss


def compute_gvae_loss(preds, labels, mu, logstd, n_nodes, norm_factor, pos_weight):
    """
    Compute the Graph Variational Autoencoder loss which consists of the
    weighted binary cross entropy loss (reconstruction loss), where positive
    examples (Aij = 1) are reweighted, as well as the Kullback-Leibler
    divergence (regularization term) as per Kingma, D. P. & Welling, M.
    Auto-Encoding Variational Bayes. arXiv [stat.ML] (2013).

    Parameters
    ----------
    preds
        Tensor that contains the adjacency matrix predictions.
    labels
        Tensor that contains the adjacency matrix labels.
    mu
        Expected values of the latent space distribution.
    logstd
        Log standard deviations of the latent space distribution.
    n_nodes
        Number of nodes of the GVAE.
    norm_factor
        Factor with which reconstruction loss is weighted compared to Kullback-
        Leibler divergence.
    pos_weight
        Weight with which positive examples (Aij = 1) are reweighted. This
        reweighting can be benefical for very sparse A.
    """

    weighted_bce = F.binary_cross_entropy_with_logits(
        preds,
        labels,
        pos_weight=pos_weight
    )

    kld = (-0.5 / n_nodes) * torch.mean(torch.sum(1 + 2 * logstd - mu.pow(2) - logstd.exp().pow(2), 1))

    print(f"BCE: {norm_factor * weighted_bce}")
    print(f"KLD: {kld}")
    cost = norm_factor * weighted_bce + kld
    return cost


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
