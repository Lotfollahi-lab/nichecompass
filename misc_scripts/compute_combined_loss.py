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
