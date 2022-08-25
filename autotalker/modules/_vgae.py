import torch
import torch.nn as nn

from autotalker.nn import DotProductGraphDecoder
from autotalker.nn import GCNEncoder
from ._losses import compute_vgae_loss
from ._vgaemodulemixin import VGAEModuleMixin


class VGAE(nn.Module, VGAEModuleMixin):
    """
    Variational Graph Autoencoder class as per Kipf, T. N. & Welling, M.
    Variational Graph Auto-Encoders. arXiv [stat.ML] (2016).

    Parameters
    ----------
    n_input:
        Number of nodes in the input layer.
    n_hidden:
        Number of nodes in the hidden layer.
    n_latent:
        Number of nodes in the latent space.
    dropout_rate:
        Probability that nodes will be dropped during training.
    """
    def __init__(self,
                 n_input: int,
                 n_hidden: int,
                 n_latent: int,
                 dropout_rate: float=0.0):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.dropout_rate = dropout_rate
        super().__init__()

        self.encoder = GCNEncoder(n_input=n_input,
                                  n_hidden=n_hidden,
                                  n_latent=n_latent,
                                  dropout_rate=dropout_rate,
                                  activation=torch.relu)
        
        self.decoder = DotProductGraphDecoder(dropout_rate=dropout_rate)

    def forward(self, x, edge_index):
        mu, logstd = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logstd)
        adj_recon_logits = self.decoder(z)
        return adj_recon_logits, mu, logstd

    def loss(self, adj_recon_logits, data_batch, mu, logstd, device):
        n_possible_edges = data_batch.x.shape[0] ** 2
        n_neg_edges = (data_batch.edge_label == 0).sum()
        edge_recon_loss_norm_factor = n_possible_edges / n_neg_edges
        edge_recon_loss_pos_weight = torch.Tensor([1]).to(device)

        vgae_loss = compute_vgae_loss(
            adj_recon_logits=adj_recon_logits,
            edge_label_index=data_batch.edge_label_index,
            edge_labels=data_batch.edge_label,
            edge_recon_loss_pos_weight=edge_recon_loss_pos_weight,
            edge_recon_loss_norm_factor=edge_recon_loss_norm_factor,
            mu=mu,
            logstd=logstd,
            n_nodes=data_batch.x.size(0))
        return vgae_loss