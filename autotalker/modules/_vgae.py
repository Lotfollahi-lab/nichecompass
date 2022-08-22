import torch
import torch.nn as nn

from autotalker.nn import DotProductGraphDecoder
from autotalker.nn import GCNEncoder
from ._vgaemodulemixin import VGAEModuleMixin
from ._losses import compute_vgae_loss
from ._losses import compute_vgae_loss_parameters


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
    def __init__(
            self,
            n_input: int,
            n_hidden: int,
            n_latent: int,
            dropout_rate: float=0.0):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.dropout_rate = dropout_rate
        super().__init__()
        self.encoder = GCNEncoder(
            n_input = n_input,
            n_hidden = n_hidden,
            n_latent = n_latent,
            dropout_rate = dropout_rate,
            activation = torch.relu)
        
        self.decoder = DotProductGraphDecoder(dropout_rate=dropout_rate)


    def forward(self, x, edge_index):
        self.mu, self.logstd = self.encoder(x, edge_index)
        self.z = self.reparameterize(self.mu, self.logstd)
        adj_recon_logits = self.decoder(self.z)
        return adj_recon_logits, self.mu, self.logstd


    def loss(self, adj_recon_logits, train_data, mu, logstd, device):
        vgae_loss_norm_factor, vgae_loss_pos_weight = \
        compute_vgae_loss_parameters(train_data.edge_index)

        vgae_loss_pos_weight = vgae_loss_pos_weight.to(device)

        vgae_loss = compute_vgae_loss(
            adj_recon_logits=adj_recon_logits,
            edge_label_index=train_data.edge_index,
            pos_weight=vgae_loss_pos_weight,
            mu=mu,
            logstd=logstd,
            n_nodes=train_data.x.size(0),
            norm_factor=vgae_loss_norm_factor)

        return vgae_loss