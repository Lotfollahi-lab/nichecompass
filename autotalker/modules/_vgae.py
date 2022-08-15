import torch
import torch.nn as nn

from autotalker.nn import DotProductDecoder
from autotalker.nn import GCNEncoder
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
        
        self.decoder = DotProductDecoder(dropout_rate=dropout_rate)


    def forward(self, x, edge_index):
        self.mu, self.logstd = self.encoder(x, edge_index)
        self.z = self.reparameterize(self.mu, self.logstd)
        adj_rec_logits = self.decoder(self.z)
        return adj_rec_logits, self.mu, self.logstd