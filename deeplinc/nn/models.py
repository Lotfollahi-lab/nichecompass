import torch
import torch.nn.functional as F
from modules import DotProductDecoder, FCLayer, GCNEncoder
from torch import nn as nn


class VGAE(nn.Module):
    """
    Variational Graph Autoencoder class as per Kipf, T. N. & Welling, M.
    Variational Graph Auto-Encoders. arXiv [stat.ML] (2016).

    Parameters
    ----------
    n_input
        Number of nodes in the input layer.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Number of nodes in the latent space.
    dropout
        Probability that nodes will be dropped during training.
    """

    def __init__(
        self, n_input: int, n_hidden: int, n_latent: int, dropout: float = 0.0
    ):
        super(VGAE, self).__init__()
        self.encoder = GCNEncoder(
            n_input=n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            dropout=dropout,
            activation=F.relu,
        )
        self.decoder = DotProductDecoder(droput=dropout, activation=F.sigmoid)

    def reparameterize(self, mu: torch.Tensor, logstd: torch.tensor):
        if self.training:
            std = torch.exp(logstd)
            eps = torch.randn_like(mu)
            return eps.mul(std).add(mu)
        else:
            return mu

    def forward(self, X, A):
        self.mu, self.logstd = self.encoder(X, A)
        self.Z = self.reparameterize(self.mu, self.logstd)
        A_pred = self.decoder(self.Z)
        return A_pred


class Discriminator(nn.Module):
    """
    Discriminator class. Adversarial network that takes a sample from the latent
    space as input and uses fully connected layers with a final sigmoid
    activation in the last layer to judge whether a sample is from a prior
    Gaussian distribution or from VGAE.

    Parameters
    ----------
    n_input
        Number of nodes in the input layer.
    n_hidden_l1
        Number of nodes in the first hidden layer.
    n_hidden_l2
        Number of nodes in the second hidden layer.
    activation
        Activation function used in the first and second hidden layer.
    """

    def __init__(
        self,
        n_input: int = 125,
        n_hidden_l1: int = 150,
        n_hidden_l2: int = 125,
        activation=F.relu,
    ):
        super(Discriminator, self).__init__()
        self.fc_l1 = FCLayer(n_input, n_hidden_l1, activation)
        self.fc_l2 = FCLayer(n_hidden_l1, n_hidden_l2, activation)
        self.fc_l3 = FCLayer(n_hidden_l2, 1, None)

    def forward(self, Z):
        hidden = self.fc_l1(Z)
        hidden = self.fc_l2(hidden)
        Y = self.fc_l3(hidden)
        return Y
