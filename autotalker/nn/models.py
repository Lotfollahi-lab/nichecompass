import anndata as ad
import torch
import torch.nn.functional as F
from torch import nn as nn

from .modules import DotProductDecoder, FCLayer, GCNEncoder

class Deeplinc(nn.Module):
    """
    ### WIP ###
    Deeplinc model class as per Li, R. & Yang, X. De novo reconstruction of cell
    interaction landscapes from single-cell spatial transcriptome data with 
    DeepLinc. Genome Biol. 23, 124 (2022).

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
    def __init__(self,
                 adata: ad.AnnData,
                 n_input: int,
                 n_hidden: int,
                 n_latent: int,
                 n_hidden1_disc: int,
                 n_hidden2_disc: int,
                 dropout: float = 0.0):
        super().__init__()
        self.vgae = VGAE()
        self.discriminator = Discriminator()
        
        self.is_trained_ = False
        self.trainer = None

    def train(self,
              n_epochs: int = 400,
              lr: float = 1e-3,
              eps: float = 0.01,
              **kwargs):
        """
        Train the model.

        Parameters
        ----------
        n_epochs:
            Number of epochs.
        lr:
            Learning rate.
        eps:
            torch.optim.Adam eps parameter.
        kwargs:
            Keyword arguments for the Deeplinc trainer.
        """
        self.trainer = None
        self.trainer.train(n_epochs, lr, eps)
        self.is_trained_ = True
        return 1
    def predict():
        return 1


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
            self,
            n_input: int,
            n_hidden: int,
            n_latent: int,
            dropout: float = 0.0):
        super(VGAE, self).__init__()
        self.encoder = GCNEncoder(
            n_input = n_input,
            n_hidden = n_hidden,
            n_latent = n_latent,
            dropout = dropout,
            activation = torch.relu)
        self.decoder = DotProductDecoder(dropout = dropout)

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
        A_rec_logits = self.decoder(self.Z)
        return A_rec_logits, self.mu, self.logstd


class Discriminator(nn.Module):
    """
    ### WIP ###
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
