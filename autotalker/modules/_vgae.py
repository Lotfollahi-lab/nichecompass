from typing import Literal

import mlflow
import torch
import torch.nn as nn
from torch_geometric.data import Data

from autotalker.nn import GCNEncoder
from autotalker.nn import DotProductGraphDecoder
from ._losses import compute_vgae_loss
from ._losses import vgae_loss_parameters
from ._vgaemodulemixin import VGAEModuleMixin


class VGAE(nn.Module, VGAEModuleMixin):
    """
    Variational Graph Autoencoder class as per Kipf, T. N. & Welling, M.
    Variational Graph Auto-Encoders. arXiv [stat.ML] (2016).

    Parameters
    ----------
    n_input:
        Number of nodes in the input layer.
    n_hidden_encoder:
        Number of nodes in the encoder hidden layer.
    n_latent:
        Number of nodes in the latent space.
    dropout_rate_encoder:
        Probability that nodes will be dropped in the encoder during training.
    dropout_rate_graph_decoder:
        Probability that nodes will be dropped in the graph decoder during 
        training.
    """
    def __init__(self,
                 n_input: int,
                 n_hidden_encoder: int,
                 n_latent: int,
                 dropout_rate_encoder: float=0.0,
                 dropout_rate_graph_decoder: float=0.0):
        super().__init__()
        self.n_input = n_input
        self.n_hidden_encoder = n_hidden_encoder
        self.n_latent = n_latent
        self.dropout_rate_encoder = dropout_rate_encoder
        self.dropout_rate_graph_decoder = dropout_rate_graph_decoder

        print("--- INITIALIZING NEW NETWORK MODULE: VGAE ---")

        self.encoder = GCNEncoder(n_input=n_input,
                                  n_hidden=n_hidden_encoder,
                                  n_latent=n_latent,
                                  dropout_rate=dropout_rate_encoder,
                                  activation=torch.relu)
        
        self.decoder = DotProductGraphDecoder(
            dropout_rate=dropout_rate_graph_decoder)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        Forward pass of the VGAE module.

        Parameters
        ----------
        x:
            Tensor containing gene expression.
        edge_index:
            Tensor containing indeces of edges.

        Returns
        ----------
        output:
            Dictionary containing reconstructed adjacency matrix logits, mu and 
            logstd from the latent space distribution.
        """
        mu, logstd = self.encoder(x, edge_index)
        z = self.reparameterize(mu, logstd)
        adj_recon_logits = self.decoder(z)
        output = dict(adj_recon_logits=adj_recon_logits,
                      mu=mu,
                      logstd=logstd)
        return output

    def loss(self,
             model_output: dict,
             data_batch: Data,
             device: Literal["cpu", "cuda"]):
        """
        Calculate loss of the VGAE module.

        Parameters
        ----------
        model_output:
            Output of the forward pass.
        data_batch:
            PyG Data object containing a batch.
        device:
            Device wheere to send the loss parameters.

        Returns
        ----------
        loss:
            VGAE loss.
        """

        vgae_loss_params = vgae_loss_parameters(data_batch=data_batch,
                                                device=device)
        edge_recon_loss_norm_factor = vgae_loss_params[0]
        edge_recon_loss_pos_weight = vgae_loss_params[1]

        loss = compute_vgae_loss(
            adj_recon_logits=model_output["adj_recon_logits"],
            edge_label_index=data_batch.edge_label_index,
            edge_labels=data_batch.edge_label,
            edge_recon_loss_pos_weight=edge_recon_loss_pos_weight,
            edge_recon_loss_norm_factor=edge_recon_loss_norm_factor,
            mu=model_output["mu"],
            logstd=model_output["logstd"],
            n_nodes=data_batch.x.size(0))
        return loss

    def log_module_hyperparams_to_mlflow(self):
        """Log module hyperparameters to Mlflow."""
        mlflow.log_param("n_hidden", self.n_hidden_encoder)
        mlflow.log_param("n_latent", self.n_latent)
        mlflow.log_param("dropout_rate_encoder", 
                         self.dropout_rate_encoder)
        mlflow.log_param("dropout_rate_graph_decoder", 
                         self.dropout_rate_graph_decoder) 