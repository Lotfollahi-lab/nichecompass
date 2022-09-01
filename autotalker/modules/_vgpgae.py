from pickletools import optimize
from typing import Literal

import mlflow
import torch
import torch.nn as nn
from torch_geometric.data import Data

from autotalker.nn import DotProductGraphDecoder
from autotalker.nn import GCNEncoder
from autotalker.nn import MaskedGeneExprDecoder
from ._losses import compute_edge_recon_loss
from ._losses import compute_gene_expr_recon_zinb_loss
from ._losses import compute_kl_loss
from ._losses import compute_vgae_loss
from ._losses import vgae_loss_parameters
from ._vgaemodulemixin import VGAEModuleMixin


class VGPGAE(nn.Module, VGAEModuleMixin):
    """
    Variational Gene Program Graph Autoencoder class.

    Parameters
    ----------
    n_input:
        Number of nodes in the input layer.
    n_hidden_encoder:
        Number of nodes in the encoder hidden layer.
    n_latent:
        Number of nodes in the latent space.
    gene_expr_decoder_mask:
        Gene program mask for the gene expression decoder.
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
                 gene_expr_decoder_mask: torch.Tensor,
                 dropout_rate_encoder: float=0.0,
                 dropout_rate_graph_decoder: float=0.0):
        super().__init__()
        self.n_input = n_input
        self.n_hidden_encoder = n_hidden_encoder
        self.n_latent = n_latent
        self.dropout_rate_encoder = dropout_rate_encoder
        self.dropout_rate_graph_decoder = dropout_rate_graph_decoder

        print("--- INITIALIZING NEW NETWORK MODULE: VGPGAE ---")

        self.encoder = GCNEncoder(n_input=n_input,
                                  n_hidden=n_hidden_encoder,
                                  n_latent=n_latent,
                                  dropout_rate=dropout_rate_encoder,
                                  activation=torch.relu)
        
        self.graph_decoder = DotProductGraphDecoder(
            dropout_rate=dropout_rate_graph_decoder)

        self.gene_expr_decoder = MaskedGeneExprDecoder(
            n_input=n_latent,
            n_output=n_input,
            mask=gene_expr_decoder_mask)
        
        # Gene-specific inverse dispersion parameters
        self.theta = torch.nn.Parameter(torch.randn(self.n_input))

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.Tensor,
                decoder: Literal["graph", "gene_expr"]="graph"):
        """
        Forward pass of the VGPGAE module.

        Parameters
        ----------
        x:
            Tensor containing gene expression.
        edge_index:
            Tensor containing indeces of edges.
        decoder:
            Which decoder to use for the forward pass, either "graph" for edge
            reconstruction or "gene_expr" for gene expression reconstruction.

        Returns
        ----------
        output:
            Dictionary containing reconstructed adjacency matrix logits, ZINB
            parameters for gene expression reconstruction, mu and logstd from
            the latent space distribution.
        """
        output = {}
        # Use observed library size as scaling factor in mean of ZINB 
        # distribution
        log_library_size = torch.log(x.sum(1)).unsqueeze(1)
        output["mu"], output["logstd"] = self.encoder(x, edge_index)
        z = self.reparameterize(output["mu"], output["logstd"])
        if decoder == "graph":
            output["adj_recon_logits"] = self.graph_decoder(z)
        elif decoder == "gene_expr":
            output["zinb_parameters"] = self.gene_expr_decoder(z,
                                                               log_library_size)
        return output

    def loss(self,
             edge_data_batch: Data,
             edge_model_output: dict,
             node_data_batch: Data,
             node_model_output: dict,
             device: Literal["cpu", "cuda"],
             include_edge_recon_loss: bool=True,
             include_gene_expr_recon_loss: bool=True):
        """
        Calculate loss of the VGPGAE module.

        Parameters
        ----------
        edge_data_batch:
            PyG Data object containing an edge-level batch.
        edge_model_output:
            Output of the forward pass for edge reconstruction.
        node_data_batch:
            PyG Data object containing a node-level batch.
        node_model_output:
            Output of the forward pass for gene expression reconstruction.
        device:
            Device where to send the loss parameters.
        include_edge_recon_loss:
            If `True` include the redge reconstruction loss in the loss 
            optimization.
        include_gene_expr_recon_loss:
            If `True`, include the gene expression reconstruction loss in the 
            loss optimization.

        Returns
        ----------
        loss_dict:
            Dictionary containing loss, edge reconstruction loss and gene
            expression reconstruction loss.
        """
        loss_dict = {}

        vgae_loss_params = vgae_loss_parameters(
            data_batch=edge_data_batch,
            device=device)
        edge_recon_loss_norm_factor = vgae_loss_params[0]
        edge_recon_loss_pos_weight = vgae_loss_params[1]

        loss_dict["edge_recon_loss"] = (edge_recon_loss_norm_factor * 
        compute_edge_recon_loss(
            adj_recon_logits=edge_model_output["adj_recon_logits"],
            edge_labels=edge_data_batch.edge_label,
            edge_label_index=edge_data_batch.edge_label_index,
            pos_weight=edge_recon_loss_pos_weight))

        loss_dict["kl_loss"] = compute_kl_loss(
            mu=edge_model_output["mu"],
            logstd=edge_model_output["logstd"],
            n_nodes=edge_data_batch.x.size(0))

        nb_means, zi_prob_logits = node_model_output["zinb_parameters"]

        # Gene-specific inverse dispersion
        theta = torch.exp(self.theta)

        loss_dict["gene_expr_recon_loss"] = compute_gene_expr_recon_zinb_loss(
            x=node_data_batch.x,
            mu=nb_means,
            theta=theta,
            zi_prob_logits=zi_prob_logits)

        loss_dict["loss"] = 0
        loss_dict["loss"] += loss_dict["kl_loss"]

        if include_edge_recon_loss:
            loss_dict["loss"] += loss_dict["edge_recon_loss"]

        if include_gene_expr_recon_loss:
            loss_dict["loss"] += loss_dict["gene_expr_recon_loss"]
        return loss_dict

    def log_module_hyperparams_to_mlflow(self):
        """Log module hyperparameters to Mlflow."""
        mlflow.log_param("n_hidden", self.n_hidden_encoder)
        mlflow.log_param("n_latent", self.n_latent)
        mlflow.log_param("dropout_rate_encoder", 
                         self.dropout_rate_encoder)
        mlflow.log_param("dropout_rate_graph_decoder", 
                         self.dropout_rate_graph_decoder) 