"""
This module contains the Variational Gene Program Graph Autoencoder, the core
module of the Autotalker model.
"""

from typing import Literal

import mlflow
import torch
import torch.nn as nn
from torch_geometric.data import Data

from autotalker.nn import (OneHopAttentionNodeLabelAggregator,
                           OneHopGCNNormNodeLabelAggregator,
                           SelfNodeLabelNoneAggregator,
                           OneHopSumNodeLabelAggregator,
                           DotProductGraphDecoder,
                           GCNEncoder,
                           MaskedGeneExprDecoder)
from .basemodulemixin import BaseModuleMixin
from .losses import (compute_addon_l1_reg_loss,
                     compute_edge_recon_loss, 
                     compute_gene_expr_recon_zinb_loss,
                     compute_group_lasso_reg_loss,
                     compute_kl_loss)
from .vgaemodulemixin import VGAEModuleMixin


class VGPGAE(nn.Module, BaseModuleMixin, VGAEModuleMixin):
    """
    Variational Gene Program Graph Autoencoder class.

    Parameters
    ----------
    n_input:
        Number of nodes in the input layer.
    n_hidden_encoder:
        Number of nodes in the encoder hidden layer.
    n_latent:
        Number of nodes in the latent space (gene programs from the gene program
        mask).
    n_addon_latent:
        Number of add-on nodes in the latent space (new gene programs).
    n_output:
        Number of nodes in the output layer.
    gene_expr_decoder_mask:
        Gene program mask for the gene expression decoder.
    dropout_rate_encoder:
        Probability that nodes will be dropped in the encoder during training.
    dropout_rate_graph_decoder:
        Probability that nodes will be dropped in the graph decoder during 
        training.
    include_edge_recon_loss:
        If `True`, include the redge reconstruction loss in the loss 
        optimization.
    include_gene_expr_recon_loss:
        If `True`, include the gene expression reconstruction loss in the 
        loss optimization.
    log_variational:
        If ´True´, transform x by log(x+1) prior to encoding for numerical 
        stability. Not normalization.

    """
    def __init__(self,
                 n_input: int,
                 n_hidden_encoder: int,
                 n_latent: int,
                 n_addon_latent: int,
                 n_output: int,
                 gene_expr_decoder_mask: torch.Tensor,
                 dropout_rate_encoder: float=0.0,
                 dropout_rate_graph_decoder: float=0.0,
                 include_edge_recon_loss: bool=True,
                 include_gene_expr_recon_loss: bool=True,
                 node_label_method: Literal[
                    "self",
                    "one-hop-norm",
                    "one-hop-sum",
                    "one-hop-attention"]="one-hop-attention",
                 log_variational: bool=True):
        super().__init__()
        self.n_input = n_input
        self.n_hidden_encoder = n_hidden_encoder
        self.n_latent = n_latent
        self.n_addon_latent = n_addon_latent
        self.n_output = n_output
        self.dropout_rate_encoder = dropout_rate_encoder
        self.dropout_rate_graph_decoder = dropout_rate_graph_decoder
        self.include_edge_recon_loss = include_edge_recon_loss
        self.include_gene_expr_recon_loss = include_gene_expr_recon_loss
        self.node_label_method = node_label_method
        self.log_variational = log_variational
        self.freeze = False

        print("--- INITIALIZING NEW NETWORK MODULE: VARIATIONAL GENE PROGRAM "
              "GRAPH AUTOENCODER ---")
        print(f"LOSS -> include_edge_recon_loss: {include_edge_recon_loss}, "
              f"include_gene_expr_recon_loss: {include_gene_expr_recon_loss}")
        print(f"NODE LABEL METHOD -> {node_label_method}")

        self.encoder = GCNEncoder(n_input=n_input,
                                  n_hidden=n_hidden_encoder,
                                  n_latent=n_latent,
                                  n_addon_latent=n_addon_latent,
                                  dropout_rate=dropout_rate_encoder,
                                  activation=torch.relu)
        
        self.graph_decoder = DotProductGraphDecoder(
            dropout_rate=dropout_rate_graph_decoder)

        self.gene_expr_decoder = MaskedGeneExprDecoder(
            n_input=n_latent,
            n_output=n_output,
            mask=gene_expr_decoder_mask,
            n_addon_input=n_addon_latent)

        if node_label_method == "self":
            self.gene_expr_node_label_aggregator = (
                SelfNodeLabelNoneAggregator())
        elif node_label_method == "one-hop-norm":
            self.gene_expr_node_label_aggregator = (
                OneHopGCNNormNodeLabelAggregator())
        elif node_label_method == "one-hop-sum":
            self.gene_expr_node_label_aggregator = (
                OneHopSumNodeLabelAggregator()) 
        elif node_label_method == "one-hop-attention": 
            self.gene_expr_node_label_aggregator = (
                OneHopAttentionNodeLabelAggregator(n_input=n_input))
        
        # Gene-specific dispersion parameters
        self.theta = torch.nn.Parameter(torch.randn(self.n_output))

    def forward(self,
                data_batch: Data,
                decoder: Literal["graph", "gene_expr"]="graph") -> dict:
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
        x = data_batch.x
        edge_index = data_batch.edge_index
        output = {}
        # Use observed library size as scaling factor in mean of ZINB 
        # distribution
        log_library_size = torch.log(x.sum(1)).unsqueeze(1)
        
        # Convert gene expression for numerical stability
        if self.log_variational:
            x_enc = torch.log(1 + x)
        else:
            x_enc = x

        output["mu"], output["logstd"] = self.encoder(x_enc, edge_index)
        
        z = self.reparameterize(output["mu"], output["logstd"])
        if decoder == "graph":
            output["adj_recon_logits"] = self.graph_decoder(z)
        elif decoder == "gene_expr":
            # Compute aggregated neighborhood gene expression for gene 
            # expression reconstruction        
            output["node_labels"] = self.gene_expr_node_label_aggregator(
                x, 
                edge_index,
                data_batch.batch_size)

            output["zinb_parameters"] = self.gene_expr_decoder(
                    z[:data_batch.batch_size],
                    log_library_size[:data_batch.batch_size])
        return output

    def loss(self,
             edge_data_batch: Data,
             edge_model_output: dict,
             node_model_output: dict,
             device: Literal["cpu", "cuda"],
             lambda_l1_addon: float=0.,
             lambda_group_lasso: float=0) -> dict:
        """
        Calculate loss of the VGPGAE module.

        Parameters
        ----------
        edge_data_batch:
            PyG Data object containing an edge-level batch.
        edge_model_output:
            Output of the forward pass for edge reconstruction.
        node_model_output:
            Output of the forward pass for gene expression reconstruction.
        device:
            Device where to send the loss parameters.
        lambda_l1_addon:
            Lambda (weighting) parameter for the L1 regularization of genes in 
            addon gene programs.
        lambda_group_lasso:
            Lambda (weighting) parameter for the group lasso regularization of 
            gene programs.

        Returns
        ----------
        loss_dict:
            Dictionary containing loss and all loss components.
        """
        loss_dict = {}

        # Compute loss parameters
        n_possible_edges = edge_data_batch.x.shape[0] ** 2
        n_neg_edges = n_possible_edges - edge_data_batch.edge_index.shape[1]
        # Factor with which edge reconstruction loss is weighted compared to 
        # Kullback-Leibler divergence and gene expression reconstruction loss.
        edge_recon_loss_norm_factor = n_possible_edges / (n_neg_edges * 2)
        # Weight with which positive examples are reweighted in the 
        # reconstruction loss calculation. Should be 1 if negative sampling 
        # ratio is 1. 
        edge_recon_loss_pos_weight = torch.Tensor([1]).to(device)
        # Factor with which gene expression reconstruction loss is weighted 
        # compared to Kullback-Leibler divergence and edge reconstruction loss.
        gene_expr_recon_loss_norm_factor = 1

        loss_dict["kl_loss"] = compute_kl_loss(
            mu=edge_model_output["mu"],
            logstd=edge_model_output["logstd"],
            n_nodes=edge_data_batch.x.size(0))

        # Gene-specific inverse dispersion
        theta = torch.exp(self.theta)

        loss_dict["loss"] = 0
        loss_dict["loss"] += loss_dict["kl_loss"]

        if self.include_edge_recon_loss:
            loss_dict["edge_recon_loss"] = (edge_recon_loss_norm_factor * 
                compute_edge_recon_loss(
                    adj_recon_logits=edge_model_output["adj_recon_logits"],
                    edge_labels=edge_data_batch.edge_label,
                    edge_label_index=edge_data_batch.edge_label_index,
                    pos_weight=edge_recon_loss_pos_weight))
            loss_dict["loss"] += loss_dict["edge_recon_loss"]

        if self.include_gene_expr_recon_loss:
            nb_means, zi_prob_logits = node_model_output["zinb_parameters"]
            
            loss_dict["gene_expr_recon_loss"] = (
                gene_expr_recon_loss_norm_factor * 
                    compute_gene_expr_recon_zinb_loss(
                        x=node_model_output["node_labels"],
                        mu=nb_means,
                        theta=theta,
                        zi_prob_logits=zi_prob_logits))
            loss_dict["loss"] += loss_dict["gene_expr_recon_loss"]

            loss_dict["group_lasso_reg_loss"] = (lambda_group_lasso * 
                compute_group_lasso_reg_loss(self.named_parameters()))
            loss_dict["loss"] += loss_dict["group_lasso_reg_loss" ]

            if self.n_addon_latent != 0:
                loss_dict["addon_gp_l1_reg_loss"] = (lambda_l1_addon * 
                    compute_addon_l1_reg_loss(self.named_parameters()))
                loss_dict["loss"] += loss_dict["addon_gp_l1_reg_loss"]
        return loss_dict

    def log_module_hyperparams_to_mlflow(self):
        """Log module hyperparameters to Mlflow."""
        mlflow.log_param("n_hidden", self.n_hidden_encoder)
        mlflow.log_param("n_latent", self.n_latent)
        mlflow.log_param("dropout_rate_encoder", 
                         self.dropout_rate_encoder)
        mlflow.log_param("dropout_rate_graph_decoder", 
                         self.dropout_rate_graph_decoder)
        mlflow.log_param("include_edge_recon_loss", 
                         self.include_edge_recon_loss)
        mlflow.log_param("include_gene_expr_recon_loss", 
                         self.include_gene_expr_recon_loss)
        mlflow.log_param("node_label_method", 
                         self.node_label_method)
        mlflow.log_param("log_variational", 
                         self.log_variational)
