"""
This module contains the Variational Gene Program Graph Autoencoder, the neural
network module underlying the Autotalker model.
"""

from typing import Literal, Tuple, Union

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
                     compute_gene_expr_recon_nb_loss,
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
    n_addon_gps:
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
    gene_expr_recon_dist:
        The distribution used for gene expression reconstruction. If `nb`, uses
        a Negative Binomial distribution. If `zinb`, uses a Zero-inflated
        Negative Binomial distribution.
    log_variational:
        If ´True´, transform x by log(x+1) prior to encoding for numerical 
        stability. Not normalization.
    use_only_active_gps:
        If ´True´, filter the latent features / gene programs for edge
        reconstruction to allow only the gene programs with the highest gene
        program gene expression decoder weight sums to be included in the
        dot product and, thus, contribute to edge reconstruction.
    active_gp_thresh_ratio:
        If ´use_only_active_gps´ is ´True´, this is the filter ratio
        relative to the maximum gene program weight sum that a gene program's
        weights must sum to to be included in the edge reconstruction.
    """
    def __init__(self,
                 n_input: int,
                 n_hidden_encoder: int,
                 n_latent: int,
                 n_addon_gps: int,
                 n_output: int,
                 gene_expr_decoder_mask: torch.Tensor,
                 dropout_rate_encoder: float=0.0,
                 dropout_rate_graph_decoder: float=0.0,
                 include_edge_recon_loss: bool=True,
                 include_gene_expr_recon_loss: bool=True,
                 gene_expr_recon_dist: Literal["nb", "zinb"]="nb",
                 node_label_method: Literal[
                    "self",
                    "one-hop-norm",
                    "one-hop-sum",
                    "one-hop-attention"]="one-hop-attention",
                 log_variational: bool=True,
                 use_only_active_gps: bool=True,
                 active_gp_thresh_ratio: float=0.2):
        super().__init__()
        self.n_input = n_input
        self.n_hidden_encoder = n_hidden_encoder
        self.n_latent = n_latent
        self.n_addon_gps = n_addon_gps
        self.n_output = n_output
        self.dropout_rate_encoder = dropout_rate_encoder
        self.dropout_rate_graph_decoder = dropout_rate_graph_decoder
        self.include_edge_recon_loss = include_edge_recon_loss
        self.include_gene_expr_recon_loss = include_gene_expr_recon_loss
        self.gene_expr_recon_dist = gene_expr_recon_dist
        self.node_label_method = node_label_method
        self.log_variational = log_variational
        self.use_only_active_gps = use_only_active_gps
        self.active_gp_thresh_ratio = active_gp_thresh_ratio
        self.freeze = False

        print("--- INITIALIZING NEW NETWORK MODULE: VARIATIONAL GENE PROGRAM "
              "GRAPH AUTOENCODER ---")
        print(f"LOSS -> include_edge_recon_loss: {include_edge_recon_loss}, "
              f"include_gene_expr_recon_loss: {include_gene_expr_recon_loss}, "
              f"gene_expr_recon_dist: {gene_expr_recon_dist}")
        print(f"NODE LABEL METHOD -> {node_label_method}")

        self.encoder = GCNEncoder(n_input=n_input,
                                  n_hidden=n_hidden_encoder,
                                  n_latent=n_latent,
                                  n_addon_latent=n_addon_gps,
                                  dropout_rate=dropout_rate_encoder,
                                  activation=torch.relu)
        
        self.graph_decoder = DotProductGraphDecoder(
            dropout_rate=dropout_rate_graph_decoder)

        self.gene_expr_decoder = MaskedGeneExprDecoder(
            n_input=n_latent,
            n_output=n_output,
            mask=gene_expr_decoder_mask,
            n_addon_input=n_addon_gps,
            gene_expr_recon_dist=self.gene_expr_recon_dist)

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
        # Use observed library size as scaling factor in nb mean of gene
        # expression reconstruction distribution
        self.log_library_size = torch.log(x.sum(1)).unsqueeze(1)
        
        # Convert gene expression for numerical stability
        if self.log_variational:
            x_enc = torch.log(1 + x)
        else:
            x_enc = x

        self.mu, self.logstd = self.encoder(x_enc, edge_index)
        output["mu"] = self.mu
        output["logstd"] = self.logstd
        
        z = self.reparameterize(output["mu"], output["logstd"])
        if decoder == "graph":
            if self.use_only_active_gps == True:
                active_gp_mask = self.get_active_gp_mask(
                    abs_gp_weights_agg_mode="sum+nzmeans")
                z = z[:, active_gp_mask]
            output["adj_recon_logits"] = self.graph_decoder(z)
        elif decoder == "gene_expr":
            # Compute aggregated neighborhood gene expression for gene 
            # expression reconstruction        
            output["node_labels"] = self.gene_expr_node_label_aggregator(
                x, 
                edge_index,
                data_batch.batch_size)

            output["gene_expr_decoder_params"] = self.gene_expr_decoder(
                    z[:data_batch.batch_size],
                    self.log_library_size[:data_batch.batch_size])
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
            if self.gene_expr_recon_dist == "nb":
                nb_means = node_model_output["gene_expr_decoder_params"]
                loss_dict["gene_expr_recon_loss"] = (
                gene_expr_recon_loss_norm_factor * 
                    compute_gene_expr_recon_nb_loss(
                        x=node_model_output["node_labels"],
                        mu=nb_means,
                        theta=theta))
            elif self.gene_expr_recon_dist == "zinb":
                nb_means, zi_prob_logits = (
                node_model_output["gene_expr_decoder_params"])
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

            if self.n_addon_gps != 0:
                loss_dict["addon_gp_l1_reg_loss"] = (lambda_l1_addon * 
                    compute_addon_l1_reg_loss(self.named_parameters()))
                loss_dict["loss"] += loss_dict["addon_gp_l1_reg_loss"]
        return loss_dict

    def get_active_gp_mask(
            self,
            abs_gp_weights_agg_mode: Literal["sum",
                                             "nzmeans",
                                             "sum+nzmeans"]="sum+nzmeans",
            return_gp_weights: bool=False
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a mask of active gene programs based on the gene expression decoder
        gene weights of gene programs. Active gene programs are gene programs
        whose absolute gene weights aggregated over all genes are greater than 
        ´self.active_gp_thresh_ratio´ times the absolute gene weights
        aggregation of the gene program with the maximum value across all gene 
        programs. Depending on ´abs_gp_weights_agg_mode´, the aggregation will 
        be either a sum of absolute gene weights (prioritizes gene programs that
        reconstruct many genes) or a mean of non-zero absolute gene weights 
        (normalizes for the number of genes that a gene program reconstructs) or
        a combination of the two.

        Parameters
        ----------
        abs_gp_weights_agg_mode:
            If ´sum´, uses sums of absolute gp weights for aggregation and
            active gp determination. If ´nzmeans´, uses means of non-zero 
            absolute gp weights for aggregation and active gp determination. If
            ´sum+nzmeans´, uses a combination of sums and means of non-zero
            absolute gp weights for aggregation and active gp determination.
        return_gp_weights:
            If ´True´, in addition return the gene expression decoder gene 
            weights of the active gene programs.

        Returns
        ----------
        active_gp_mask:
            Boolean tensor of gene programs which contains `True` for active
            gene programs and `False` for inactive gene programs.
        active_gp_weights:
            Tensor containing the gene expression decoder gene weights of active
            gene programs.
        """
        # Get gp gene expression decoder gene weights
        gp_weights = (self.gene_expr_decoder.nb_means_normalized_decoder
                      .masked_l.weight.data).clone().detach()
        if self.n_addon_gps > 0:
            gp_weights = torch.cat(
                [gp_weights, 
                 (self.gene_expr_decoder.nb_means_normalized_decoder.addon_l
                  .weight.data).clone().detach()])

        # Correct gp weights for zero inflation using zero inflation 
        # probabilities over all observations if zinb distribution is used to 
        # model gene expression
        if self.gene_expr_recon_dist == "zinb":
            _, zi_probs = self.get_gene_expr_decoder_params(
                z=self.mu,
                log_library_size=self.log_library_size)
            non_zi_probs = 1 - zi_probs
            non_zi_probs_sum = non_zi_probs.sum(0).unsqueeze(1) # sum over obs 
            gp_weights *= non_zi_probs_sum 

        # Aggregate absolute gp weights based on ´abs_gp_weights_agg_mode´ and 
        # calculate thresholds of aggregated absolute gp weights and get active
        # gp mask and (optionally) active gp weights
        abs_gp_weights_sums = gp_weights.norm(p=1, dim=0)
        if abs_gp_weights_agg_mode in ["sum", "sum+nzmeans"]:
            max_abs_gp_weights_sum = max(abs_gp_weights_sums)
            min_abs_gp_weights_sum_thresh = (self.active_gp_thresh_ratio * 
                                             max_abs_gp_weights_sum)
            active_gp_mask = (abs_gp_weights_sums >= 
                              min_abs_gp_weights_sum_thresh)
        if abs_gp_weights_agg_mode in ["nzmeans", "sum+nzmeans"]:
            abs_gp_weights_nzmeans = (abs_gp_weights_sums / 
                                      torch.count_nonzero(gp_weights, dim=0))
            max_abs_gp_weights_nzmean = max(abs_gp_weights_nzmeans)
            min_abs_gp_weights_nzmean_thresh = (self.active_gp_thresh_ratio * 
                                                max_abs_gp_weights_nzmean)
            if abs_gp_weights_agg_mode == "nzmeans":
                active_gp_mask = (abs_gp_weights_nzmeans >= 
                                  min_abs_gp_weights_nzmean_thresh)
            elif abs_gp_weights_agg_mode == "sum+nzmeans":
                # Combine active gp mask
                active_gp_mask = active_gp_mask | (abs_gp_weights_nzmeans >= 
                                 min_abs_gp_weights_nzmean_thresh)
        if return_gp_weights:
            active_gp_weights = gp_weights[:, active_gp_mask]
            return active_gp_mask, active_gp_weights
        else:
            return active_gp_mask

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

    @torch.no_grad()
    def get_latent_representation(
            self,
            x: torch.Tensor,
            edge_index: torch.Tensor,
            use_only_active_gps=True,
            return_mu_std: bool=False
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode input features x and edge index into the latent space normal 
        distribution parameters and return z. If the module is not in training
        mode, mu will be returned.
           
        Parameters
        ----------
        x:
            Feature matrix to be encoded into latent space.
        edge_index:
            Edge index of the graph.
        use_only_active_gps:
            If `True`, only return the latent representation for active gene 
            programs.
        return_mu_std:
            If `True`, return mu and logstd instead of a random sample from the
            latent space.

        Returns
        -------
        z:
            Latent space encoding.
        mu:
            Expected values of the latent posterior.
        std:
            Standard deviations of the latent posterior.
        """
        mu, logstd = self.encoder(x, edge_index)

        if use_only_active_gps:
            active_gp_mask = self.get_active_gp_mask()
            mu, logstd = mu[:, active_gp_mask], logstd[:, active_gp_mask]

        if return_mu_std:
            std = torch.exp(logstd)
            return mu, std
        else:
            z = self.reparameterize(mu, logstd)
            return z

    @torch.no_grad()
    def get_gene_expr_decoder_params(
            self,
            z: torch.Tensor,
            log_library_size: torch.Tensor
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Decode latent features ´z´ to return the parameters of the distribution
        used for gene expression reconstruction (either (´nb_means´, 
        ´zi_prob_logits´) for a zero-inflated negative binomial or ´nb_means´ 
        for a negative binomial).

        Parameters
        ----------
        z:
            Tensor containing the latent features / gene program scores.
        log_library_size:
            Tensor containing the log library size of the observations / cells.

        Returns
        ----------
        gene_expr_decoder_params:
            Parameters of the gene expression decoder. Contains ´nb_means´ if
            ´self.gene_expr_recon_dist´ == ´nb´ and 
            (´nb_means´, ´zi_prob_logits´) if ´self.gene_expr_recon_dist´ == 
            ´zinb´.
        """
        gene_expr_decoder_params = self.gene_expr_decoder(
            z,
            log_library_size)
        return gene_expr_decoder_params