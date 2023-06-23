"""
This module contains the Variational Gene Program Graph Autoencoder class, the 
neural network module that underlies the NicheCompass model.
"""

from typing import List, Literal, Optional, Tuple, Union

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data

from nichecompass.nn import (CosineSimGraphDecoder,
                           Encoder,
                           MaskedGeneExprDecoder,
                           MaskedChromAccessDecoder,
                           OneHopAttentionNodeLabelAggregator,
                           OneHopGCNNormNodeLabelAggregator,
                           OneHopSumNodeLabelAggregator,
                           SelfNodeLabelNoneAggregator)
from .basemodulemixin import BaseModuleMixin
from .losses import (compute_addon_l1_reg_loss,
                     compute_cat_covariates_contrastive_loss,
                     compute_edge_recon_loss,
                     compute_omics_recon_nb_loss,
                     compute_gene_expr_recon_zinb_loss,
                     compute_group_lasso_reg_loss,
                     compute_kl_reg_loss,
                     compute_masked_l1_reg_loss)
from .vgaemodulemixin import VGAEModuleMixin


class VGPGAE(nn.Module, BaseModuleMixin, VGAEModuleMixin):
    """
    Variational Gene Program Graph Autoencoder class.

    Parameters
    ----------
    n_input:
        Number of nodes in the input layer.
    n_layers_encoder:
        Number of layers in the encoder.
    n_hidden_encoder:
        Number of nodes in the encoder hidden layer.
    n_nonaddon_gps:
        Number of nodes in the latent space (gene programs from the gene program
        mask).
    n_addon_gps:
        Number of add-on nodes in the latent space (de-novo gene programs).
    cat_covariates_embeds_nums:
        List of number of embedding nodes for all categorical covariates.
    n_output_genes:
        Number of nodes in the output layer.
    n_output_peaks:
        Number of output peaks for the masked chromatin accessibility decoder.
    n_genes_in_mask:
        Number of source and target genes that are included in the gp mask.
    gene_expr_decoder_mask:
        Gene program mask for the gene expression decoder.
    target_gene_expr_mask_idx:
        Index of target genes that are in gps of the gp mask.
    source_gene_expr_mask_idx:
        Index of source genes that are in gps of the gp mask.
    target_chrom_access_mask_idx:
        Index of target peaks that are in gps of the ca mask.
    source_chrom_access_mask_idx:
        Index of source peaks that are in gps of the ca mask.
    gene_peaks_mask:
        A mask to map from genes to peaks, used to turn off peaks in the
        chromatin accessibility decoder if the corresponding genes have been
        turned off by gene regularization.
    cat_covariates_cats:
        List of category lists for each categorical covariate for the
        categorical covariates embeddings.
    cat_covariates_no_edges:
        List of booleans that indicate whether there can be edges between
        different categories of the categorical covariates. If this is ´True´
        for a specific categorical covariate, this covariate will be excluded
        from the edge reconstruction loss.
    conv_layer_encoder:
        Convolutional layer used in the graph encoder.
    encoder_n_attention_heads:
        Only relevant if ´conv_layer_encoder == gatv2conv´. Number of attention
        heads used.
    dropout_rate_encoder:
        Probability that nodes will be dropped in the encoder during training.
    dropout_rate_graph_decoder:
        Probability that nodes will be dropped in the graph decoder during 
        training.
    include_edge_recon_loss:
        If `True`, includes the redge reconstruction loss in the loss 
        optimization.
    include_gene_expr_recon_loss:
        If `True`, includes the gene expression reconstruction loss in the 
        loss optimization.
    include_chrom_access_recon_loss:
        If `True`, includes the chromatin accessibility reconstruction loss in
        the loss optimization.    
    include_cat_covariates_contrastive_loss:
        If `True`, includes the categorical covariates contrastive loss in the
        backpropagation.
    gene_expr_recon_dist:
        The distribution used for gene expression reconstruction. If `nb`, uses
        a negative binomial distribution. If `zinb`, uses a zero-inflated
        negative binomial distribution.
    node_label_method:
        Node label method that will be used for gene expression reconstruction. 
        If ´self´, use only the input features of the node itself as node labels
        for gene expression reconstruction. If ´one-hop-sum´, use a 
        concatenation of the node's input features with the sum of the input 
        features of all nodes in the node's one-hop neighborhood. If 
        ´one-hop-norm´, use a concatenation of the node`s input features with
        the node's one-hop neighbors input features normalized as per Kipf, T. 
        N. & Welling, M. Semi-Supervised Classification with Graph Convolutional
        Networks. arXiv [cs.LG] (2016). If ´one-hop-attention´, use a 
        concatenation of the node`s input features with the node's one-hop 
        neighbors input features weighted by an attention mechanism.
    active_gp_thresh_ratio:
        Ratio that determines which gene programs are considered active and are
        used for edge reconstruction. All inactive gene programs will be dropped
        out. Aggregations of the absolute values of the gene weights of the 
        gene expression decoder per gene program are calculated. The maximum 
        value, i.e. the value of the gene program with the highest aggregated 
        value will be used as a benchmark and all gene programs whose aggregated
        value is smaller than ´active_gp_thresh_ratio´ times this maximum value 
        will be set to inactive. If ´==0´, all gene programs will be considered
        active. More information can be found in ´self.get_active_gp_mask()´.
    log_variational:
        If ´True´, transforms x by log(x+1) prior to encoding for numerical 
        stability (not normalization).
    cat_covariates_embeds_injection:
        List of VGPGAE modules in which the categorical covariates embeddings
        are injected.
    """
    def __init__(self,
                 n_input: int,
                 n_layers_encoder: int,
                 n_hidden_encoder: int,
                 n_nonaddon_gps: int,
                 n_addon_gps: int,
                 cat_covariates_embeds_nums: List[int],
                 n_output_genes: int,
                 gene_expr_decoder_mask: torch.Tensor,
                 gene_expr_mask_idx: torch.Tensor,
                 target_gene_expr_mask_idx: torch.Tensor,
                 source_gene_expr_mask_idx: torch.Tensor,
                 n_output_peaks: int=0,
                 chrom_access_decoder_mask: Optional[torch.Tensor]=None,
                 chrom_access_mask_idx: Optional[torch.Tensor]=None,
                 target_chrom_access_mask_idx: Optional[torch.Tensor]=None,
                 source_chrom_access_mask_idx: Optional[torch.Tensor]=None,
                 gene_peaks_mask: Optional[torch.Tensor]=None,
                 cat_covariates_cats: List[List]=[],
                 cat_covariates_no_edges: List[bool]=[],
                 conv_layer_encoder: Literal["gcnconv", "gatv2conv"]="gcnconv",
                 encoder_n_attention_heads: int=4,
                 dropout_rate_encoder: float=0.,
                 dropout_rate_graph_decoder: float=0.,
                 include_edge_recon_loss: bool=True,
                 include_gene_expr_recon_loss: bool=True,
                 include_chrom_access_recon_loss: bool=True,
                 include_cat_covariates_contrastive_loss: bool=True,
                 gene_expr_recon_dist: Literal["nb", "zinb"]="nb",
                 chrom_access_recon_dist: Literal["nb"]="nb",
                 node_label_method: Literal[
                    "self",
                    "one-hop-norm",
                    "one-hop-sum",
                    "one-hop-attention"]="one-hop-attention",
                 active_gp_thresh_ratio: float=0.03,
                 log_variational: bool=True,
                 cat_covariates_embeds_injection: Optional[List[
                     Literal["encoder",
                             "gene_expr_decoder",
                             "chrom_access_decoder"]]]=["gene_expr_decoder",
                                                        "chrom_access_decoder"]):
        super().__init__()
        self.n_input_ = n_input
        self.n_layers_encoder_ = n_layers_encoder
        self.n_hidden_encoder_ = n_hidden_encoder
        self.n_nonaddon_gps_ = n_nonaddon_gps
        self.n_addon_gps_ = n_addon_gps
        self.cat_covariates_embeds_nums_ = cat_covariates_embeds_nums
        self.n_output_genes_ = n_output_genes
        self.n_output_peaks_ = n_output_peaks
        self.gene_expr_mask_idx_ = gene_expr_mask_idx
        self.target_gene_expr_mask_idx_ = target_gene_expr_mask_idx
        self.source_gene_expr_mask_idx_ = source_gene_expr_mask_idx
        self.chrom_access_mask_idx_ = chrom_access_mask_idx
        self.target_chrom_access_mask_idx_ = target_chrom_access_mask_idx
        self.source_chrom_access_mask_idx_ = source_chrom_access_mask_idx
        self.gene_peaks_mask_ = gene_peaks_mask
        self.cat_covariates_cats_ = cat_covariates_cats
        self.n_cat_covariates___ = len(cat_covariates_cats)
        self.cat_covariates_no_edges_ = cat_covariates_no_edges
        self.nums_cat_covariates_cats_ = [
            len(cat_covariate_cats) for cat_covariate_cats in cat_covariates_cats]
        self.cat_covariates_label_encoders_ = [
            {k: v for k, v in zip(cat_covariate_cats,
                                  range(len(cat_covariate_cats)))}
                                  for cat_covariate_cats in cat_covariates_cats]
        self.conv_layer_encoder_ = conv_layer_encoder
        self.encoder_n_attention_heads_ = encoder_n_attention_heads
        self.dropout_rate_encoder_ = dropout_rate_encoder
        self.dropout_rate_graph_decoder_ = dropout_rate_graph_decoder
        self.include_edge_recon_loss_ = include_edge_recon_loss
        self.include_gene_expr_recon_loss_ = include_gene_expr_recon_loss
        self.include_chrom_access_recon_loss_ = include_chrom_access_recon_loss
        self.include_cat_covariates_contrastive_loss_ = include_cat_covariates_contrastive_loss
        self.gene_expr_recon_dist_ = gene_expr_recon_dist
        self.chrom_access_recon_dist_ = chrom_access_recon_dist
        self.node_label_method_ = node_label_method
        self.active_gp_thresh_ratio_ = active_gp_thresh_ratio
        self.log_variational_ = log_variational
        self.cat_covariates_embeds_injection_ = cat_covariates_embeds_injection
        self.freeze_ = False
        self.modalities_ = ["gene_expr"]
        if chrom_access_decoder_mask is not None:
            self.modalities_.append("chrom_access")
            self.features_idx_ = np.concatenate(
                (target_gene_expr_mask_idx,
                 (target_chrom_access_mask_idx + int(n_output_genes / 2)),
                 (source_gene_expr_mask_idx + int(n_output_genes / 2) + int(n_output_peaks / 2)),
                 (source_chrom_access_mask_idx + n_output_genes + int(n_output_peaks / 2))),
                 axis=0)

        else:
            self.features_idx_ = self.gene_expr_mask_idx_

        print("--- INITIALIZING NEW NETWORK MODULE: VARIATIONAL GENE PROGRAM "
              "GRAPH AUTOENCODER ---")
        print(f"LOSS -> include_edge_recon_loss: {include_edge_recon_loss}, "
              f"include_gene_expr_recon_loss: {include_gene_expr_recon_loss}, "
              f"gene_expr_recon_dist: {gene_expr_recon_dist}", end="")
        if "chrom_access" in self.modalities_:
            print(", include_chrom_access_recon_loss: "
                  f"{include_chrom_access_recon_loss}, "
                  "chrom_access_recon_dist: "
                  f"{chrom_access_recon_dist}", end=" ")
        print(f"\nNODE LABEL METHOD -> {node_label_method}")
        print(f"ACTIVE GP THRESHOLD RATIO -> {active_gp_thresh_ratio}")
        print(f"LOG VARIATIONAL -> {log_variational}")
        if len(cat_covariates_cats) != 0:
            print("CATEGORICAL COVARIATES EMBEDDINGS INJECTION -> "
                  f"{cat_covariates_embeds_injection}")
            
        # Initialize categorical covariates embedder modules
        if len(self.cat_covariates_cats_) > 0:
            self.cat_covariates_embedders = []
            for i in range(len(self.cat_covariates_cats_)):
                cat_covariate_embedder = nn.Embedding(
                    self.nums_cat_covariates_cats_[i],
                    cat_covariates_embeds_nums[i])
                self.cat_covariates_embedders.append(cat_covariate_embedder)
                # Set attribute so PyTorch recognizes the layer and moves it
                # to GPU
                setattr(self,
                        f"cat_covariate{i}_embedder",
                        cat_covariate_embedder)

        # Initialize encoder module
        self.encoder = Encoder(
            n_input=n_input,
            n_cat_covariates_embed_input=(sum(cat_covariates_embeds_nums)
                                          if ("encoder" in self.cat_covariates_embeds_injection_)
                                          & (self.n_cat_covariates___ > 0)
                                          else 0),
            n_cat_covariates_embed_input=0,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden_encoder,
            n_latent=n_nonaddon_gps,
            n_addon_latent=n_addon_gps,
            conv_layer=conv_layer_encoder,
            n_attention_heads=encoder_n_attention_heads,
            dropout_rate=dropout_rate_encoder,
            activation=torch.relu)
        
        # Initialize graph decoder module
        self.graph_decoder = CosineSimGraphDecoder(
            dropout_rate=dropout_rate_graph_decoder)

        # Initialize masked gene expression decoder
        self.gene_expr_decoder = MaskedGeneExprDecoder(
            n_input=n_nonaddon_gps,
            n_addon_input=n_addon_gps,
            n_cat_covariates_embed_input=(sum(cat_covariates_embeds_nums)
                                          if ("gene_expr_decoder" in self.cat_covariates_embeds_injection_)
                                          & (self.n_cat_covariates___ > 0)
                                          else 0),
            n_output=n_output_genes,
            mask=gene_expr_decoder_mask,
            mask_idx=gene_expr_mask_idx,
            recon_dist=self.gene_expr_recon_dist_)
        
        if "chrom_access" in self.modalities_:
            # Initialize masked chromatin accessibility decoder
            self.chrom_access_decoder = MaskedChromAccessDecoder(
                n_input=n_nonaddon_gps,
                n_addon_input=n_addon_gps,
                n_cat_covariates_embed_input=(sum(cat_covariates_embeds_nums)
                                              if cat_covariates_embeds_nums is not None
                                              else 0),
                n_output=n_output_peaks,
                mask=chrom_access_decoder_mask,
                mask_idx=chrom_access_mask_idx,
                recon_dist=self.chrom_access_recon_dist_)            

        if node_label_method == "self":
            self.node_label_aggregator = SelfNodeLabelNoneAggregator(
                features_idx=self.features_idx_)
        if node_label_method == "one-hop-norm":
            self.node_label_aggregator = OneHopGCNNormNodeLabelAggregator(
                features_idx=self.features_idx_)
        elif node_label_method == "one-hop-sum":
            self.node_label_aggregator = OneHopSumNodeLabelAggregator(
                features_idx=self.features_idx_)
        elif node_label_method == "one-hop-attention":
            self.node_label_aggregator = OneHopAttentionNodeLabelAggregator(
                n_input=n_input,
                features_idx=self.features_idx_)
        
        # Initialize gene-specific dispersion parameters
        self.theta = torch.nn.Parameter(torch.randn(len(gene_expr_mask_idx)))

        if "chrom_access" in self.modalities_:
            # Initialize peak-specific dispersion parameters
            self.theta_atac = torch.nn.Parameter(torch.randn(len(chrom_access_mask_idx)))

    def forward(self,
                data_batch: Data,
                decoder: Literal["graph", "omics"],
                use_only_active_gps: bool=False,
                return_agg_weights: bool=False,
                turn_off_peaks_based_on_genes: bool=True) -> dict:
        """
        Forward pass of the VGPGAE module.

        Parameters
        ----------
        data_batch:
            PyG Data object containing either an edge-level batch if 
            ´decoder == graph´ or a node-level batch if ´decoder == gene_expr´.
        decoder:
            Decoder to use for the forward pass, either ´graph´ for edge
            reconstruction or ´omics´ for gene expression and (if specified)
            chromatin accessibility reconstruction.
        use_only_active_gps:
            If ´True´, use only active gene programs as input to decoder.
        return_agg_weights:
            If ´True´, also return the aggregation weights of the node label
            aggregator.
        turn_off_peaks_based_on_genes:
            If ´True´, turn off the mapped peaks (peak gp weights) for genes
            that have been turned off in a gene program by L1 regularization.

        Returns
        ----------
        output:
            Dictionary containing reconstructed adjacency matrix logits if
            ´decoder == graph´ or the parameters of the gene expression 
            distribution if ´decoder == gene_expr´, as well as ´mu´ and ´logstd´ 
            from the latent space distribution.
        """
        x = data_batch.x # dim: n_obs x n_omics_features
        edge_index = data_batch.edge_index # dim: 2 x n_edges
        output = {}

        # Logarithmitize omics feature vector for numerical stability
        if self.log_variational_:
            x_enc = torch.log(1 + x)
        else:
            x_enc = x

        # Get index of sampled nodes for current batch as well as node or edge
        # labels depending on decoder
        if decoder == "omics":
            # ´data_batch´ will be a node batch and first node_batch_size
            # elements are the sampled nodes, leading to a dim of ´batch_idx´ of
            # node_batch_size
            batch_idx = torch.tensor(range(data_batch.batch_size))

            # Compute aggregated neighborhood omics feature vector to create
            # concatenated omics reconstruction labels
            node_label_aggregator_output = self.node_label_aggregator(
                    x=x,
                    edge_index=edge_index,
                    return_agg_weights=return_agg_weights)
            output["node_labels"] = node_label_aggregator_output[0][batch_idx]
        elif decoder == "graph":
            # ´data_batch´ will be an edge batch with sampled positive and
            # negative edges of size edge_batch_size respectively. Each edge has
            # a source and destination node, leading to a dim of ´batch_idx´ of
            # 4 * edge_batch_size
            batch_idx = torch.cat((data_batch.edge_label_index[0],
                                   data_batch.edge_label_index[1]), 0)
            
            # Store edge labels for loss computation
            output["edge_recon_labels"] = data_batch.edge_label
                
            # Store edge categorical covariates label for loss computation
            if len(self.cat_covariates_cats_) > 0:
                output["edge_same_cat_covariates_cat"] = []
                for cat_covariate_idx in range(len(self.cat_covariates_cats_)):
                    edge_same_cat_covariate_cat = (
                        data_batch.cat_covariates_cats[
                            data_batch.edge_label_index[0],
                            cat_covariate_idx] ==
                        data_batch.cat_covariates_cats[
                            data_batch.edge_label_index[1],
                            cat_covariate_idx])
                    output["edge_same_cat_covariates_cat"].append(
                        edge_same_cat_covariate_cat)
            else:
                output["edge_same_cat_covariates_cat"] = None

        # Get categorical covariate embeddings
        if len(self.cat_covariates_cats_) > 0:
            cat_covariates_embeds = []
            for i in range(len(self.cat_covariates_embedders)):
                cat_covariates_embeds.append(self.cat_covariates_embedders[i](
                     data_batch.cat_covariates_cats[:, i]))
                self.cat_covariates_embed = torch.cat(
                    cat_covariates_embeds,
                    dim=1)
        else:
            self.cat_covariates_embed = None         

        # Use encoder and reparameterization trick to get latent distribution
        # parameters and features for current batch
        encoder_outputs = self.encoder(
            x=x_enc,
            edge_index=edge_index,
            cat_covariates_embed=(self.cat_covariates_embed if "encoder" in
                                  self.cat_covariates_embeds_injection_ else
                                  None))
        self.mu = encoder_outputs[0][batch_idx, :]
        self.logstd = encoder_outputs[1][batch_idx, :]
        output["mu"] = self.mu
        output["logstd"] = self.logstd
        z = self.reparameterize(self.mu, self.logstd)

        if use_only_active_gps:
            # Only retain active gene programs
            active_gp_mask = self.get_active_gp_mask()
            z[:, ~active_gp_mask] = 0

        # Use decoder to get either the edge reconstruction logits or the omics
        # distribution parameters from the latent feature vectors
        if decoder == "graph":
            output["edge_recon_logits"] = self.graph_decoder(
                z=z)
        elif decoder == "omics":
            if "chrom_access" in self.modalities_:
                # Separate node feature vector into RNA and ATAC part
                x_atac = x[:, int(self.n_output_genes_ / 2):]
                x = x[:, :int(self.n_output_genes_ / 2)]
                assert x_atac.size(1) == int(self.n_output_peaks_ / 2)
                target_node_labels_atac_start_idx = len(self.target_gene_expr_mask_idx_)
                source_node_labels_atac_start_idx = (
                    len(self.target_gene_expr_mask_idx_) + 
                    len(self.target_chrom_access_mask_idx_) +
                    len(self.source_gene_expr_mask_idx_))
                source_node_labels_rna_start_idx = (
                    len(self.target_gene_expr_mask_idx_) + 
                    len(self.target_chrom_access_mask_idx_))

                output["node_labels_atac"] = torch.cat((
                    output["node_labels"][:, target_node_labels_atac_start_idx:
                                          source_node_labels_rna_start_idx],
                    output["node_labels"][:, source_node_labels_atac_start_idx:
                                          ]), dim=1)
                output["node_labels"] = torch.cat((
                    output["node_labels"][:, :target_node_labels_atac_start_idx],
                    output["node_labels"][:, source_node_labels_rna_start_idx: 
                                          source_node_labels_atac_start_idx]),
                                          dim=1)
                assert output["node_labels_atac"].size(1) == len(self.chrom_access_mask_idx_)
                assert output["node_labels"].size(1) == len(self.gene_expr_mask_idx_)

            # Use observed library size as scaling factor for the negative
            # binomial means of the gene expression distribution
            self.log_library_size = torch.log(x.sum(1)).unsqueeze(1)[batch_idx]

            # Get gene expression reconstruction distribution parameters
            output["gene_expr_dist_params"] = self.gene_expr_decoder(
                z=z,
                log_library_size=self.log_library_size,
                cat_covariates_embed=(self.cat_covariates_embed[batch_idx] if
                                      "gene_expr_decoder" in
                                      self.cat_covariates_embeds_injection_ else
                                      None))
            
            if "chrom_access" in self.modalities_:
                # Use observed library size as scaling factor for the negative
                # binomial means of the chromatin accessibility distribution            
                self.log_library_size_atac = torch.log(
                    x_atac.sum(1)).unsqueeze(1)[batch_idx]
                
                if turn_off_peaks_based_on_genes:
                    # Get dynamic gene weight peak mask to turn off peaks that
                    # correspond to genes that are turned off
                    with torch.no_grad():
                        # Round to 4 decimals as genes are never completely
                        # turned off due to L1 being not differentiable at 0
                        gp_weights = self.get_gp_weights(use_mask_idx=False)[0]
                        gp_weights = torch.round(gp_weights, decimals=4)

                        non_zero_gene_weights = torch.ne(
                                gp_weights, 
                                0).float() # dim: (2 x n_genes, n_gps)

                        non_zero_target_gene_weights = non_zero_gene_weights[
                            :int(non_zero_gene_weights.size(0) / 2), :]
                            # dim: (n_genes, n_gps)
                        non_zero_source_gene_weights = non_zero_gene_weights[
                            int(non_zero_gene_weights.size(0) / 2):, :]
                            # dim: (n_genes, n_gps)

                        gene_weight_target_peak_mask = torch.matmul(
                            non_zero_target_gene_weights.t(), # dim: (n_gps, n_genes)
                            self.gene_peaks_mask_) # dim: (n_genes, n_peaks)
                            # dim: (n_gps, n_peaks)
                        gene_weight_target_peak_mask = torch.ne(
                            gene_weight_target_peak_mask, 
                            0).float() # dim: (n_gps, n_peaks)

                        gene_weight_source_peak_mask = torch.matmul(
                            non_zero_source_gene_weights.t(),
                            self.gene_peaks_mask_)
                        gene_weight_source_peak_mask = torch.ne(
                            gene_weight_source_peak_mask, 
                            0).float()

                        gene_weight_peak_mask = torch.cat(
                            (gene_weight_target_peak_mask,
                             gene_weight_source_peak_mask), dim=1).t()
                            # dim: (2 x n_peaks, n_gps)
                else:
                    gene_weight_peak_mask = None

                # Get chromatin accessibility reconstruction distribution
                # parameters
                output["chrom_access_dist_params"] = self.chrom_access_decoder(
                    z=z,
                    log_library_size=self.log_library_size_atac,
                    gene_weight_peak_mask=gene_weight_peak_mask,
                    cat_covariates_embed=(self.cat_covariates_embed[batch_idx]
                                          if "chrom_access_decoder" in
                                          self.cat_covariates_embeds_injection_
                                          else None))

        return output

    def loss(self,
             edge_model_output: dict,
             node_model_output: dict,
             lambda_l1_masked: float,
             l1_mask: np.array,
             lambda_l1_addon: float,
             lambda_group_lasso: float,
             lambda_gene_expr_recon: float=300.,
             lambda_chrom_access_recon: float=100.,
             lambda_edge_recon: Optional[float]=500000.,
             lambda_cat_covariates_contrastive: Optional[float]=100000.,
             contrastive_logits_pos_ratio: float=0.125,
             contrastive_logits_neg_ratio: float=0.,
             edge_recon_active: bool=True,
             cat_covariates_contrastive_active: bool=True) -> dict:
        """
        Calculate the optimization loss for backpropagation as well as the 
        global loss that also contains components omitted from optimization 
        (not backpropagated) and is used for early stopping evaluation.

        Parameters
        ----------
        edge_model_output:
            Output of the edge-level forward pass for edge reconstruction.
        node_model_output:
            Output of the node-level forward pass for gene expression 
            reconstruction.
        lambda_edge_recon:
            Lambda (weighting factor) for the edge reconstruction loss. If ´>0´,
            this will enforce gene programs to be meaningful for edge
            reconstruction and, hence, to preserve spatial colocalization
            information.
        lambda_gene_expr_recon:
            Lambda (weighting factor) for the gene expression reconstruction
            loss. If ´>0´, this will enforce interpretable gene programs that
            can be combined in a linear way to reconstruct gene expression.
        lambda_chrom_access_recon:
            Lambda (weighting factor) for the chromatin accessibility
            reconstruction loss. If ´>0´, this will enforce interpretable gene
            programs that can be combined in a linear way to reconstruct
            chromatin accessibility.
        lambda_group_lasso:
            Lambda (weighting factor) for the group lasso regularization loss of
            gene programs. If ´>0´, this will enforce sparsity of gene programs.
        lambda_l1_masked:
            Lambda (weighting factor) for the L1 regularization loss of genes in
            masked gene programs. If ´>0´, this will enforce sparsity of genes
            in masked gene programs.
        l1_mask:
            Boolean gene program gene mask that is True for all gene program genes
            to which the L1 regularization loss should be applied (dim: 2 x n_genes,
            n_gps)
        lambda_l1_addon:
            Lambda (weighting factor) for the L1 regularization loss of genes in
            addon gene programs. If ´>0´, this will enforce sparsity of genes in
            addon gene programs.
        lambda_cat_covariates_contrastive:
            Lambda (weighting factor) for the categorical covariates contrastive
            loss. If ´>0´, this will enforce observations with different
            categorical covariates categories with very similar latent
            representations to become more similar, and observations with
            different latent representations to become more different.
        contrastive_logits_pos_ratio:
            Ratio for determining the logits threshold of positive contrastive
            examples of node pairs from different categorical covariates
            categories. The top (´contrastive_logits_pos_ratio´ * 100)% logits
            of node pairs from different categorical covariates categories serve
            as positive labels for the contrastive loss.
        contrastive_logits_neg_ratio:
            Ratio for determining the logits threshold of negative contrastive
            examples of node pairs from different categorical covariates
            categories. The bottom (´contrastive_logits_neg_ratio´ * 100)%
            logits of node pairs from different categorical covariates
            categories serve as negative labels for the contrastive loss.
        edge_recon_active:
            If ´True´, includes the edge reconstruction loss in the optimization
            / backpropagation. Setting this to ´False´ at the beginning of model
            training allows pretraining using other loss components.
        cat_covariates_contrastive_active:
            If ´True´, includes the categorical covariates contrastive loss in
            the optimization / backpropagation. Setting this to ´False´ at the
            beginning of model training allows pretraining using other loss
            components.        

        Returns
        ----------
        loss_dict:
            Dictionary containing the loss used for backpropagation 
            (loss_dict["optim_loss"]), which consists of all loss components 
            used for optimization, the global loss (loss_dict["global_loss"]), 
            which contains all loss components irrespective of whether they are
            used for optimization (needed as metric for early stopping and best
            model saving), as well as all individual loss components that 
            contribute to the global loss.
        """
        loss_dict = {}

        # Compute Kullback-Leibler divergence loss for edge and node batch
        loss_dict["kl_reg_loss"] = compute_kl_reg_loss(
            mu=edge_model_output["mu"],
            logstd=edge_model_output["logstd"])
        loss_dict["kl_reg_loss"] += compute_kl_reg_loss(
            mu=node_model_output["mu"],
            logstd=node_model_output["logstd"])

        # Determine edges to be included in edge reconstruction loss
        print(self.cat_covariates_no_edges_)
        print(edge_model_output["edge_same_cat_covariates_cat"])

        cat_covariates_cat_edge_incl = []
        for cat_covariate_no_edge, edge_same_cat_covariate_cat in zip(
            self.cat_covariates_no_edges_,
            edge_model_output["edge_same_cat_covariates_cat"]):
            if not cat_covariate_no_edge:
                cat_covariates_cat_edge_incl.append(
                    torch.ones_like(edge_same_cat_covariate_cat,
                                    dtype=torch.bool))
            else:
                cat_covariates_cat_edge_incl.append(edge_same_cat_covariate_cat)
        if len(cat_covariates_cat_edge_incl) > 0:
            edge_incl = torch.all(
                torch.stack(cat_covariates_cat_edge_incl),
                            dim=0)
        else:
            edge_incl = None

        # Compute edge reconstruction binary cross entropy loss
        loss_dict["edge_recon_loss"] = (
            lambda_edge_recon * compute_edge_recon_loss(
                edge_recon_logits=edge_model_output["edge_recon_logits"],
                edge_recon_labels=edge_model_output["edge_recon_labels"],
                edge_incl=edge_incl))
            
        # Compute categorical covariates contrastive loss
        if (edge_model_output["edge_same_cat_covariates_cat"] is not None) & (
        lambda_cat_covariates_contrastive > 0):
            loss_dict["cat_covariates_contrastive_loss"] = (
                lambda_cat_covariates_contrastive * compute_cat_covariates_contrastive_loss(
                edge_recon_logits=edge_model_output["edge_recon_logits"],
                edge_recon_labels=edge_model_output["edge_recon_labels"],
                edge_same_cat_covariates_cat=edge_model_output[
                    "edge_same_cat_covariates_cat"],
                contrastive_logits_pos_ratio=contrastive_logits_pos_ratio,
                contrastive_logits_neg_ratio=contrastive_logits_neg_ratio))

        # Compute gene expression reconstruction negative binomial or
        # zero-inflated negative binomial loss
        theta = torch.exp(self.theta) # gene-specific inverse dispersion
        if self.gene_expr_recon_dist_ == "nb":
            nb_means = node_model_output["gene_expr_dist_params"]
            loss_dict["gene_expr_recon_loss"] = (lambda_gene_expr_recon * 
            compute_omics_recon_nb_loss(
                    x=node_model_output["node_labels"],
                    mu=nb_means,
                    theta=theta))
        elif self.gene_expr_recon_dist_ == "zinb":
            nb_means, zi_prob_logits = (
                node_model_output["gene_expr_dist_params"])
            loss_dict["gene_expr_recon_loss"] = (lambda_gene_expr_recon * 
            compute_gene_expr_recon_zinb_loss(
                    x=node_model_output["node_labels"],
                    mu=nb_means,
                    theta=theta,
                    zi_prob_logits=zi_prob_logits))
            
        loss_dict["masked_gp_l1_reg_loss"] = (lambda_l1_masked *
            compute_masked_l1_reg_loss(self,
                                       l1_mask=l1_mask))

        # Compute group lasso regularization loss of gene programs
        loss_dict["group_lasso_reg_loss"] = (lambda_group_lasso *
        compute_group_lasso_reg_loss(self))

        # Compute l1 regularization loss of genes in addon gene programs
        if self.n_addon_gps_ != 0:
            loss_dict["addon_gp_l1_reg_loss"] = (lambda_l1_addon *
            compute_addon_l1_reg_loss(self))

        if "chrom_access" in self.modalities_:
            # Compute chromatin accessibility reconstruction negative binomial
            # loss
            theta_atac = torch.exp(self.theta_atac) # peak-specific inverse
                                                    # dispersion
            nb_means_atac = node_model_output["chrom_access_dist_params"]
            loss_dict["chrom_access_recon_loss"] = (lambda_chrom_access_recon * 
            compute_omics_recon_nb_loss(
                    x=node_model_output["node_labels_atac"],
                    mu=nb_means_atac,
                    theta=theta_atac))

        # Compute optimization loss used for backpropagation as well as global
        # loss used for early stopping of model training and best model saving
        loss_dict["global_loss"] = 0
        loss_dict["optim_loss"] = 0
        loss_dict["global_loss"] += loss_dict["kl_reg_loss"]
        loss_dict["optim_loss"] += loss_dict["kl_reg_loss"]
        if self.include_edge_recon_loss_:
            loss_dict["global_loss"] += loss_dict["edge_recon_loss"]
            if edge_recon_active:
                loss_dict["optim_loss"] += loss_dict["edge_recon_loss"]
        if self.include_cat_covariates_contrastive_loss_ & (
        "cat_covariates_contrastive_loss" in loss_dict.keys()):
            loss_dict["global_loss"] += loss_dict["cat_covariates_contrastive_loss"]
            if cat_covariates_contrastive_active:
                loss_dict["optim_loss"] += loss_dict["cat_covariates_contrastive_loss"] 
        if self.include_gene_expr_recon_loss_:
            loss_dict["global_loss"] += loss_dict["gene_expr_recon_loss"]
            loss_dict["optim_loss"] += loss_dict["gene_expr_recon_loss"]
            loss_dict["global_loss"] += loss_dict["group_lasso_reg_loss"]
            loss_dict["optim_loss"] += loss_dict["group_lasso_reg_loss"]
            loss_dict["global_loss"] += loss_dict["masked_gp_l1_reg_loss"]
            loss_dict["optim_loss"] += loss_dict["masked_gp_l1_reg_loss"]
            if self.n_addon_gps_ != 0:
                loss_dict["global_loss"] += loss_dict["addon_gp_l1_reg_loss"]
                loss_dict["optim_loss"] += loss_dict["addon_gp_l1_reg_loss"]
        if ("chrom_access" in self.modalities_) & self.include_chrom_access_recon_loss_:
            loss_dict["global_loss"] += loss_dict["chrom_access_recon_loss"]
            loss_dict["optim_loss"] += loss_dict["chrom_access_recon_loss"]
        return loss_dict

    def get_gp_weights(self,
                       use_mask_idx: bool=False) -> torch.Tensor:
        """
        Get the gene weights of the gene expression negative binomial means
        decoder.

        Returns:
        ----------
        gp_weights_all_modalities:
            Tuple of tensors containing the decoder gp weights (dim:
            n_gps x n_genes)
        gp_peak_weights:
            Tensor containing the chromatin accessibility decoder peak weights (
            dim: n_gps x n_peaks)
        """
        gp_weights_all_modalities = []

        for modality in self.modalities_:
            decoder = getattr(self, modality + "_decoder")
            
            # Get decoder weights of masked gps
            gp_weights = (
                decoder.nb_means_normalized_decoder.masked_l.weight.data
                ).clone()

            # Add decoder weights of addon gps
            if self.n_addon_gps_ > 0:
                gp_weights_addon = (
                    decoder.nb_means_normalized_decoder.addon_l.weight.data
                    ).clone()
                gp_weights = torch.cat([gp_weights, gp_weights_addon], axis=1)

            # Only keep omics features in mask
            if use_mask_idx:
                mask_idx = getattr(self, modality + "_mask_idx_")
                gp_weights = gp_weights[mask_idx, :]
            
            # append masked_decoder_weights to the list of weights
            gp_weights_all_modalities.append(gp_weights)

        # return the weights as a tuple
        return gp_weights_all_modalities

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
        ´self.active_gp_thresh_ratio_´ times the absolute gene weights
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
        gp_weights = self.get_gp_weights(use_mask_idx=True)[0]

        # Correct gp weights for zero inflation using zero inflation
        # probabilities over all observations if zinb distribution is used to
        # model gene expression
        if self.gene_expr_recon_dist_ == "zinb":
            _, zi_probs = self.get_gene_expr_dist_params(
                z=self.mu,
                log_library_size=self.log_library_size,
                cat_covariates_embed=self.cat_covariates_embed[batch_idx])
            non_zi_probs = 1 - zi_probs
            non_zi_probs_sum = non_zi_probs.sum(0).unsqueeze(1) # sum over obs
            gp_weights *= non_zi_probs_sum

        # Aggregate absolute gp weights based on ´abs_gp_weights_agg_mode´ and
        # calculate thresholds of aggregated absolute gp weights and get active
        # gp mask and (optionally) active gp weights
        abs_gp_weights_sums = gp_weights.norm(p=1, dim=0)
        if abs_gp_weights_agg_mode in ["sum", "sum+nzmeans"]:
            max_abs_gp_weights_sum = max(abs_gp_weights_sums)
            min_abs_gp_weights_sum_thresh = (self.active_gp_thresh_ratio_ * 
                                             max_abs_gp_weights_sum)
            active_gp_mask = (abs_gp_weights_sums >= 
                              min_abs_gp_weights_sum_thresh)
        if abs_gp_weights_agg_mode in ["nzmeans", "sum+nzmeans"]:
            abs_gp_weights_nzmeans = (abs_gp_weights_sums / 
                                      torch.count_nonzero(gp_weights, dim=0))
            max_abs_gp_weights_nzmean = max(abs_gp_weights_nzmeans)
            min_abs_gp_weights_nzmean_thresh = (self.active_gp_thresh_ratio_ *
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

    def log_module_hyperparams_to_mlflow(
            self,
            excluded_attr: list=["gene_expr_mask_idx_",
                                 "target_gene_expr_mask_idx_",
                                 "source_gene_expr_mask_idx_",
                                 "chrom_access_mask_idx_",
                                 "target_chrom_access_mask_idx_",
                                 "source_chrom_access_mask_idx_",
                                 "features_idx_",
                                 "gene_peaks_mask_"]):
        """
        Log module hyperparameters to Mlflow.
        
        Parameters
        ----------
        excluded_attr:
            Attributes that are excluded despite being public because of length
            restrictions of mlflow.
        """
        for attr, attr_value in self._get_public_attributes().items():
            if attr not in excluded_attr:
                if attr == "cat_covariates_cats_":
                    for i in range(len(attr_value)):
                        mlflow.log_param(f"cat_covariate{i}_cats",
                                         attr_value[0])
                elif attr == "cat_covariates_label_encoders_":
                    for i in range(len(attr_value)):
                        mlflow.log_param(f"cat_covariate{i}_label_encoder",
                                         attr_value[0])
                else:                   
                    mlflow.log_param(attr, attr_value)

    def get_latent_representation(
            self,
            node_batch: Data,
            only_active_gps: bool=True,
            return_mu_std: bool=False
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode input features ´x´ and ´edge_index´ into the latent distribution
        parameters and return either the distribution parameters themselves or
        latent features ´z´.
           
        Parameters
        ----------
        node_batch:
            PyG Data object containing a node-level batch.
        only_active_gps:
            If ´True´, return only the latent representation of active gps.
        return_mu_std:
            If ´True´, return ´mu´ and ´std´ instead of latent features ´z´.

        Returns
        -------
        z:
            Latent space features (dim: n_obs, n_active_gps).
        mu:
            Expected values of the latent posterior (dim: n_obs, n_active_gps).
        std:
            Standard deviations of the latent posterior (dim: n_obs, 
            n_active_gps).
        """
        # Convert gene expression if done during training
        if self.log_variational_:
            x_enc = torch.log(1 + node_batch.x)
        else:
            x_enc = node_batch.x # dim: n_obs x n_omics_features
            
        # Get categorical covariate embeddings
        if len(self.cat_covariates_cats_) > 0:
            cat_covariates_embeds = []
            for i in range(len(self.cat_covariates_embedders)):
                cat_covariates_embeds.append(self.cat_covariates_embedders[i](
                    node_batch.cat_covariates_cats[:, i]))
                cat_covariates_embed = torch.cat(
                    cat_covariates_embeds,
                    dim=1)
        else:
            cat_covariates_embed = None
            
        # Get latent distribution parameters
        encoder_outputs = self.encoder(
            x=x_enc,
            edge_index=node_batch.edge_index, # dim: 2 x n_edges
            cat_covariates_embed=(cat_covariates_embed if "encoder" in
                                  self.cat_covariates_embeds_injection_ else
                                  None))
        mu = encoder_outputs[0][:node_batch.batch_size, :]
        logstd = encoder_outputs[1][:node_batch.batch_size, :]

        if only_active_gps:
            # Filter to active gene programs only
            active_gp_mask = self.get_active_gp_mask()
            mu, logstd = mu[:, active_gp_mask], logstd[:, active_gp_mask]

        if return_mu_std:
            # (?) check whether this is redundant
            std = torch.exp(logstd)
            return mu, std
        else:
            # Sample latent features from the latent normal distribution if in 
            # training mode or return ´mu´ if not in training mode
            z = self.reparameterize(mu, logstd)
            return z

    def get_gene_expr_dist_params(
            self,
            z: torch.Tensor,
            log_library_size: torch.Tensor,
            cat_covariates_embed: Optional[torch.Tensor]=None,
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Decode latent features ´z´ to return the parameters of the distribution
        used for gene expression reconstruction (either (´nb_means´, ´zi_probs´)
        if a zero-inflated negative binomial is used or ´nb_means´ if a negative
        binomial is used).

        Parameters
        ----------
        z:
            Tensor containing the latent features / gene program scores (dim: 
            n_obs x n_gps).
        log_library_size:
            Tensor containing the log library size of the observations / cells 
            (dim: n_obs x 1).
        cat_covariates_embed:
            Tensor containing the categorical covariates embedding
            (dim: n_obs x sum(cat_covariates_embeds_num)).

        Returns
        ----------
        nb_means:
            Expected values of the negative binomial distribution (dim: n_obs x
            n_genes).
        zi_probs:
            Zero-inflation probabilities of the zero-inflated negative binomial
            distribution (dim: n_obs x n_genes).
        """
        if self.gene_expr_recon_dist_ == "nb":
            nb_means = self.gene_expr_decoder(
                z=z,
                log_library_size=log_library_size,
                cat_covariates_embed=cat_covariates_embed)
            return nb_means
        if self.gene_expr_recon_dist_ == "zinb":
            nb_means, zi_prob_logits = self.gene_expr_decoder(
                z=z,
                log_library_size=log_library_size,
                cat_covariates_embed=cat_covariates_embed)
            zi_probs = torch.sigmoid(zi_prob_logits)
            return nb_means, zi_probs