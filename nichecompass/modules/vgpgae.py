"""
This module contains the Variational Gene Program Graph Autoencoder class, the 
neural network module that underlies the NicheCompass model.
"""

from typing import List, Literal, Optional, Tuple, Union

import mlflow
import numpy as np
import torch
import torch.nn as nn
from mlflow.exceptions import MlflowException
from torch_geometric.data import Data

from nichecompass.nn import (CosineSimGraphDecoder,
                             Encoder,
                             FCOmicsFeatureDecoder,
                             MaskedOmicsFeatureDecoder,
                             OneHopAttentionNodeLabelAggregator,
                             OneHopGCNNormNodeLabelAggregator,
                             OneHopSumNodeLabelAggregator)
from .basemodulemixin import BaseModuleMixin
from .losses import (compute_cat_covariates_contrastive_loss,
                     compute_edge_recon_loss,
                     compute_gp_group_lasso_reg_loss,
                     compute_gp_l1_reg_loss,
                     compute_kl_reg_loss,
                     compute_omics_recon_nb_loss)
from .vgaemodulemixin import VGAEModuleMixin


class VGPGAE(nn.Module, BaseModuleMixin, VGAEModuleMixin):
    """
    Variational Gene Program Graph Autoencoder class.

    Parameters
    ----------
    n_input:
        Number of nodes in the input layer.
    n_fc_layers_encoder:
        Number of fully connected layers in the encoder.
    n_layers_encoder:
        Number of message passing layers in the encoder.
    n_hidden_encoder:
        Number of nodes in the encoder hidden layer.
    n_prior_gp:
        Number of prior nodes in the latent space (gene programs from the
        gene program masks).
    n_addon_gp:
        Number of add-on nodes in the latent space (de-novo gene programs).
    cat_covariates_embeds_nums:
        List of number of embedding nodes for all categorical covariates.
    n_output_genes:
        Number of output genes for the rna decoders.
    target_rna_decoder_mask:
        Gene program mask for the target rna decoder.
    source_rna_decoder_mask:
        Gene program mask for the source rna decoder.
    features_idx_dict:
        Dictionary containing indices which omics features are masked and which
        are unmasked.
    n_output_peaks:
        Number of output peaks for the atac decoders.
    target_atac_decoder_mask:
        Gene program mask for the target atac decoder.
    source_atac_decoder_mask:
        Gene program mask for the source atac decoder.
    gene_peaks_mask:
        A mask to map from genes to peaks, used to turn off peaks in the atac
        decoders if the corresponding genes have been turned off in the rna
        decoders by gene regularization.
    cat_covariates_cats:
        List of category lists for each categorical covariate for the
        categorical covariates embeddings.
    cat_covariates_no_edges:
        List of booleans that indicate whether there can be edges between
        different categories of the categorical covariates. If this is ´True´
        for a specific categorical covariate, this covariate will be excluded
        from the edge reconstruction loss.
    conv_layer_encoder:
        Convolutional layer used in the encoder.
    encoder_n_attention_heads:
        Only relevant if ´conv_layer_encoder == gatv2conv´. Number of attention
        heads used.
    encoder_use_bn:
        If ´True´, uses a batch normalization layer at the end of the encoder to
        normalize ´mu´.
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
        loss optimization.
    rna_recon_loss:
        The loss used for gene expression reconstruction. If `nb`, uses a
        negative binomial loss.
    node_label_method:
        Node label method that will be used for omics reconstruction. If
        ´one-hop-sum´, uses a concatenation of the node's input features with
        the sum of the input  features of all nodes in the node's one-hop
        neighborhood. If ´one-hop-norm´, use a concatenation of the node`s input
        features with the node's one-hop neighbors input features normalized as
        per Kipf, T. N. & Welling, M. Semi-Supervised Classification with Graph
        Convolutional Networks. arXiv [cs.LG] (2016). If ´one-hop-attention´,
        uses a concatenation of the node`s input features with the node's
        one-hop neighbors input features weighted by an attention mechanism.
    active_gp_thresh_ratio:
        Ratio that determines which gene programs are considered active and are
        used for edge reconstruction and omics reconstruction. All inactive gene
        programs will be dropped out. Aggregations of the absolute values of the
        gene weights of the gene expression decoder per gene program are
        calculated. The maximum value, i.e. the value of the gene program with
        the highest aggregated value will be used as a benchmark and all gene
        programs whose aggregated value is smaller than ´active_gp_thresh_ratio´
        times this maximum value will be set to inactive. If ´==0´, all gene
        programs will be considered active. More information can be found in
        ´self.get_active_gp_mask()´.
    active_gp_type:
        Type to determine active gene programs. Can be ´mixed´, in which case
        active gene programs are determined across prior and add-on gene programs
        jointly or ´separate´ in which case they are determined separately for
        prior adn add-on gene programs.
    log_variational:
        If ´True´, transforms x by log(x+1) prior to encoding for numerical 
        stability (not normalization).
    cat_covariates_embeds_injection:
        List of VGPGAE modules in which the categorical covariates embeddings
        are injected.
    use_fc_decoder:
        If ´True´, uses a fully connected decoder instead of masked decoder.
        Just for ablation purposes.
    fc_decoder_n_layers:
        Number of layers to use if ´use_fc_decoder == True´.
    """
    def __init__(self,
                 n_input: int,
                 n_fc_layers_encoder: int,
                 n_layers_encoder: int,
                 n_hidden_encoder: int,
                 n_prior_gp: int,
                 n_addon_gp: int,
                 cat_covariates_embeds_nums: List[int],
                 n_output_genes: int,
                 target_rna_decoder_mask: torch.Tensor,
                 source_rna_decoder_mask: torch.Tensor,
                 features_idx_dict: dict,
                 features_scale_factors: torch.Tensor,
                 n_output_peaks: int=0,
                 target_atac_decoder_mask: Optional[torch.Tensor]=None,
                 source_atac_decoder_mask: Optional[torch.Tensor]=None,
                 gene_peaks_mask: Optional[torch.Tensor]=None,
                 cat_covariates_cats: List[List]=[],
                 cat_covariates_no_edges: List[bool]=[],
                 conv_layer_encoder: Literal["gcnconv", "gatv2conv"]="gcnconv",
                 encoder_n_attention_heads: int=4,
                 encoder_use_bn: bool=False,
                 dropout_rate_encoder: float=0.,
                 dropout_rate_graph_decoder: float=0.,
                 include_edge_recon_loss: bool=True,
                 include_gene_expr_recon_loss: bool=True,
                 include_chrom_access_recon_loss: bool=True,
                 include_cat_covariates_contrastive_loss: bool=True,
                 rna_recon_loss: Literal["nb"]="nb",
                 atac_recon_loss: Literal["nb"]="nb",
                 node_label_method: Literal[
                    "one-hop-norm",
                    "one-hop-sum",
                    "one-hop-attention"]="one-hop-norm",
                 active_gp_thresh_ratio: float=0.03,
                 active_gp_type: Literal["mixed", "separate"]="separate",
                 log_variational: bool=True,
                 cat_covariates_embeds_injection: Optional[List[
                     Literal["encoder",
                             "gene_expr_decoder",
                             "chrom_access_decoder"]]]=["gene_expr_decoder",
                                                        "chrom_access_decoder"],
                 use_fc_decoder: bool=False,
                 fc_decoder_n_layers: int=2,
                 include_edge_kl_loss: bool=True):
        super().__init__()
        print("--- INITIALIZING NEW NETWORK MODULE: VARIATIONAL GENE PROGRAM "
              "GRAPH AUTOENCODER ---")
        print(f"LOSS -> include_edge_recon_loss: {include_edge_recon_loss}, "
              f"include_gene_expr_recon_loss: {include_gene_expr_recon_loss}, "
              f"rna_recon_loss: {rna_recon_loss}", end="")
        if target_atac_decoder_mask is not None:
            print(", include_chrom_access_recon_loss: "
                  f"{include_chrom_access_recon_loss}, "
                  "atac_recon_loss: "
                  f"{atac_recon_loss}", end=" ")
        print(f"\nNODE LABEL METHOD -> {node_label_method}")
        print(f"ACTIVE GP THRESHOLD RATIO -> {active_gp_thresh_ratio}")
        print(f"LOG VARIATIONAL -> {log_variational}")
        if len(cat_covariates_cats) != 0:
            print("CATEGORICAL COVARIATES EMBEDDINGS INJECTION -> "
                  f"{cat_covariates_embeds_injection}")

        self.n_input_ = n_input
        self.n_fc_layers_encoder_ = n_fc_layers_encoder
        self.n_layers_encoder_ = n_layers_encoder
        self.n_hidden_encoder_ = n_hidden_encoder
        self.n_prior_gp_ = n_prior_gp
        self.n_addon_gp_ = n_addon_gp
        self.cat_covariates_embeds_nums_ = cat_covariates_embeds_nums
        self.n_output_genes_ = n_output_genes
        self.target_rna_decoder_mask = target_rna_decoder_mask
        self.source_rna_decoder_mask = source_rna_decoder_mask
        self.n_output_peaks_ = n_output_peaks
        self.target_atac_decoder_mask = target_atac_decoder_mask
        self.source_atac_decoder_mask = source_atac_decoder_mask
        self.features_idx_dict_ = features_idx_dict
        self.features_scale_factors_ = features_scale_factors
        self.gene_peaks_mask_ = gene_peaks_mask
        #assert(torch.all(torch.logical_or(gene_peaks_mask == 0,
        #                                  gene_peaks_mask == 1)))
        self.cat_covariates_cats_ = cat_covariates_cats
        self.n_cat_covariates_ = len(cat_covariates_cats)
        self.cat_covariates_no_edges_ = cat_covariates_no_edges
        self.nums_cat_covariates_cats_ = [
            len(cat_covariate_cats) for cat_covariate_cats in
            cat_covariates_cats]
        self.cat_covariates_label_encoders_ = [
            {k: v for k, v in zip(cat_covariate_cats,
                                  range(len(cat_covariate_cats)))}
                                  for cat_covariate_cats in cat_covariates_cats]
        self.conv_layer_encoder_ = conv_layer_encoder
        self.encoder_n_attention_heads_ = encoder_n_attention_heads
        self.encoder_use_bn_ = encoder_use_bn
        self.dropout_rate_encoder_ = dropout_rate_encoder
        self.dropout_rate_graph_decoder_ = dropout_rate_graph_decoder
        self.include_edge_recon_loss_ = include_edge_recon_loss
        self.include_gene_expr_recon_loss_ = include_gene_expr_recon_loss
        self.include_chrom_access_recon_loss_ = include_chrom_access_recon_loss
        self.include_cat_covariates_contrastive_loss_ = include_cat_covariates_contrastive_loss
        self.include_edge_kl_loss_ = include_edge_kl_loss
        self.rna_recon_loss_ = rna_recon_loss
        self.atac_recon_loss_ = atac_recon_loss
        self.node_label_method_ = node_label_method
        self.active_gp_thresh_ratio_ = active_gp_thresh_ratio
        self.active_gp_type_ = active_gp_type
        self.log_variational_ = log_variational
        self.cat_covariates_embeds_injection_ = cat_covariates_embeds_injection
        self.freeze_ = False
        self.modalities_ = ["rna"]
        if target_atac_decoder_mask is not None:
            self.modalities_.append("atac")
            
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
        
        # Initialize node-label aggregator module
        if node_label_method == "one-hop-norm":
            self.rna_node_label_aggregator = OneHopGCNNormNodeLabelAggregator(
                modality="rna")
        elif node_label_method == "one-hop-sum":
            self.rna_node_label_aggregator = OneHopSumNodeLabelAggregator(
                modality="rna")
        elif node_label_method == "one-hop-attention":
            self.rna_node_label_aggregator = OneHopAttentionNodeLabelAggregator(
                modality="rna",
                n_input=n_input)
        if "atac" in self.modalities_:
            if node_label_method == "one-hop-norm":
                self.atac_node_label_aggregator = OneHopGCNNormNodeLabelAggregator(
                    modality="atac")
            elif node_label_method == "one-hop-sum":
                self.atac_node_label_aggregator = OneHopSumNodeLabelAggregator(
                    modality="atac")

        # Initialize encoder module
        self.encoder = Encoder(
            n_input=n_input,
            n_cat_covariates_embed_input=(
                sum(cat_covariates_embeds_nums) if 
                ("encoder" in self.cat_covariates_embeds_injection_) &
                (self.n_cat_covariates_ > 0)
                else 0),
            n_fc_layers=n_fc_layers_encoder,
            n_layers=n_layers_encoder,
            n_hidden=n_hidden_encoder,
            n_latent=n_prior_gp,
            n_addon_latent=n_addon_gp,
            conv_layer=conv_layer_encoder,
            n_attention_heads=encoder_n_attention_heads,
            dropout_rate=dropout_rate_encoder,
            activation=torch.relu,
            use_bn=encoder_use_bn)
        
        # Initialize graph decoder module
        self.graph_decoder = CosineSimGraphDecoder(
            dropout_rate=dropout_rate_graph_decoder)

        if not use_fc_decoder:
            # Check validity of mask indices
            target_rna_idx_intersect = set(
                features_idx_dict["target_masked_rna_idx"]).intersection(
                set(features_idx_dict["target_unmasked_rna_idx"]))
            assert len(target_rna_idx_intersect) == 0
            source_rna_idx_intersect = set(
                features_idx_dict["source_masked_rna_idx"]).intersection(
                set(features_idx_dict["source_unmasked_rna_idx"]))
            assert len(source_rna_idx_intersect) == 0

            for entity in ["target", "source"]:
                if n_addon_gp > 0:
                    # Initialize rna add-on masks which are 0 everywhere except
                    # for the genes that are unmasked, in which case they are 1
                    rna_decoder_addon_mask = torch.zeros(
                        n_addon_gp,
                        n_output_genes,
                        dtype=torch.float32)
                    rna_decoder_addon_mask[
                        :, features_idx_dict[f"{entity}_unmasked_rna_idx"]] = 1.
                    setattr(self,
                            f"{entity}_rna_decoder_addon_mask",
                            rna_decoder_addon_mask)
                    
                    # Set add-on rna idx to unmasked rna idx as all unmasked
                    # genes are part of add-on gps
                    features_idx_dict[f"{entity}_addon_rna_idx"] = (
                        features_idx_dict[f"{entity}_unmasked_rna_idx"])
                    
                    if "atac" in self.modalities_:
                        # Initialize atac add-on masks which are 0 everywhere
                        # except for the peaks that are mapped to genes that are
                        # unmasked, in which case they are 1
                        atac_decoder_addon_mask = torch.mm(
                            getattr(self,
                                    f"{entity}_rna_decoder_addon_mask").to(torch.int),
                            self.gene_peaks_mask_.to(torch.int)).to(torch.bool)
                        setattr(self,
                                f"{entity}_atac_decoder_addon_mask",
                                atac_decoder_addon_mask)

                        # Determine add-on atac idx based on peaks that are
                        # mapped to unmasked genes
                        features_idx_dict[f"{entity}_addon_atac_idx"] = (
                            torch.nonzero(
                            (atac_decoder_addon_mask.sum(axis=0) > 0)
                            ).squeeze().tolist())
                else:
                    for modality in self.modalities_:
                        setattr(self,
                                f"{entity}_{modality}_decoder_addon_mask",
                                None)
                        features_idx_dict[f"{entity}_addon_{modality}_idx"] = None   

            # Initialize masked gene expression decoders
            self.target_rna_decoder = MaskedOmicsFeatureDecoder(
                modality="rna",
                entity="target",
                n_prior_gp_input=n_prior_gp,
                n_addon_gp_input=n_addon_gp,
                n_cat_covariates_embed_input=(
                    sum(cat_covariates_embeds_nums) if
                    ("gene_expr_decoder" in
                     self.cat_covariates_embeds_injection_) &
                     (self.n_cat_covariates_ > 0)
                     else 0),
                n_output=n_output_genes,
                mask=self.target_rna_decoder_mask,
                addon_mask=self.target_rna_decoder_addon_mask,
                masked_features_idx=features_idx_dict["target_masked_rna_idx"],
                recon_loss=self.rna_recon_loss_)
            self.source_rna_decoder = MaskedOmicsFeatureDecoder(
                modality="rna",
                entity="source",
                n_prior_gp_input=n_prior_gp,
                n_addon_gp_input=n_addon_gp,
                n_cat_covariates_embed_input=(
                    sum(cat_covariates_embeds_nums) if
                    ("gene_expr_decoder" in
                     self.cat_covariates_embeds_injection_) &
                     (self.n_cat_covariates_ > 0)
                     else 0),
                n_output=n_output_genes,
                mask=self.source_rna_decoder_mask,
                addon_mask=self.source_rna_decoder_addon_mask,
                masked_features_idx=features_idx_dict["source_masked_rna_idx"],
                recon_loss=self.rna_recon_loss_)
        else:
            # Initialize fc expression decoders
            self.target_rna_decoder = FCOmicsFeatureDecoder(
                modality="rna",
                entity="target",
                n_prior_gp_input=n_prior_gp,
                n_addon_gp_input=n_addon_gp,
                n_cat_covariates_embed_input=(
                    sum(cat_covariates_embeds_nums) if
                    ("gene_expr_decoder" in
                     self.cat_covariates_embeds_injection_) &
                     (self.n_cat_covariates_ > 0)
                     else 0),
                n_output=n_output_genes,
                n_layers=fc_decoder_n_layers,
                recon_loss=self.rna_recon_loss_)
            self.source_rna_decoder = FCOmicsFeatureDecoder(
                modality="rna",
                entity="source",
                n_prior_gp_input=n_prior_gp,
                n_addon_gp_input=n_addon_gp,
                n_cat_covariates_embed_input=(
                    sum(cat_covariates_embeds_nums) if
                    ("gene_expr_decoder" in
                     self.cat_covariates_embeds_injection_) &
                     (self.n_cat_covariates_ > 0)
                     else 0),
                n_output=n_output_genes,
                n_layers=fc_decoder_n_layers,
                recon_loss=self.rna_recon_loss_)          
        
        # Initialize gene-specific dispersion parameters for all genes
        self.target_rna_theta = torch.nn.Parameter(torch.randn(
            n_output_genes))
        self.source_rna_theta = torch.nn.Parameter(torch.randn(
            n_output_genes))
        
        # Initialize running mean abs gp scores
        self.register_buffer("running_mean_abs_mu",
                             torch.zeros(n_prior_gp + n_addon_gp))
        
        # Initialize rna dynamic decoder masks
        self.target_rna_dynamic_decoder_mask = torch.ones(
            (n_prior_gp + n_addon_gp), n_output_genes, dtype=torch.bool)
        self.source_rna_dynamic_decoder_mask = torch.ones(
            (n_prior_gp + n_addon_gp), n_output_genes, dtype=torch.bool)
        
        if "atac" in self.modalities_:
            if not use_fc_decoder:
            # Check validity of mask indices
                target_atac_idx_intersect = set(
                    features_idx_dict["target_masked_atac_idx"]).intersection(
                    set(features_idx_dict["target_unmasked_atac_idx"]))
                assert len(target_atac_idx_intersect) == 0
                source_atac_idx_intersect = set(
                    features_idx_dict["source_masked_atac_idx"]).intersection(
                    set(features_idx_dict["source_unmasked_atac_idx"]))
                assert len(source_atac_idx_intersect) == 0               
                
                # Initialize masked atac decoders
                self.target_atac_decoder = MaskedOmicsFeatureDecoder(
                    modality="atac",
                    entity="target",
                    n_prior_gp_input=n_prior_gp,
                    n_addon_gp_input=n_addon_gp,
                    n_cat_covariates_embed_input=(
                        sum(cat_covariates_embeds_nums) if
                        ("chrom_access_decoder" in
                         self.cat_covariates_embeds_injection_) &
                         (self.n_cat_covariates_ > 0)
                         else 0),
                    n_output=n_output_peaks,
                    mask=self.target_atac_decoder_mask,
                    addon_mask=self.target_atac_decoder_addon_mask,
                    masked_features_idx=features_idx_dict[
                        "target_masked_atac_idx"],
                    recon_loss="nb")
                self.source_atac_decoder = MaskedOmicsFeatureDecoder(
                    modality="atac",
                    entity="source",
                    n_prior_gp_input=n_prior_gp,
                    n_addon_gp_input=n_addon_gp,
                    n_cat_covariates_embed_input=(
                        sum(cat_covariates_embeds_nums) if
                        ("chrom_access_decoder" in
                         self.cat_covariates_embeds_injection_) &
                         (self.n_cat_covariates_ > 0)
                         else 0),
                    n_output=n_output_peaks,
                    mask=self.source_atac_decoder_mask,
                    addon_mask=self.source_atac_decoder_addon_mask,
                    masked_features_idx=features_idx_dict[
                        "source_masked_atac_idx"],
                    recon_loss="nb")
            else:
                # Initialize fc atac decoders
                self.target_atac_decoder = FCOmicsFeatureDecoder(
                    modality="atac",
                    entity="target",
                    n_prior_gp_input=n_prior_gp,
                    n_addon_gp_input=n_addon_gp,
                    n_cat_covariates_embed_input=(
                        sum(cat_covariates_embeds_nums) if
                        ("chrom_access_decoder" in
                         self.cat_covariates_embeds_injection_) &
                         (self.n_cat_covariates_ > 0)
                         else 0),
                    n_output=n_output_peaks,
                    n_layers=fc_decoder_n_layers,
                    recon_loss="nb")

                self.source_atac_decoder = FCOmicsFeatureDecoder(
                    modality="atac",
                    entity="source",
                    n_prior_gp_input=n_prior_gp,
                    n_addon_gp_input=n_addon_gp,
                    n_cat_covariates_embed_input=(
                        sum(cat_covariates_embeds_nums) if
                        ("chrom_access_decoder" in
                         self.cat_covariates_embeds_injection_) &
                         (self.n_cat_covariates_ > 0)
                         else 0),
                    n_output=n_output_peaks,
                    n_layers=fc_decoder_n_layers,
                    recon_loss="nb")
                            
            # Initialize peak-specific dispersion parameters
            self.target_atac_theta = torch.nn.Parameter(torch.randn(
                n_output_peaks))
            self.source_atac_theta = torch.nn.Parameter(torch.randn(
                n_output_peaks))
            
            # Initialize atac dynamic decoder masks
            self.target_atac_dynamic_decoder_mask = torch.ones(
                (n_prior_gp + n_addon_gp), n_output_peaks, dtype=torch.bool)
            self.source_atac_dynamic_decoder_mask = torch.ones(
                (n_prior_gp + n_addon_gp), n_output_peaks, dtype=torch.bool)

    def forward(self,
                data_batch: Data,
                decoder: Literal["graph", "omics"],
                use_only_active_gps: bool=False,
                return_agg_weights: bool=False,
                update_atac_dynamic_decoder_mask: bool=False) -> dict:
        """
        Forward pass of the VGPGAE module.

        Parameters
        ----------
        data_batch:
            PyG Data object containing either an edge-level batch if 
            ´decoder == graph´ or a node-level batch if ´decoder == omics´.
        decoder:
            Decoder to use for the forward pass. Either ´graph´ for edge
            reconstruction or ´omics´ for gene expression and (if specified)
            chromatin accessibility reconstruction.
        use_only_active_gps:
            If ´True´, use only active gene programs as input to decoder.
        return_agg_weights:
            If ´True´, also return the aggregation weights of the node label
            aggregator.
        update_atac_dynamic_decoder_mask:
            If ´True´, turn off the mapped peaks for genes that have been
            turned off in a gene program (set peak gp weights to 0).

        Returns
        ----------
        output:
            Dictionary containing reconstructed edge logits if
            ´decoder == graph´ or the parameters of the omics feature
            distributions if ´decoder == omics´, as well as ´mu´ and ´logstd´ 
            from the latent space distribution.
        """
        x = data_batch.x # dim: n_obs x n_omics_features
        edge_index = data_batch.edge_index # dim: 2 x n_edges (incl. all edges
                                           # of sampled graph)
        
        # Get index of sampled nodes for current batch (neighbors of sampled
        # nodes are also part of the batch for message passing layers but
        # should be excluded in backpropagation)
        if decoder == "omics":
            # ´data_batch´ will be a node batch and first node_batch_size
            # elements are the sampled nodes, leading to a dim of ´batch_idx´ of
            # ´node_batch_size´
            batch_idx = slice(None, data_batch.batch_size)
        elif decoder == "graph":
            # ´data_batch´ will be an edge batch with sampled positive and
            # negative edges of size ´edge_batch_size´ respectively. Each edge
            # has a source and target node, leading to a dim of ´batch_idx´ of
            # 4 * ´edge_batch_size´
            batch_idx = torch.cat((data_batch.edge_label_index[0],
                                   data_batch.edge_label_index[1]), 0)

        # Logarithmitize omics feature vector (only) for encoder input for
        # numerical stability. This will not affect node labels.
        if self.log_variational_:
            x_enc = torch.log(1 + x)
        else:
            x_enc = x
            
        # Get categorical covariates embedding
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

        output = {}
        
        # Use encoder to get latent distribution parameters for current batch
        # and reparameterization trick to get latent features (gp scores).
        # Filter for nodes in current batch
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
            active_gp_mask = self.get_active_gp_mask()
            
            # Set gp scores of inactive gene programs to 0 to not affect 
            # graph decoder
            z[:, ~active_gp_mask] = 0                

        if decoder == "omics":
            with torch.no_grad():
                if self.training:
                    # Update running mean absolute gp scores using exponential
                    # moving average with momentum of 0.1
                    mean_abs_mu = self.mu.norm(p=1, dim=0) / self.mu.size(0)
                    self.running_mean_abs_mu = (
                        0.1 * mean_abs_mu + 0.9 * self.running_mean_abs_mu)
                    
                if use_only_active_gps:
                    # Set running mean abs mu of inactive gene programs to 0 for
                    # active gp determination
                    self.running_mean_abs_mu[~active_gp_mask] = 0  

                    # Set dynamic mask to 0 for all inactive gene programs to
                    # not affect omics decoders
                    self.target_rna_dynamic_decoder_mask[~active_gp_mask, :] = 0
                    self.source_rna_dynamic_decoder_mask[~active_gp_mask, :] = 0

                    if "atac" in self.modalities_:
                        self.target_atac_dynamic_decoder_mask[~active_gp_mask, :] = 0
                        self.source_atac_dynamic_decoder_mask[~active_gp_mask, :] = 0
                    
            # Determine which features should be reconstructed based on
            # static and dynamic masks (if a feature is not connected to any
            # node it should not be reconstructed to not influence softmax
            # activation outputs). This can happen when no add-on gene programs
            # are present or when gene programs are turned off.
            if self.n_addon_gp_ > 0:
                target_rna_decoder_static_mask = torch.cat(
                    (self.target_rna_decoder_mask,
                     self.target_rna_decoder_addon_mask), dim=0)
                source_rna_decoder_static_mask = torch.cat(
                    (self.source_rna_decoder_mask,
                     self.source_rna_decoder_addon_mask), dim=0)
            else:
                target_rna_decoder_static_mask = self.target_rna_decoder_mask
                source_rna_decoder_static_mask = self.source_rna_decoder_mask

            self.target_n_gps_per_gene = (
                target_rna_decoder_static_mask
                * self.target_rna_dynamic_decoder_mask
                ).sum(0)
            self.features_idx_dict_["target_reconstructed_rna_idx"] = (
                torch.nonzero(self.target_n_gps_per_gene)).flatten().tolist()

            self.source_n_gps_per_gene = (
                source_rna_decoder_static_mask
                * self.source_rna_dynamic_decoder_mask
                ).sum(0)
            self.features_idx_dict_["source_reconstructed_rna_idx"] = (
                torch.nonzero(self.source_n_gps_per_gene)).flatten().tolist()

            self.target_rna_theta_reconstructed = self.target_rna_theta[
                self.features_idx_dict_["target_reconstructed_rna_idx"]]
            self.source_rna_theta_reconstructed = self.source_rna_theta[
                self.features_idx_dict_["source_reconstructed_rna_idx"]]
                    
            output["node_labels"] = {}

            # Get rna and atac part from omics feature vector
            x_atac = x[:, self.n_output_genes_:]
            x = x[:, :self.n_output_genes_]
        
            # Compute aggregated neighborhood rna feature vector
            rna_node_label_aggregator_output = self.rna_node_label_aggregator(
                    x=x,
                    edge_index=edge_index,
                    return_agg_weights=return_agg_weights)
            x_neighbors = rna_node_label_aggregator_output[0]
 
            # Retrieve rna node labels and only keep nodes in current node batch
            # and reconstructed features
            assert x.size(1) == self.n_output_genes_
            assert x_neighbors.size(1) == self.n_output_genes_
            output["node_labels"]["target_rna"] = x[batch_idx][
                :, self.features_idx_dict_["target_reconstructed_rna_idx"]]
            output["node_labels"]["source_rna"] = x_neighbors[batch_idx][
                :, self.features_idx_dict_["source_reconstructed_rna_idx"]]
            
            # Use observed library size as scaling factor for the negative
            # binomial means of the rna distribution
            target_rna_library_size = output["node_labels"]["target_rna"].sum(
                1).unsqueeze(1)
            source_rna_library_size = output["node_labels"]["source_rna"].sum(
                1).unsqueeze(1)
            self.target_rna_log_library_size = torch.log(target_rna_library_size)
            self.source_rna_log_library_size = torch.log(source_rna_library_size)

            # Get gene expression reconstruction distribution parameters for
            # reconstructed genes
            output["target_rna_nb_means"] = self.target_rna_decoder(
                z=z,
                log_library_size=self.target_rna_log_library_size,
                cat_covariates_embed=(
                    self.cat_covariates_embed[batch_idx] if
                    (self.cat_covariates_embed is not None) &
                    ("gene_expr_decoder" in
                     self.cat_covariates_embeds_injection_)
                     else None))[
                    :, self.features_idx_dict_["target_reconstructed_rna_idx"]]
            output["source_rna_nb_means"] = self.source_rna_decoder(
                z=z,
                log_library_size=self.source_rna_log_library_size,
                cat_covariates_embed=(
                self.cat_covariates_embed[batch_idx] if
                (self.cat_covariates_embed is not None) &
                ("gene_expr_decoder" in
                 self.cat_covariates_embeds_injection_)
                 else None))[
                    :, self.features_idx_dict_["source_reconstructed_rna_idx"]]
            
            if "atac" in self.modalities_:
                # Determine which features should be reconstructed based on
                # masks (if a feature is not connected to any node it should not
                # be reconstructed to not influence softmax activation outputs)
                if self.n_addon_gp_ > 0:
                    target_atac_decoder_static_mask = torch.cat(
                        (self.target_atac_decoder_mask,
                         self.target_atac_decoder_addon_mask), dim=0)
                    source_atac_decoder_static_mask = torch.cat(
                        (self.source_atac_decoder_mask,
                         self.source_atac_decoder_addon_mask), dim=0)
                else:
                    target_atac_decoder_static_mask = self.target_atac_decoder_mask
                    source_atac_decoder_static_mask = self.source_atac_decoder_mask

                self.target_n_gps_per_peak = (
                    target_atac_decoder_static_mask
                    * self.target_atac_dynamic_decoder_mask
                    ).sum(0)
                self.features_idx_dict_["target_reconstructed_atac_idx"] = (
                    torch.nonzero(self.target_n_gps_per_peak)).flatten().tolist()

                self.source_n_gps_per_peak = (
                    source_atac_decoder_static_mask
                    * self.source_atac_dynamic_decoder_mask
                    ).sum(0)
                self.features_idx_dict_["source_reconstructed_atac_idx"] = (
                    torch.nonzero(self.source_n_gps_per_peak)).flatten().tolist()

                self.target_atac_theta_reconstructed = self.target_atac_theta[
                    self.features_idx_dict_["target_reconstructed_atac_idx"]]
                self.source_atac_theta_reconstructed = self.source_atac_theta[
                    self.features_idx_dict_["source_reconstructed_atac_idx"]]

                # Compute aggregated neighborhood atac feature vector
                atac_node_label_aggregator_output = (
                    self.atac_node_label_aggregator(
                        x=x_atac,
                        edge_index=edge_index,
                        return_agg_weights=return_agg_weights))
                x_neighbors_atac = atac_node_label_aggregator_output[0]

                # Retrieve node labels and only keep nodes in current node batch
                # and reconstructed features
                assert x_atac.size(1) == self.n_output_peaks_
                assert x_neighbors_atac.size(1) == self.n_output_peaks_
                output["node_labels"]["target_atac"] = x_atac[batch_idx][
                    :, self.features_idx_dict_["target_reconstructed_atac_idx"]]  
                output["node_labels"]["source_atac"] = x_neighbors_atac[batch_idx][
                    :, self.features_idx_dict_["source_reconstructed_atac_idx"]]

                # Use observed library size as scaling factor for the negative
                # binomial means of the atac distribution
                target_atac_library_size = output["node_labels"][
                    "target_atac"].sum(1).unsqueeze(1)
                source_atac_library_size = output["node_labels"][
                    "source_atac"].sum(1).unsqueeze(1)
                self.target_atac_log_library_size = torch.log(
                    target_atac_library_size)
                self.source_atac_log_library_size = torch.log(
                    source_atac_library_size)
                
                if update_atac_dynamic_decoder_mask:
                    # Get atac dynamic decoder masks to turn off peaks that
                    # are mapped to only genes that are turned off
                    with torch.no_grad():
                        # Retrieve rna decoder gp weights
                        gp_weights = self.get_gp_weights(
                            only_masked_features=False)[0].detach().cpu()
                        
                        # Round to 4 decimals as genes are never completely
                        # turned off due to L1 being not differentiable at 0
                        gp_weights = torch.round(gp_weights, decimals=4)

                        # Get boolean mask of non zero target and source gene 
                        # weights
                        non_zero_gene_weights = torch.ne(
                                gp_weights, 
                                0) # dim: (2 x n_genes, n_gps)
                        non_zero_target_gene_weights = non_zero_gene_weights[
                            :self.n_output_genes_, :] # dim: (n_genes, n_gps)
                        non_zero_source_gene_weights = non_zero_gene_weights[
                            self.n_output_genes_:, :] # dim: (n_genes, n_gps)
                        
                        # Multiply boolean mask with gene peak mapping to remove
                        # peaks that are mapped to only turned off genes
                        target_atac_dynamic_decoder_mask = torch.mm(
                            non_zero_target_gene_weights.t().to(torch.float32), # dim: (n_gps,
                                                              #       n_genes)
                            self.gene_peaks_mask_.to(torch.float32)).to(torch.bool) # dim: (n_genes,
                                                   # n_peaks)
                            # dim: (n_gps, n_peaks)
                        source_atac_dynamic_decoder_mask = torch.mm(
                            non_zero_source_gene_weights.t().to(torch.float32),
                            self.gene_peaks_mask_.to(torch.float32)).to(torch.bool)
                        
                        # Create boolean mask of peaks (until here multiple
                        # active genes in a gp can be mapped to the same peak,
                        # resulting in values > 1.)
                        self.target_atac_dynamic_decoder_mask = (
                            self.target_atac_dynamic_decoder_mask & torch.ne(
                            target_atac_dynamic_decoder_mask, 
                            0)) # dim: (n_gps, n_peaks)
                        self.source_atac_dynamic_decoder_mask = (
                            self.source_atac_dynamic_decoder_mask & torch.ne(
                            source_atac_dynamic_decoder_mask, 
                            0))
                    
                # Get chromatin accessibility reconstruction distribution
                # parameters for reconstructed peaks
                output["target_atac_nb_means"] = self.target_atac_decoder(
                    z=z,
                    log_library_size=self.target_atac_log_library_size,
                    dynamic_mask=self.target_atac_dynamic_decoder_mask,
                    cat_covariates_embed=(
                        self.cat_covariates_embed[batch_idx] if
                        (self.cat_covariates_embed is not None) & 
                        ("chrom_access_decoder" in
                         self.cat_covariates_embeds_injection_) else
                        None))[
                    :, self.features_idx_dict_["target_reconstructed_atac_idx"]]
                output["source_atac_nb_means"] = self.source_atac_decoder(
                    z=z,
                    log_library_size=self.source_atac_log_library_size,
                    dynamic_mask=self.source_atac_dynamic_decoder_mask,
                    cat_covariates_embed=(
                        self.cat_covariates_embed[batch_idx] if
                        (self.cat_covariates_embed is not None) &
                        ("chrom_access_decoder" in
                         self.cat_covariates_embeds_injection_) else
                        None))[
                    :, self.features_idx_dict_["source_reconstructed_atac_idx"]]
        elif decoder == "graph":
            # Store edge labels in output for loss computation
            output["edge_recon_labels"] = data_batch.edge_label
                
            # For each categorical covariate, create a boolean tensor to
            # indicate for each sampled edge (negative & positive edges) whether
            # the edge nodes have the same category and store in a list
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
                
                # Based on the categorical covariate and its possibility for
                # edges to exist for different categories (this might only be
                # the case for certain categorical covariates, others might only
                # use disconnected neighbor graphs for different categories),
                # create a boolean mask for edges whether they should be
                # included in the edge reconstruction loss and edge
                # reconstruction performance evaluation
                cat_covariates_cat_edge_incl = []
                for cat_covariate_no_edge, edge_same_cat_covariate_cat in zip(
                    self.cat_covariates_no_edges_,
                    output["edge_same_cat_covariates_cat"]):
                    if not cat_covariate_no_edge:
                        cat_covariates_cat_edge_incl.append(
                            torch.ones_like(edge_same_cat_covariate_cat,
                                            dtype=torch.bool))
                    else:
                        cat_covariates_cat_edge_incl.append(
                            edge_same_cat_covariate_cat)
                output["edge_incl"] = torch.all(
                    torch.stack(cat_covariates_cat_edge_incl),
                                dim=0)
            else:
                output["edge_same_cat_covariates_cat"] = None
                output["edge_incl"] = None

            # Use decoder to get the edge reconstruction logits
            output["edge_recon_logits"] = self.graph_decoder(z=z)
        return output

    def loss(self,
             edge_model_output: dict,
             node_model_output: dict,
             lambda_l1_masked: float,
             l1_targets_mask: torch.Tensor,
             l1_sources_mask: torch.Tensor,
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
        (not backpropagated) but is used for early stopping evaluation.

        Parameters
        ----------
        edge_model_output:
            Output of the edge-level forward pass for edge reconstruction.
        node_model_output:
            Output of the node-level forward pass for omics reconstruction.
        lambda_l1_masked:
            Lambda (weighting factor) for the L1 regularization loss of genes in
            masked gene programs. If ´>0´, this will enforce sparsity of genes
            in masked gene programs.
        l1_targets_mask:
            Boolean gene program gene mask that is True for all gene program
            target genes to which the L1 regularization loss should be applied
            (dim: n_genes, n_gps).
        l1_sources_mask:
            Boolean gene program gene mask that is True for all gene program
            source genes to which the L1 regularization loss should be applied
            (dim: n_genes, n_gps).
        lambda_l1_addon:
            Lambda (weighting factor) for the L1 regularization loss of genes in
            addon gene programs. If ´>0´, this will enforce sparsity of genes in
            addon gene programs.
        lambda_group_lasso:
            Lambda (weighting factor) for the group lasso regularization loss of
            masked gene programs. If ´>0´, this will enforce sparsity of masked
            gene programs.
        lambda_gene_expr_recon:
            Lambda (weighting factor) for the gene expression reconstruction
            loss. If ´>0´, this will enforce interpretable gene programs that
            can be combined in a linear way to reconstruct gene expression.
        lambda_chrom_access_recon:
            Lambda (weighting factor) for the chromatin accessibility
            reconstruction loss. If ´>0´, this will enforce interpretable gene
            programs that can be combined in a linear way to reconstruct
            chromatin accessibility.
        lambda_edge_recon:
            Lambda (weighting factor) for the edge reconstruction loss. If ´>0´,
            this will enforce gene programs to be meaningful for edge
            reconstruction and, hence, to preserve spatial colocalization
            information.
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
            mu=node_model_output["mu"],
            logstd=node_model_output["logstd"])
        if self.include_edge_kl_loss_:
            loss_dict["kl_reg_loss"] += compute_kl_reg_loss(
                mu=edge_model_output["mu"],
                logstd=edge_model_output["logstd"])

        # Compute edge reconstruction binary cross entropy loss for edge batch
        loss_dict["edge_recon_loss"] = (
            lambda_edge_recon * compute_edge_recon_loss(
                edge_recon_logits=edge_model_output["edge_recon_logits"],
                edge_recon_labels=edge_model_output["edge_recon_labels"],
                edge_incl=edge_model_output["edge_incl"]))
            
        # Compute categorical covariates contrastive loss for edge batch
        if (edge_model_output["edge_same_cat_covariates_cat"] is not None) & (
        lambda_cat_covariates_contrastive > 0):
            loss_dict["cat_covariates_contrastive_loss"] = (
                lambda_cat_covariates_contrastive *
                compute_cat_covariates_contrastive_loss(
                    edge_recon_logits=edge_model_output["edge_recon_logits"],
                    edge_recon_labels=edge_model_output["edge_recon_labels"],
                    edge_same_cat_covariates_cat=edge_model_output[
                        "edge_same_cat_covariates_cat"],
                    contrastive_logits_pos_ratio=contrastive_logits_pos_ratio,
                    contrastive_logits_neg_ratio=contrastive_logits_neg_ratio))

        # Compute target and source gene expression reconstruction negative
        # binomial loss for node batch
        if self.rna_recon_loss_ == "nb":
            loss_dict["gene_expr_recon_loss"] = (
                lambda_gene_expr_recon * 
            compute_omics_recon_nb_loss(
                    x=node_model_output["node_labels"]["target_rna"],
                    mu=node_model_output["target_rna_nb_means"],
                    theta=torch.exp(self.target_rna_theta_reconstructed)))
            loss_dict["gene_expr_recon_loss"] += (
                lambda_gene_expr_recon * 
            compute_omics_recon_nb_loss(
                    x=node_model_output["node_labels"]["source_rna"],
                    mu=node_model_output["source_rna_nb_means"],
                    theta=torch.exp(self.source_rna_theta_reconstructed)))
            
        # Compute l1 reg loss of genes in masked gene programs
        loss_dict["masked_gp_l1_reg_loss"] = (lambda_l1_masked *
            compute_gp_l1_reg_loss(
                self,
                gp_type="prior",
                l1_targets_mask=l1_targets_mask,
                l1_sources_mask=l1_sources_mask))

        # Compute group lasso regularization loss of masked gene programs
        loss_dict["group_lasso_reg_loss"] = (lambda_group_lasso *
            compute_gp_group_lasso_reg_loss(self))

        # Compute l1 regularization loss of genes in addon gene programs
        if self.n_addon_gp_ != 0:
            loss_dict["addon_gp_l1_reg_loss"] = (lambda_l1_addon *
            compute_gp_l1_reg_loss(self,
                                   gp_type="addon"))

        if "atac" in self.modalities_:
            # Compute target and source chromatin accessibility reconstruction
            # negative binomial loss for node batch
            loss_dict["chrom_access_recon_loss"] = (
                lambda_chrom_access_recon * 
            compute_omics_recon_nb_loss(
                    x=node_model_output["node_labels"]["target_atac"],
                    mu=node_model_output["target_atac_nb_means"],
                    theta=torch.exp(self.target_atac_theta_reconstructed))) 
            loss_dict["chrom_access_recon_loss"] += (
                lambda_chrom_access_recon * 
            compute_omics_recon_nb_loss(
                    x=node_model_output["node_labels"]["source_atac"],
                    mu=node_model_output["source_atac_nb_means"],
                    theta=torch.exp(self.source_atac_theta_reconstructed)))

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
            loss_dict["global_loss"] += loss_dict[
                "cat_covariates_contrastive_loss"]
            if cat_covariates_contrastive_active:
                loss_dict["optim_loss"] += loss_dict[
                    "cat_covariates_contrastive_loss"] 
        if self.include_gene_expr_recon_loss_:
            loss_dict["global_loss"] += loss_dict["gene_expr_recon_loss"]
            loss_dict["optim_loss"] += loss_dict["gene_expr_recon_loss"]
            loss_dict["global_loss"] += loss_dict["group_lasso_reg_loss"]
            loss_dict["optim_loss"] += loss_dict["group_lasso_reg_loss"]
            loss_dict["global_loss"] += loss_dict["masked_gp_l1_reg_loss"]
            loss_dict["optim_loss"] += loss_dict["masked_gp_l1_reg_loss"]
            if self.n_addon_gp_ != 0:
                loss_dict["global_loss"] += loss_dict["addon_gp_l1_reg_loss"]
                loss_dict["optim_loss"] += loss_dict["addon_gp_l1_reg_loss"]
        if ("atac" in self.modalities_) & self.include_chrom_access_recon_loss_:
            loss_dict["global_loss"] += loss_dict["chrom_access_recon_loss"]
            loss_dict["optim_loss"] += loss_dict["chrom_access_recon_loss"]
        return loss_dict

    @torch.no_grad()
    def get_gp_weights(self,
                       only_masked_features: bool=False,
                       gp_type: Literal["all", "prior", "addon"]="all"
                       ) -> List[torch.Tensor]:
        """
        Get the gene program weights of the omics feature decoders.

        Returns:
        ----------
        gp_weights_all_modalities:
            List of tensors containing the decoder gp weights for each
            omics modality (dim: (n_prior_gp + n_addon_gp) x n_omics_features)
        """
        gp_weights_all_modalities = []

        for modality in self.modalities_:
            target_decoder = getattr(self, f"target_{modality}_decoder")
            source_decoder = getattr(self, f"source_{modality}_decoder")

            if gp_type != "addon":
                # Get decoder weights of masked gps
                target_gp_weights_masked = (
                    target_decoder.nb_means_normalized_decoder.masked_l.weight.data
                    ).clone()
                source_gp_weights_masked = (
                    source_decoder.nb_means_normalized_decoder.masked_l.weight.data
                    ).clone()
                gp_weights = torch.cat((target_gp_weights_masked,
                                        source_gp_weights_masked),
                                    dim=0)

            # Add decoder weights of addon gps
            if (gp_type != "masked") & (self.n_addon_gp_ > 0):
                target_gp_weights_addon = (
                    target_decoder.nb_means_normalized_decoder.addon_l.weight.data
                    ).clone()
                source_gp_weights_addon = (
                    source_decoder.nb_means_normalized_decoder.addon_l.weight.data
                    ).clone()
                gp_weights_addon = torch.cat((target_gp_weights_addon,
                                              source_gp_weights_addon),
                                             dim=0)
                
            if (gp_type == "all") & (self.n_addon_gp_ > 0):
                gp_weights = torch.cat([gp_weights, gp_weights_addon], axis=1)
            elif gp_type == "addon":
                gp_weights = gp_weights_addon

            # Only keep omics features in mask
            if only_masked_features:
                mask_idx = getattr(self, "features_idx_dict_")[
                    f"masked_{modality}_idx"]
                gp_weights = gp_weights[mask_idx, :]
            
            # Append current modality to output list
            gp_weights_all_modalities.append(gp_weights)
        return gp_weights_all_modalities
 
    @torch.no_grad()
    def get_active_gp_mask(
            self,
            abs_gp_weights_agg_mode: Literal["sum",
                                             "nzmeans",
                                             "sum+nzmeans",
                                             "nzmedians",
                                             "sum+nzmedians"]="sum+nzmeans",
            return_gp_weights: bool=False,
            normalize_gp_weights_with_features_scale_factors: bool=False,
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get a mask of active gene programs based on the rna decoder gene weights
        of gene programs. Active gene programs are gene programs whose absolute
        gene weights aggregated over all genes are greater than
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
            If ´True´, in addition return the rna decoder gene weights of the
            active gene programs.

        Returns
        ----------
        active_gp_mask:
            Boolean tensor of gene programs which contains `True` for active
            gene programs and `False` for inactive gene programs.
        active_gp_weights:
            Tensor containing the rna decoder gene weights of active gene
            programs.
        """
        device = next(self.parameters()).device
        
        active_gp_mask = torch.zeros(self.n_prior_gp_ + self.n_addon_gp_,
                                     dtype=torch.bool,
                                     device=device)

        if self.active_gp_type_ == "mixed":
            gp_types = ["all"]
        elif (self.n_addon_gp_ > 0):
            gp_types = ["masked", "addon"]
        else:
            gp_types = ["masked"]

        for gp_type in gp_types:
            gp_weights = self.get_gp_weights(only_masked_features=False,
                                             gp_type=gp_type)[0]
            
            # Get index of gps based on ´gp_type´
            if gp_type == "masked":
                gp_idx = slice(None, self.n_prior_gp_)
            elif gp_type == "addon":
                gp_idx = slice(self.n_prior_gp_, None)
            elif gp_type == "all":
                gp_idx = slice(None, None)
            
            # Normalize gp weights with features scale factors
            if normalize_gp_weights_with_features_scale_factors:
                gp_weights_normalized = (gp_weights /
                                         self.features_scale_factors_[:, None].to(device))
            else:
                gp_weights_normalized = gp_weights
            
            # Normalize gp weights with running mean absolute gp scores
            gp_weights_normalized = (self.running_mean_abs_mu[gp_idx] *
                                     gp_weights_normalized)

            # Aggregate absolute normalized gp weights based on
            # ´abs_gp_weights_agg_mode´ and calculate thresholds of aggregated
            # absolute normalized gp weights and get active gp mask and (optionally)
            # active gp weights
            abs_gp_weights_sums = gp_weights_normalized.norm(p=1, dim=0)
            if abs_gp_weights_agg_mode in ["sum", "sum+nzmeans", "sum+nzmedians"]:
                max_abs_gp_weights_sum = max(abs_gp_weights_sums)
                min_abs_gp_weights_sum_thresh = (self.active_gp_thresh_ratio_ * 
                                                max_abs_gp_weights_sum)
                active_gp_mask[gp_idx] = active_gp_mask[gp_idx] | (
                    abs_gp_weights_sums >= min_abs_gp_weights_sum_thresh)
            
            if abs_gp_weights_agg_mode in ["nzmeans", "sum+nzmeans"]:
                abs_gp_weights_nzmeans = (
                    abs_gp_weights_sums / 
                    torch.count_nonzero(gp_weights_normalized, dim=0))
                abs_gp_weights_nzmeans = torch.nan_to_num(abs_gp_weights_nzmeans)
                max_abs_gp_weights_nzmean = max(abs_gp_weights_nzmeans)
                min_abs_gp_weights_nzmean_thresh = (self.active_gp_thresh_ratio_ *
                                                    max_abs_gp_weights_nzmean)
                if abs_gp_weights_agg_mode == "nzmeans":
                    active_gp_mask[gp_idx] = active_gp_mask[gp_idx] | (
                        abs_gp_weights_nzmeans >= 
                        min_abs_gp_weights_nzmean_thresh)
                elif abs_gp_weights_agg_mode == "sum+nzmeans":
                    # Combine active gp mask
                    active_gp_mask[gp_idx] = active_gp_mask[gp_idx] | (
                        abs_gp_weights_nzmeans >=
                        min_abs_gp_weights_nzmean_thresh)
            if abs_gp_weights_agg_mode in ["nzmedians", "sum+nzmedians"]:
                zero_mask = (gp_weights_normalized == 0)
                abs_gp_weights_normalized_with_nan = torch.where(zero_mask, torch.tensor(float('nan')), torch.abs(gp_weights_normalized))
                abs_gp_weights_nzmedians = torch.nanmedian(abs_gp_weights_normalized_with_nan, dim=0).values
                abs_gp_weights_nzmedians = torch.nan_to_num(abs_gp_weights_nzmedians)
                max_abs_gp_weights_nzmedian = torch.max(abs_gp_weights_nzmedians)
                min_abs_gp_weights_nzmedian_thresh = (0.01 *
                                                      max_abs_gp_weights_nzmedian)
                if abs_gp_weights_agg_mode == "nzmedians":
                    active_gp_mask[gp_idx] = active_gp_mask[gp_idx] | (
                        abs_gp_weights_nzmedians >= 
                        min_abs_gp_weights_nzmedian_thresh)
                elif abs_gp_weights_agg_mode == "sum+nzmedians":
                    # Combine active gp mask
                    active_gp_mask[gp_idx] = active_gp_mask[gp_idx] | (
                        abs_gp_weights_nzmedians >=
                        min_abs_gp_weights_nzmedian_thresh)
        if return_gp_weights:
            active_gp_weights = gp_weights[:, active_gp_mask]
            return active_gp_mask, active_gp_weights
        else:
            return active_gp_mask

    def log_module_hyperparams_to_mlflow(
            self,
            excluded_attr: list=["features_idx_dict_",
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
                        try:
                            mlflow.log_param(f"cat_covariate{i}_cats",
                                             attr_value[i])
                        except MlflowException:
                            continue
                elif attr == "cat_covariates_label_encoders_":
                    for i in range(len(attr_value)):
                        try:
                            mlflow.log_param(f"cat_covariate{i}_label_encoder",
                                             attr_value[i])
                        except MlflowException:
                            continue
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
            Latent space features (dim: n_obs x n_gps or n_obs x n_active_gps).
        mu:
            Expected values of the latent posteriors (dim: n_obs x n_gps or
            n_obs x n_active_gps).
        std:
            Standard deviations of the latent posteriors (dim: n_obs x 
            n_gps or n_obs x n_active_gps).
        """
        # Logarithmitize omics feature vector if done during training
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
            std = torch.exp(logstd)
            return mu, std
        else:
            z = self.reparameterize(mu, logstd)
            return z

    def get_omics_decoder_outputs(
            self,
            node_batch: Data,
            only_active_gps: bool=True,
            ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Decode latent features ´z´ to return 

        Parameters
        ----------
        node_batch:
            PyG Data object containing a node-level batch.
        only_active_gps:
            If ´True´, return only the latent representation of active gps.
        cat_covariates_embed:
            Tensor containing the categorical covariates embedding
            (dim: n_obs x sum(cat_covariates_embeds_num)).

        Encode input features ´x´ and ´edge_index´ into the latent distribution
        parameters and decode them to return the parameters of the distribution
        used for omics reconstruction.
           
        Parameters
        ----------

        return_mu_std:
            If ´True´, return ´mu´ and ´std´ instead of latent features ´z´.

        Returns
        -------
        z:
            Latent space features (dim: n_obs x n_gps or n_obs x n_active_gps).
        mu:
            Expected values of the latent posteriors (dim: n_obs x n_gps or
            n_obs x n_active_gps).
        std:
            Standard deviations of the latent posteriors (dim: n_obs x 
            n_gps or n_obs x n_active_gps).
        """
        x = node_batch.x # dim: n_obs x n_omics_features
        edge_index = node_batch.edge_index # dim: 2 x n_edges (incl. all edges
                                           # of sampled graph)
        
        batch_idx = slice(None, node_batch.batch_size)
        
        # Logarithmitize omics feature vector if done during training
        if self.log_variational_:
            x_enc = torch.log(1 + x)
        else:
            x_enc = x # dim: n_obs x n_omics_features
            
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
        mu = encoder_outputs[0][batch_idx, :]
        logstd = encoder_outputs[1][batch_idx, :]
        
        z = self.reparameterize(mu, logstd)

        if only_active_gps:
            active_gp_mask = self.get_active_gp_mask()
            
            # Set gp scores of inactive gene programs to 0 to not affect 
            # graph decoder
            z[:, ~active_gp_mask] = 0

        output = {}
        output["node_labels"] = {}

        # Get rna and atac part from omics feature vector
        x_atac = x[:, self.n_output_genes_:]
        x = x[:, :self.n_output_genes_]

        # Compute aggregated neighborhood rna feature vector
        rna_node_label_aggregator_output = self.rna_node_label_aggregator(
                x=x,
                edge_index=edge_index,
                return_agg_weights=False)
        x_neighbors = rna_node_label_aggregator_output[0]

        # Retrieve rna node labels and only keep nodes in current node batch
        # and reconstructed features
        assert x.size(1) == self.n_output_genes_
        assert x_neighbors.size(1) == self.n_output_genes_

        # This may include genes that are excluded from reconstruction due to
        # turn off of inactive gene programs
        output["node_labels"]["target_rna"] = x[batch_idx]
        output["node_labels"]["source_rna"] = x_neighbors[batch_idx]

        # Use observed library size as scaling factor for the negative
        # binomial means of the rna distribution
        target_rna_library_size = output["node_labels"]["target_rna"].sum(
            1).unsqueeze(1)
        source_rna_library_size = output["node_labels"]["source_rna"].sum(
            1).unsqueeze(1)
        target_rna_log_library_size = torch.log(target_rna_library_size)
        source_rna_log_library_size = torch.log(source_rna_library_size)

        # Get gene expression reconstruction distribution parameters for
        # reconstructed genes
        output["target_rna_nb_means"] = self.target_rna_decoder(
            z=z,
            log_library_size=target_rna_log_library_size,
            cat_covariates_embed=(
                cat_covariates_embed[batch_idx] if
                (cat_covariates_embed is not None) &
                ("gene_expr_decoder" in
                 self.cat_covariates_embeds_injection_)
                 else None))
        output["source_rna_nb_means"] = self.source_rna_decoder(
            z=z,
            log_library_size=source_rna_log_library_size,
            cat_covariates_embed=(
            cat_covariates_embed[batch_idx] if
            (cat_covariates_embed is not None) &
            ("gene_expr_decoder" in
             self.cat_covariates_embeds_injection_)
             else None))

        if "atac" in self.modalities_:
            # Compute aggregated neighborhood atac feature vector
            atac_node_label_aggregator_output = (
                self.atac_node_label_aggregator(
                    x=x_atac,
                    edge_index=edge_index,
                    return_agg_weights=False))
            x_neighbors_atac = atac_node_label_aggregator_output[0]

            # Retrieve node labels and only keep nodes in current node batch
            # and reconstructed features
            assert x_atac.size(1) == self.n_output_peaks_
            assert x_neighbors_atac.size(1) == self.n_output_peaks_
            output["node_labels"]["target_atac"] = x_atac[batch_idx][
                :, self.features_idx_dict_["target_reconstructed_atac_idx"]]  
            output["node_labels"]["source_atac"] = x_neighbors_atac[batch_idx][
                :, self.features_idx_dict_["source_reconstructed_atac_idx"]]

            # Use observed library size as scaling factor for the negative
            # binomial means of the atac distribution
            target_atac_library_size = output["node_labels"][
                "target_atac"].sum(1).unsqueeze(1)
            source_atac_library_size = output["node_labels"][
                "source_atac"].sum(1).unsqueeze(1)
            target_atac_log_library_size = torch.log(
                target_atac_library_size)
            source_atac_log_library_size = torch.log(
                source_atac_library_size)

            # Get chromatin accessibility reconstruction distribution
            # parameters for reconstructed peaks
            output["target_atac_nb_means"] = self.target_atac_decoder(
                z=z,
                log_library_size=target_atac_log_library_size,
                dynamic_mask=self.target_atac_dynamic_decoder_mask,
                cat_covariates_embed=(
                    cat_covariates_embed[batch_idx] if
                    (cat_covariates_embed is not None) & 
                    ("chrom_access_decoder" in
                     self.cat_covariates_embeds_injection_) else
                    None))[
                :, self.features_idx_dict_["target_reconstructed_atac_idx"]]
            output["source_atac_nb_means"] = self.source_atac_decoder(
                z=z,
                log_library_size=source_atac_log_library_size,
                dynamic_mask=self.source_atac_dynamic_decoder_mask,
                cat_covariates_embed=(
                    cat_covariates_embed[batch_idx] if
                    (cat_covariates_embed is not None) &
                    ("chrom_access_decoder" in
                     self.cat_covariates_embeds_injection_) else
                    None))[
                :, self.features_idx_dict_["source_reconstructed_atac_idx"]]
        return output

    
    
    
                    
