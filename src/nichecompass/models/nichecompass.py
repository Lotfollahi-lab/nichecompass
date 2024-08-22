"""
This module contains the NicheCompass model. Different analysis capabilities are
integrated directly into the model API for easy use.
"""

from typing import Literal, List, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from anndata import AnnData
from scipy.special import erfc

from nichecompass.data import (initialize_dataloaders,
                               prepare_data)
from nichecompass.modules import VGPGAE
from nichecompass.train import Trainer
from .basemodelmixin import BaseModelMixin


class NicheCompass(BaseModelMixin):
    """
    NicheCompass model class.

    Parameters
    ----------
    adata:
        AnnData object with gene expression raw counts stored in
        ´adata.layers[counts_key]´ or ´adata.X´, depending on ´counts_key´,
        sparse adjacency matrix stored in ´adata.obsp[adj_key]´, gene program
        names stored in ´adata.uns[gp_names_key]´, and binary gene program
        targets and sources masks stored in ´adata.varm[gp_targets_mask_key]´
        and ´adata.varm[gp_sources_mask_key]´ respectively.
    adata_atac:
        Optional AnnData object with paired spatial chromatin accessibility
        raw counts stored in ´adata_atac.X´, and sparse boolean chromatin
        accessibility targets and sources masks stored in
        ´adata_atac.varm[ca_targets_mask_key]´ and
        ´adata_atac.varm[ca_sources_mask_key]´ respectively.
    counts_key:
        Key under which the gene expression raw counts are stored in
        ´adata.layer´. If ´None´, uses ´adata.X´ as counts. 
    adj_key:
        Key under which the sparse adjacency matrix is stored in ´adata.obsp´.
    gp_names_key:
        Key under which the gene program names are stored in ´adata.uns´.
    active_gp_names_key:
        Key under which the active gene program names will be stored in 
        ´adata.uns´.
    gp_targets_mask_key:
        Key under which the gene program targets mask is stored in ´adata.varm´.
    gp_sources_mask_key:
        Key under which the gene program sources mask is stored in ´adata.varm´.
    ca_targets_mask_key:
        Key under which the chromatin accessibility targets mask is stored in
        ´adata_atac.varm´.
    ca_sources_mask_key:
        Key under which the chromatin accessibility sources mask is stored in
        ´adata_atac.varm´.
    latent_key:
        Key under which the latent / gene program representation of active gene
        programs will be stored in ´adata.obsm´ after model training.
    cat_covariates_keys:
        Keys under which the categorical covariates are stored in ´adata.obs´.
    cat_covariates_no_edges:
        List of booleans that indicate whether there can be edges between
        different categories of the categorical covariates. If this is ´True´
        for a specific categorical covariate, this covariate will be excluded
        from the edge reconstruction loss.
    cat_covariates_embeds_keys:
        Keys under which the categorical covariates embeddings will be stored in
        ´adata.uns´.
    cat_covariates_embeds_injection:
        List of VGPGAE modules in which the categorical covariates embeddings
        are injected.
    genes_idx_key:
        Key in ´adata.uns´ where the index of a concatenated vector of target
        and source genes that are in the gene program masks are stored.    
    target_genes_idx_key:
        Key in ´adata.uns´ where the index of target genes that are in the gene
        program masks are stored.
    source_genes_idx_key:
        Key in ´adata.uns´ where the index of source genes that are in the gene
        program masks are stored.
    peaks_idx_key:
        Key in ´adata_atac.uns´ where the index of a concatenated vector of
        target and source peaks that are in the chromatin accessibility masks
        are stored.          
    target_peaks_idx_key:
        Key in ´adata_atac.uns´ where the index of target peaks that are in the
        chromatin accessibility masks are stored.
    source_peaks_idx_key:
        Key in ´adata_atac.uns´ where the index of source peaks that are in the
        chromatin accessibility masks are stored.
    gene_peaks_mask_key:
        Key in ´adata.varm´ where the gene peak mapping mask is stored.    
    recon_adj_key:
        Key in ´adata.obsp´ where the reconstructed adjacency matrix edge
        probabilities will be stored.
    agg_weights_key:
        Key in ´adata.obsp´ where the aggregation weights of the node label
        aggregator will be stored.
    include_edge_recon_loss:
        If `True`, includes the edge reconstruction loss in the backpropagation.
    include_gene_expr_recon_loss:
        If `True`, includes the gene expression reconstruction loss in the
        backpropagation.
    include_chrom_access_recon_loss:
        If `True`, includes the chromatin accessibility reconstruction loss in
        the backpropagation.
    include_cat_covariates_contrastive_loss:
        If `True`, includes the categorical covariates contrastive loss in the
        backpropagation.
    gene_expr_recon_dist:
        The distribution used for gene expression reconstruction. If `nb`, uses
        a negative binomial distribution. If `zinb`, uses a zero-inflated
        negative binomial distribution.
    log_variational:
        If ´True´, transforms x by log(x+1) prior to encoding for numerical 
        stability (not for normalization).
    node_label_method:
        Node label method that will be used for omics reconstruction. If ´self´,
        uses only the input features of the node itself as node labels for omics
        reconstruction. If ´one-hop-sum´, uses a concatenation of the node's
        input features with the sum of the input features of all nodes in the
        node's one-hop neighborhood. If ´one-hop-norm´, uses a concatenation of
        the node`s input features with the node's one-hop neighbors input
        features normalized as per Kipf, T. N. & Welling, M. Semi-Supervised
        Classification with Graph Convolutional Networks. arXiv [cs.LG] (2016).
        If ´one-hop-attention´, uses a concatenation of the node`s input
        features with the node's one-hop neighbors input features weighted by an
        attention mechanism.
    active_gp_thresh_ratio:
        Ratio that determines which gene programs are considered active and are
        used in the latent representation after model training. All inactive
        gene programs will be dropped during model training after a determined
        number of epochs. Aggregations of the absolute values of the gene
        weights of the gene expression decoder per gene program are calculated.
        The maximum value, i.e. the value of the gene program with the highest
        aggregated value will be used as a benchmark and all gene programs whose
        aggregated value is smaller than ´active_gp_thresh_ratio´ times this
        maximum value will be set to inactive. If ´==0´, all gene programs will
        be considered active. More information can be found in 
        ´self.model.get_active_gp_mask()´.
    active_gp_type:
        Type to determine active gene programs. Can be ´mixed´, in which case
        active gene programs are determined across prior and add-on gene programs
        jointly or ´separate´ in which case they are determined separately for
        prior adn add-on gene programs.
    n_fc_layers_encoder:
        Number of fully connected layers in the encoder before message passing
        layers.
    n_layers_encoder:
        Number of message passing layers in the encoder.
    n_hidden_encoder:
        Number of nodes in the encoder hidden layers. If ´None´ is determined
        automatically based on the number of input genes and gene programs.
    conv_layer_encoder:
        Convolutional layer used as GNN in the encoder.
    encoder_n_attention_heads:
        Only relevant if ´conv_layer_encoder == gatv2conv´. Number of attention
        heads used in the GNN layers of the encoder.
    encoder_use_bn:
        If ´True´, uses a batch normalization layer at the end of the encoder to
        normalize ´mu´.
    dropout_rate_encoder:
        Probability that nodes will be dropped in the encoder during training.
    dropout_rate_graph_decoder:
        Probability that nodes will be dropped in the graph decoder during 
        training.
    cat_covariates_cats:
        List of category lists for each categorical covariate to get the right
        encoding when used after reloading.
    n_addon_gp:
        Number of addon gene programs (i.e. gene programs that are not included
        in masks but can be learned de novo).
    cat_covariates_embeds_nums:
        List of number of embedding nodes for all categorical covariates.
    use_cuda_if_available:
        If `True`, use cuda if available.
    seed:
        Random seed to get reproducible results.
    kwargs:
        NicheCompass kwargs (to support legacy versions).
    """
    def __init__(self,
                 adata: AnnData,
                 adata_atac: Optional[AnnData]=None,
                 counts_key: Optional[str]="counts",
                 adj_key: str="spatial_connectivities",
                 gp_names_key: str="nichecompass_gp_names",
                 active_gp_names_key: str="nichecompass_active_gp_names",
                 gp_targets_mask_key: str="nichecompass_gp_targets",
                 gp_targets_categories_mask_key: str="nichecompass_gp_targets_categories",
                 targets_categories_label_encoder_key: str="nichecompass_targets_categories_label_encoder",
                 gp_sources_mask_key: str="nichecompass_gp_sources",
                 gp_sources_categories_mask_key: str="nichecompass_gp_sources_categories",
                 sources_categories_label_encoder_key: str="nichecompass_sources_categories_label_encoder",
                 ca_targets_mask_key: Optional[str]="nichecompass_ca_targets",
                 ca_sources_mask_key: Optional[str]="nichecompass_ca_sources",
                 latent_key: str="nichecompass_latent",
                 cat_covariates_embeds_keys: Optional[List[str]]=None,
                 cat_covariates_embeds_injection: Optional[List[
                     Literal["encoder",
                             "gene_expr_decoder",
                             "chrom_access_decoder"]]]=["gene_expr_decoder",
                                                        "chrom_access_decoder"],
                 cat_covariates_keys: Optional[List[str]]=None,
                 cat_covariates_no_edges: Optional[List[bool]]=None,
                 genes_idx_key: str="nichecompass_genes_idx",
                 target_genes_idx_key: str="nichecompass_target_genes_idx",
                 source_genes_idx_key: str="nichecompass_source_genes_idx",
                 peaks_idx_key: str="nichecompass_peaks_idx",
                 target_peaks_idx_key: str="nichecompass_target_peaks_idx",
                 source_peaks_idx_key: str="nichecompass_source_peaks_idx",
                 gene_peaks_mask_key: str="nichecompass_gene_peaks",
                 recon_adj_key: Optional[str]="nichecompass_recon_connectivities",
                 agg_weights_key: Optional[str]="nichecompass_agg_weights",
                 include_edge_recon_loss: bool=True,
                 include_gene_expr_recon_loss: bool=True,
                 include_chrom_access_recon_loss: Optional[bool]=True,
                 include_cat_covariates_contrastive_loss: bool=False,
                 gene_expr_recon_dist: Literal["nb"]="nb",
                 log_variational: bool=True,
                 node_label_method: Literal[
                    "one-hop-sum",
                    "one-hop-norm",
                    "one-hop-attention"]="one-hop-norm",
                 active_gp_thresh_ratio: float=0.01,
                 active_gp_type: Literal["mixed", "separate"]="separate",
                 n_fc_layers_encoder: int=1,
                 n_layers_encoder: int=1,
                 n_hidden_encoder: Optional[int]=None,
                 conv_layer_encoder: Literal["gcnconv", "gatv2conv"]="gatv2conv",
                 encoder_n_attention_heads: Optional[int]=4,
                 encoder_use_bn: bool=False,
                 dropout_rate_encoder: float=0.,
                 dropout_rate_graph_decoder: float=0.,
                 cat_covariates_cats: Optional[List[List]]=None,
                 n_addon_gp: int=100,
                 cat_covariates_embeds_nums: Optional[List[int]]=None,
                 include_edge_kl_loss: bool=True,
                 use_cuda_if_available: bool=True,
                 seed: int=0,
                 **kwargs):
        self.adata = adata
        self.adata_atac = adata_atac
        self.counts_key_ = counts_key
        self.adj_key_ = adj_key
        self.gp_names_key_ = gp_names_key
        self.active_gp_names_key_ = active_gp_names_key
        self.gp_targets_mask_key_ = gp_targets_mask_key
        self.gp_targets_categories_mask_key_ = gp_targets_categories_mask_key
        self.targets_categories_label_encoder_key_ = (
            targets_categories_label_encoder_key)
        self.gp_sources_mask_key_ = gp_sources_mask_key
        self.gp_sources_categories_mask_key_ = gp_sources_categories_mask_key
        self.sources_categories_label_encoder_key_ = (
            sources_categories_label_encoder_key)
        self.ca_targets_mask_key_ = ca_targets_mask_key
        self.ca_sources_mask_key_ = ca_sources_mask_key
        self.latent_key_ = latent_key
        self.cat_covariates_embeds_keys_ = cat_covariates_embeds_keys
        self.cat_covariates_embeds_injection_ = cat_covariates_embeds_injection
        self.cat_covariates_keys_ = cat_covariates_keys
        self.cat_covariates_embeds_keys_ = cat_covariates_embeds_keys
        self.genes_idx_key_ = genes_idx_key
        self.target_genes_idx_key_ = target_genes_idx_key
        self.source_genes_idx_key_ = source_genes_idx_key
        self.peaks_idx_key_ = peaks_idx_key
        self.target_peaks_idx_key_ = target_peaks_idx_key
        self.source_peaks_idx_key_ = source_peaks_idx_key
        self.gene_peaks_mask_key_ = gene_peaks_mask_key
        self.recon_adj_key_ = recon_adj_key
        self.agg_weights_key_ = agg_weights_key
        self.include_edge_recon_loss_ = include_edge_recon_loss
        self.include_gene_expr_recon_loss_ = include_gene_expr_recon_loss
        self.include_chrom_access_recon_loss_ = include_chrom_access_recon_loss
        self.include_cat_covariates_contrastive_loss_ = (
            include_cat_covariates_contrastive_loss)
        self.gene_expr_recon_dist_ = gene_expr_recon_dist
        self.log_variational_ = log_variational
        self.node_label_method_ = node_label_method
        self.active_gp_thresh_ratio_ = active_gp_thresh_ratio
        self.active_gp_type_ = active_gp_type
        self.include_edge_kl_loss_ = include_edge_kl_loss
        self.seed_ = seed

        # Set seed for reproducibility
        np.random.seed(self.seed_)
        if use_cuda_if_available & torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed_)
            torch.manual_seed(self.seed_)
        else:
            torch.manual_seed(self.seed_)

        # Retrieve gene program masks
        if gp_targets_mask_key in adata.varm:
            # NOTE: dtype can be changed to bool and should be able to handle sparse
            # mask
            self.gp_targets_mask_ = torch.tensor(
                adata.varm[gp_targets_mask_key].T,
                dtype=torch.bool)
        else:
            raise ValueError("Please specify an adequate ´gp_targets_mask_key´ "
                             "for your adata object. The targets mask needs to "
                             "be stored in ´adata.varm[gp_targets_mask_key]´. "
                             " If you do not want to mask gene expression "
                             "reconstruction, you can create a mask of 1s that"
                             " allows all gene program latent nodes to "
                             "reconstruct all genes.")

        if gp_sources_mask_key in adata.varm:
            # NOTE: dtype can be changed to bool and should be able to handle
            # sparse mask
            self.gp_sources_mask_ = torch.tensor(
                adata.varm[gp_sources_mask_key].T,
                dtype=torch.bool)
                                           
        else:
            raise ValueError("Please specify an adequate "
                             "´gp_sources_mask_key´ for your adata object. "
                             "The sources mask needs to be stored in "
                             "´adata.varm[gp_sources_mask_key]´. If you do "
                             "not want to mask gene expression "
                             "reconstruction, you can create a mask of 1s "
                             " that allows all gene program latent nodes to"
                             " reconstruct all genes.")
            
        # Determine features scale factors
        self.features_scale_factors_ = torch.concat(
            (torch.tensor(self.adata.X.sum(0))[0],
             torch.tensor(self.adata.X.sum(0))[0]))
    
        # Retrieve chromatin accessibility masks
        if adata_atac is None:
            self.ca_targets_mask_ = None
            self.ca_sources_mask_ = None
            gene_peaks_mask = None
        else:
            gene_peaks_mask = adata.varm[gene_peaks_mask_key].tocoo()
            gene_peaks_mask = torch.sparse_coo_tensor(
                indices=[gene_peaks_mask.row, gene_peaks_mask.col],
                values=gene_peaks_mask.data,
                size=gene_peaks_mask.shape,
                dtype=torch.bool) # bool does not work with torch.mm
            if ca_targets_mask_key in adata_atac.varm:
                ca_targets_mask = adata_atac.varm[ca_targets_mask_key].T.tocoo()
            else:
                raise ValueError("Please specify an adequate "
                                 "´ca_targets_mask_key´ for your adata_atac "
                                 "object. The targets mask needs to be stored "
                                 "in ´adata_atac.varm[ca_targets_mask_key]´. If"
                                 " you do not want to mask chromatin "
                                 " accessibility reconstruction, you can create"
                                 " a mask of 1s that allows all gene program "
                                 "latent nodes to reconstruct all peaks.")
            self.ca_targets_mask_ = torch.sparse_coo_tensor(
                indices=[ca_targets_mask.row, ca_targets_mask.col],
                values=ca_targets_mask.data,
                size=ca_targets_mask.shape,
                dtype=torch.bool).to_dense() # for now
            if ca_sources_mask_key in adata_atac.varm:
                ca_sources_mask = adata_atac.varm[
                    ca_sources_mask_key].T.tocoo()
                self.ca_sources_mask_ = torch.sparse_coo_tensor(
                    indices=[ca_sources_mask.row, ca_sources_mask.col],
                    values=ca_sources_mask.data,
                    size=ca_sources_mask.shape,
                    dtype=torch.bool).to_dense() # for now
            else:
                raise ValueError("Please specify an adequate "
                                "´ca_sources_mask_key´ for your adata_atac "
                                "object. The sources mask needs to be "
                                "stored in "
                                "´adata_atac.varm[ca_sources_mask_key]´. If"
                                "you do not want to mask chromatin "
                                " accessibility reconstruction, you can "
                                "create a mask of 1s that allows all gene "
                                "program latent nodes to reconstruct all "
                                "peaks.")

        # Retrieve index of genes in gp mask and index of genes not in gp mask
        self.features_idx_dict_ = {}
        self.features_idx_dict_["masked_rna_idx"] = adata.uns[
            genes_idx_key]
        self.features_idx_dict_["unmasked_rna_idx"] = [
            i for i in range(len(adata.var_names))
            if i not in self.features_idx_dict_["masked_rna_idx"]]
        self.features_idx_dict_["target_masked_rna_idx"] = list(
            adata.uns[target_genes_idx_key])
        self.features_idx_dict_["target_unmasked_rna_idx"] = [
            i for i in range(len(adata.var_names))
            if i not in self.features_idx_dict_["target_masked_rna_idx"]]
        self.features_idx_dict_["source_masked_rna_idx"] = list(
            adata.uns[source_genes_idx_key])
        self.features_idx_dict_["source_unmasked_rna_idx"] = [
            i for i in range(len(adata.var_names))
            if i not in self.features_idx_dict_["source_masked_rna_idx"]]
        
        # Retrieve index of peaks in ca mask and index of peaks not in ca mask
        if adata_atac is not None:
            self.peaks_idx_ = adata_atac.uns[peaks_idx_key]
            self.target_peaks_idx_ = adata_atac.uns[target_peaks_idx_key]
            self.source_peaks_idx_ = adata_atac.uns[source_peaks_idx_key]
            
            self.features_idx_dict_["masked_atac_idx"] = adata_atac.uns[
                peaks_idx_key]
            self.features_idx_dict_["unmasked_atac_idx"] = [
                i for i in range(len(adata_atac.var_names))
                if i not in self.features_idx_dict_["masked_atac_idx"]]
            self.features_idx_dict_["target_masked_atac_idx"] = list(
                adata_atac.uns[target_peaks_idx_key])
            self.features_idx_dict_["target_unmasked_atac_idx"] = [
                i for i in range(len(adata_atac.var_names))
                if i not in self.features_idx_dict_["target_masked_atac_idx"]]
            self.features_idx_dict_["source_masked_atac_idx"] = list(
                adata_atac.uns[source_peaks_idx_key])
            self.features_idx_dict_["source_unmasked_atac_idx"] = [
                i for i in range(len(adata_atac.var_names))
                if i not in self.features_idx_dict_["source_masked_atac_idx"]]

        # Determine VGPGAE inputs
        self.n_input_ = adata.n_vars
        self.n_output_genes_ = adata.n_vars
        if adata_atac is not None:
            self.modalities_ = ["rna", "atac"]
            if not np.all(adata.obs.index == adata_atac.obs.index):
                raise ValueError("Please make sure that 'adata' and "
                                 "'adata_atac' contain the same observations in"
                                 " the same order.")
            # Peaks are concatenated to genes in input
            self.n_input_ += adata_atac.n_vars
            self.n_output_peaks_ = adata_atac.n_vars
        else:
            self.modalities_ = ["rna"]
            self.n_output_peaks_ = 0
        self.n_fc_layers_encoder_ = n_fc_layers_encoder
        self.n_layers_encoder_ = n_layers_encoder
        self.conv_layer_encoder_ = conv_layer_encoder
        if conv_layer_encoder == "gatv2conv":
            self.encoder_n_attention_heads_ = encoder_n_attention_heads
        else:
            self.encoder_n_attention_heads_ = 0
        self.encoder_use_bn_ = encoder_use_bn
        self.dropout_rate_encoder_ = dropout_rate_encoder
        self.dropout_rate_graph_decoder_ = dropout_rate_graph_decoder
        self.n_prior_gp_ = len(self.gp_targets_mask_)
        self.n_addon_gp_ = n_addon_gp
        
        if n_addon_gp > 0:
            # Add add-on gps to adata
            gp_list = list(self.adata.uns[self.gp_names_key_])
            for i in range(n_addon_gp):
                if f"Add-on_{i}_GP" not in gp_list:
                    gp_list.append(f"Add-on_{i}_GP")
            self.adata.uns[self.gp_names_key_] = np.array(gp_list)
        else:
            # Remove add-on gps from adata
            for gp_name in list(adata.uns[gp_names_key]):
                if "Add-on" in gp_name:
                    self.adata.uns[gp_names_key] = np.delete(
                        self.adata.uns[gp_names_key],
                        list(self.adata.uns[gp_names_key]).index(gp_name))

        # Retrieve categorical covariates categories
        if cat_covariates_cats is None:
            if cat_covariates_keys is not None:
                self.cat_covariates_cats_ = [
                    adata.obs[cat_covariate_key].unique().tolist() 
                    for cat_covariate_key in cat_covariates_keys]
            else:
                self.cat_covariates_cats_ = []
        else:
            self.cat_covariates_cats_ = cat_covariates_cats
        
        # Define dimensionality of categorical covariates embeddings as
        # number of categories of each categorical covariate respectively
        # if not provided explicitly
        if cat_covariates_embeds_nums is None:
            cat_covariates_embeds_nums = []
            for cat_covariate_cats in self.cat_covariates_cats_:
                cat_covariates_embeds_nums.append(len(cat_covariate_cats))
        self.cat_covariates_embeds_nums_ = cat_covariates_embeds_nums

        # Determine dimensionality of hidden encoder layer if not provided
        if n_hidden_encoder is None:
            if len(adata.var) > (self.n_prior_gp_ + self.n_addon_gp_):
                n_hidden_encoder = (self.n_prior_gp_ + self.n_addon_gp_)
            else:
                n_hidden_encoder = len(adata.var)
        self.n_hidden_encoder_ = n_hidden_encoder
            
        # Define categorical covariates no edges as all 'True' if not
        # explicitly provided, so that they are excluded from the edge
        # reconstruction loss
        if ((cat_covariates_no_edges is None) &
            (len(self.cat_covariates_cats_) > 0)):
            self.cat_covariates_no_edges_ = (
                [True] * len(self.cat_covariates_cats_))
        else:
            self.cat_covariates_no_edges_ = cat_covariates_no_edges
        
        # Validate counts layer key and counts values
        if counts_key is not None and counts_key not in adata.layers:
            raise ValueError("Please specify an adequate ´counts_key´. By "
                             "default the counts are assumed to be stored in "
                             "data.layers['counts'].")
        if include_gene_expr_recon_loss and log_variational:
            if counts_key is None:
                x = adata.X
            else:
                x = adata.layers[counts_key]
            if (x < 0).sum() > 0:
                raise ValueError("Please make sure that "
                                 "´adata.layers[counts_key]´ contains the"
                                 " raw counts (not log library size "
                                 "normalized) if ´include_gene_expr_recon_loss´"
                                 " is ´True´ and ´log_variational´ is ´True´. "
                                 "If you want to use log library size "
                                 " normalized counts, make sure that "
                                 "´log_variational´ is ´False´.")

        # Validate adjacency key
        if adj_key not in adata.obsp:
            raise ValueError("Please specify an adequate ´adj_key´. "
                             "By default the adjacency matrix is assumed to be "
                             "stored in adata.obsm['spatial_connectivities'].")

        # Validate gp key
        if gp_names_key not in adata.uns:
            raise ValueError("Please specify an adequate ´gp_names_key´. "
                             "By default the gene program names are assumed to "
                             "be stored in adata.uns['nichecompass_gp_names'].")

        # Validate categorical covariates keys
        if cat_covariates_keys is not None:
            for cat_covariate_key in cat_covariates_keys:
                if cat_covariate_key not in adata.obs:
                    raise ValueError(
                        "Please specify adequate ´cat_covariates_keys´. "
                        f"The key {cat_covariate_key} was not found in adata.")
        
        # Initialize model with Variational Gene Program Graph Autoencoder 
        # neural network module
        self.model = VGPGAE(
            n_input=self.n_input_,
            n_fc_layers_encoder=self.n_fc_layers_encoder_,
            n_layers_encoder=self.n_layers_encoder_,
            n_hidden_encoder=self.n_hidden_encoder_,
            n_prior_gp=self.n_prior_gp_,
            n_addon_gp=self.n_addon_gp_,
            cat_covariates_embeds_nums=self.cat_covariates_embeds_nums_,
            n_output_genes=self.n_output_genes_,
            n_output_peaks=self.n_output_peaks_,
            target_rna_decoder_mask=self.gp_targets_mask_,
            source_rna_decoder_mask=self.gp_sources_mask_,
            target_atac_decoder_mask=self.ca_targets_mask_,
            source_atac_decoder_mask=self.ca_sources_mask_,
            features_idx_dict=self.features_idx_dict_,
            features_scale_factors=self.features_scale_factors_,
            gene_peaks_mask=gene_peaks_mask,
            cat_covariates_cats=self.cat_covariates_cats_,
            cat_covariates_no_edges=self.cat_covariates_no_edges_,
            conv_layer_encoder=self.conv_layer_encoder_,
            encoder_n_attention_heads=self.encoder_n_attention_heads_,
            encoder_use_bn=self.encoder_use_bn_,
            dropout_rate_encoder=self.dropout_rate_encoder_,
            dropout_rate_graph_decoder=self.dropout_rate_graph_decoder_,
            include_edge_recon_loss=self.include_edge_recon_loss_,
            include_gene_expr_recon_loss=self.include_gene_expr_recon_loss_,
            include_chrom_access_recon_loss=self.include_chrom_access_recon_loss_,
            include_cat_covariates_contrastive_loss=self.include_cat_covariates_contrastive_loss_,
            rna_recon_loss=self.gene_expr_recon_dist_,
            node_label_method=self.node_label_method_,
            active_gp_thresh_ratio=self.active_gp_thresh_ratio_,
            active_gp_type=self.active_gp_type_,
            log_variational=self.log_variational_,
            cat_covariates_embeds_injection=self.cat_covariates_embeds_injection_,
            include_edge_kl_loss=self.include_edge_kl_loss_)

        self.is_trained_ = False

        # Store init params for saving and loading
        self.init_params_ = self._get_init_params(locals())

    def train(self,
              n_epochs: int=100,
              n_epochs_all_gps: int=25,
              n_epochs_no_edge_recon: int=0,
              n_epochs_no_cat_covariates_contrastive: int=5,
              lr: float=0.001,
              weight_decay: float=0.,
              lambda_edge_recon: Optional[float]=500000.,
              lambda_gene_expr_recon: float=300.,
              lambda_chrom_access_recon: float=100.,
              lambda_cat_covariates_contrastive: float=0.,
              contrastive_logits_pos_ratio: float=0.,
              contrastive_logits_neg_ratio: float=0.,
              lambda_group_lasso: float=0.,
              lambda_l1_masked: float=0.,
              l1_targets_categories: Optional[list]=["target_gene"],
              l1_sources_categories: Optional[list]=None,
              lambda_l1_addon: float=30.,
              edge_val_ratio: float=0.1,
              node_val_ratio: float=0.1,
              edge_batch_size: int=256,
              node_batch_size: Optional[int]=None,
              mlflow_experiment_id: Optional[str]=None,
              retrieve_cat_covariates_embeds: bool=False,
              retrieve_recon_edge_probs: bool=False,
              retrieve_agg_weights: bool=False,
              use_cuda_if_available: bool=True,
              n_sampled_neighbors: int=-1,
              latent_dtype: type=np.float64,
              **trainer_kwargs):
        """
        Train the NicheCompass model.
        
        Parameters
        ----------
        n_epochs:
            Number of epochs.
        n_epochs_all_gps:
            Number of epochs during which all gene programs are used for model
            training. After that only active gene programs are retained.
        n_epochs_no_edge_recon:
            Number of epochs during which the edge reconstruction loss is
            excluded from backpropagation for pretraining using the other loss
            components.
        n_epochs_no_cat_covariates_contrastive:
            Number of epochs during which the categorical covariates contrastive loss
            is excluded from backpropagation for pretraining using the other
            loss components.
        lr:
            Learning rate.
        weight_decay:
            Weight decay (L2 penalty).
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
        lambda_group_lasso:
            Lambda (weighting factor) for the group lasso regularization loss of
            gene programs. If ´>0´, this will enforce sparsity of gene programs.
        lambda_l1_masked:
            Lambda (weighting factor) for the L1 regularization loss of genes in
            masked gene programs. If ´>0´, this will enforce sparsity of genes
            in masked gene programs.
        l1_targets_categories:
            Gene program mask targets categories for which l1 regularization loss
            will be applied.
        l1_sources_categories:
            Gene program mask sources categories for which l1 regularization loss
            will be applied.
        lambda_l1_addon:
            Lambda (weighting factor) for the L1 regularization loss of genes in
            addon gene programs. If ´>0´, this will enforce sparsity of genes in
            addon gene programs.
        edge_val_ratio:
            Fraction of the data that is used as validation set on edge-level.
            The rest of the data will be used as training set on edge-level.
        node_val_ratio:
            Fraction of the data that is used as validation set on node-level.
            The rest of the data will be used as training set on node-level.
        edge_batch_size:
            Batch size for the edge-level dataloaders.
        node_batch_size:
            Batch size for the node-level dataloaders. If ´None´, is
            automatically determined based on ´edge_batch_size´.
        mlflow_experiment_id:
            ID of the Mlflow experiment used for tracking training parameters
            and metrics.
        retrieve_cat_covariates_embeds:
            If ´True´, retrieve the categorical covariates embeddings after
            model training is finished if multiple categorical covariates
            categories are present.
        retrieve_recon_edge_probs:
            If ´True´, retrieve the reconstructed edge probabilities after model
            training is finished.
        retrieve_agg_weights:
            If ´True´, retrieve the node label aggregation weights after model
            training is finished.
        use_cuda_if_available:
            If `True`, use cuda if available.
        n_sampled_neighbors:
            Number of neighbors that are sampled during model training from the spatial
            neighborhood graph.
        latent_dtype:
            Data type for storing the latent representations. Set to np.float16 for
            really big datasets (>1m observations).
        trainer_kwargs:
            Kwargs for the model Trainer.
        """
        self.trainer = Trainer(
            adata=self.adata,
            adata_atac=self.adata_atac,
            model=self.model,
            counts_key=self.counts_key_,
            adj_key=self.adj_key_,
            gp_targets_mask_key=self.gp_targets_mask_key_,
            gp_sources_mask_key=self.gp_sources_mask_key_,
            cat_covariates_keys=self.cat_covariates_keys_,
            edge_val_ratio=edge_val_ratio,
            node_val_ratio=node_val_ratio,
            edge_batch_size=edge_batch_size,
            node_batch_size=node_batch_size,
            use_cuda_if_available=use_cuda_if_available,
            n_sampled_neighbors=n_sampled_neighbors,
            latent_dtype=latent_dtype,
            **trainer_kwargs)
        
        if lambda_l1_masked > 0.:
            # Create mask for l1 regularization loss
            if l1_targets_categories is None:
                l1_targets_categories_encoded = list(self.adata.uns[
                    self.targets_categories_label_encoder_key_].values())
            else:
                l1_targets_categories_encoded = [
                    self.adata.uns[
                        self.targets_categories_label_encoder_key_][category]
                    for category in l1_targets_categories if category in
                    self.adata.uns[self.targets_categories_label_encoder_key_]]
            if l1_sources_categories is None:
                l1_sources_categories_encoded = list(self.adata.uns[
                    self.sources_categories_label_encoder_key_].values())
            else:
                l1_sources_categories_encoded = [
                    self.adata.uns[
                        self.sources_categories_label_encoder_key_][category]
                    for category in l1_sources_categories if category in
                    self.adata.uns[self.sources_categories_label_encoder_key_]]
            l1_targets_mask = torch.from_numpy(np.isin(
                self.adata.varm[self.gp_targets_categories_mask_key_],
                l1_targets_categories_encoded))
            l1_sources_mask = torch.from_numpy(np.isin(
                self.adata.varm[self.gp_sources_categories_mask_key_],
                l1_sources_categories_encoded))
        else:
            l1_targets_mask = None
            l1_sources_mask = None

        self.trainer.train(
            n_epochs=n_epochs,
            n_epochs_no_edge_recon=n_epochs_no_edge_recon,
            n_epochs_no_cat_covariates_contrastive=n_epochs_no_cat_covariates_contrastive,
            n_epochs_all_gps=n_epochs_all_gps,
            lr=lr,
            weight_decay=weight_decay,
            lambda_edge_recon=lambda_edge_recon,
            lambda_gene_expr_recon=lambda_gene_expr_recon,
            lambda_chrom_access_recon=lambda_chrom_access_recon,
            lambda_cat_covariates_contrastive=lambda_cat_covariates_contrastive,
            contrastive_logits_pos_ratio=contrastive_logits_pos_ratio,
            contrastive_logits_neg_ratio=contrastive_logits_neg_ratio,
            lambda_group_lasso=lambda_group_lasso,
            lambda_l1_masked=lambda_l1_masked,
            l1_targets_mask=l1_targets_mask,
            l1_sources_mask=l1_sources_mask,
            lambda_l1_addon=lambda_l1_addon,
            mlflow_experiment_id=mlflow_experiment_id)
        
        self.node_batch_size_ = self.trainer.node_batch_size_
        
        self.is_trained_ = True
        self.model.eval()

        self.adata.obsm[self.latent_key_], _ = self.get_latent_representation(
           adata=self.adata,
           counts_key=self.counts_key_,
           adj_key=self.adj_key_,
           cat_covariates_keys=self.cat_covariates_keys_,
           only_active_gps=True,
           return_mu_std=True,
           node_batch_size=self.node_batch_size_,
           dtype=latent_dtype)

        self.adata.uns[self.active_gp_names_key_] = self.get_active_gps()

        if ((len(self.cat_covariates_cats_) > 0) &
            retrieve_cat_covariates_embeds):
            for cat_covariates_embed_key, cat_covariate_embed in zip(
                self.cat_covariates_embeds_keys_,
                self.get_cat_covariates_embeddings()):
                self.adata.uns[cat_covariates_embed_key] = cat_covariate_embed

        if retrieve_recon_edge_probs:
            self.adata.obsp[self.recon_adj_key_] = self.get_recon_edge_probs()

        if retrieve_agg_weights:
            self.adata.obsp[self.agg_weights_key_] = (
                self.get_neighbor_importances(
                    node_batch_size=self.node_batch_size_))

        if mlflow_experiment_id is not None:
            mlflow.log_metric("n_active_gps",
                              len(self.adata.uns[self.active_gp_names_key_]))

    def run_differential_gp_tests(
            self,
            cat_key: str,
            selected_cats: Optional[Union[str, list]]=None,
            comparison_cats: Union[str, list]="rest",
            selected_gps: Optional[Union[str, list]]=None,
            n_sample: int=10000,
            log_bayes_factor_thresh: float=2.3,
            key_added: str="nichecompass_differential_gp_test_results",
            seed: int=0,
            adata: Optional[AnnData]=None) -> list:
        """
        Run differential gene program tests by comparing gene program / latent
        scores between a category and specified comparison categories for all
        categories in ´selected_cats´ (by default all categories in
        ´adata.obs[cat_key]´). Enriched category gene programs are determined
        through the log Bayes Factor between the hypothesis h0 that the
        (normalized) gene program / latent scores of observations of the
        category under consideration (z0) are higher than the (normalized) gene
        program / latent scores of observations of the comparison categories
        (z1) versus the alternative hypothesis h1 that the (normalized) gene
        program / latent scores of observations of the comparison categories
        (z1) are higher or equal to the (normalized) gene program / latent
        scores of observations of the category under consideration (z0). The
        results of the differential tests including the log Bayes Factors for
        enriched category gene programs are stored in a pandas DataFrame under
        ´adata.uns[key_added]´. The DataFrame also stores p_h0, the probability
        that z0 > z1 and p_h1, the probability that z1 >= z0. The rows are
        ordered by the log Bayes Factor. In addition, the (normalized) gene
        program / latent scores of enriched gene programs across any of the
        categories are stored in ´adata.obs´.

        Parts of the implementation are adapted from Lotfollahi, M. et al.
        Biologically informed deep learning to query gene programs in
        single-cell atlases. Nat. Cell Biol. 25, 337–350 (2023);
        https://github.com/theislab/scarches/blob/master/scarches/models/expimap/expimap_model.py#L429
        (24.11.2022).

        Parameters
        ----------
        cat_key:
            Key under which the categories and comparison categories are stored
            in ´adata.obs´.
        selected_cats:
            List of category labels for which differential tests will be run. If
            ´None´, uses all category labels from ´adata.obs[cat_key]´.
        comparison_cats:
            Categories used as comparison group. If ´rest´, all categories other
            than the category under consideration are used as comparison group.
        selected_gps:
            List of gene program names for which differential tests will be run.
            If ´None´, uses all active gene programs.
        n_sample:
            Number of observations to be drawn from the category and comparison
            categories for the log Bayes Factor computation.
        log_bayes_factor_thresh:
            Log bayes factor threshold. Category gene programs with a higher
            absolute score than this threshold are considered enriched.
        key_added:
            Key under which the test results pandas DataFrame is stored in
            ´adata.uns´.
        seed:
            Random seed for reproducible sampling.
        adata:
            AnnData object to be used. If ´None´, uses the adata object stored
            in the model instance.

        Returns
        ----------
        enriched_gps:
            Names of enriched gene programs across all categories (duplicate
            gene programs that appear for multiple catgories are only considered
            once).
        """
        self._check_if_trained(warn=True)

        np.random.seed(seed)

        if adata is None:
            adata = self.adata

        active_gps = list(adata.uns[self.active_gp_names_key_])

        # Get selected gps
        if selected_gps is None:
            selected_gps = active_gps
        else:
            if isinstance(selected_gps, str):
                selected_gps = [selected_gps]
            for gp in selected_gps:
                if gp not in active_gps:
                    print(f"GP '{gp}' is not an active gene program. Continuing"
                          " anyways.")

        # Get indeces and weights for selected gps
        selected_gps_idx, selected_gps_weights, chrom_access_gp_weights = self.get_gp_data(
            selected_gps=selected_gps)

        # Get gp / latent scores for selected gps
        mu, std = self.get_latent_representation(
            adata=adata,
            counts_key=self.counts_key_,
            adj_key=self.adj_key_,
            cat_covariates_keys=self.cat_covariates_keys_,
            only_active_gps=False,
            return_mu_std=True,
            node_batch_size=self.node_batch_size_)
        mu_selected_gps = mu[:, selected_gps_idx]
        std_selected_gps = std[:, selected_gps_idx]

        # Retrieve category values for each observation, as well as all existing
        # unique categories
        cat_values = adata.obs[cat_key].replace(np.nan, "NaN")
        cats = cat_values.unique()
        if selected_cats is None:
            selected_cats = cats
        elif isinstance(selected_cats, str):
            selected_cats = [selected_cats]

        # Check specified comparison categories
        if comparison_cats != "rest" and isinstance(comparison_cats, str):
            comparison_cats = [comparison_cats]
        if (comparison_cats != "rest" and not
        set(comparison_cats).issubset(cats)):
            raise ValueError("Comparison categories should be 'rest' (for "
                             "comparison with all other categories) or contain "
                             "existing categories.")

        # Run differential gp tests for all selected categories that are not
        # part of the comparison categories
        results = []
        for cat in selected_cats:
            if cat in comparison_cats:
                continue
            # Filter gp scores and normalization factors for the category under
            # consideration and comparison categories
            cat_mask = cat_values == cat
            if comparison_cats == "rest":
                comparison_cat_mask = ~cat_mask
            else:
                comparison_cat_mask = cat_values.isin(comparison_cats)          

            mu_selected_gps_cat = mu_selected_gps[cat_mask]
            std_selected_gps_cat = std_selected_gps[cat_mask]
            mu_selected_gps_comparison_cat = mu_selected_gps[comparison_cat_mask]
            std_selected_gps_comparison_cat = std_selected_gps[comparison_cat_mask]

            # Generate random samples of category and comparison categories
            # observations with equal size
            cat_idx = np.random.choice(cat_mask.sum(),
                                       n_sample)
            comparison_cat_idx = np.random.choice(comparison_cat_mask.sum(),
                                                  n_sample)
            mu_selected_gps_cat_sample = mu_selected_gps_cat[cat_idx]
            std_selected_gps_cat_sample = std_selected_gps_cat[cat_idx]
            mu_selected_gps_comparison_cat_sample = (
                mu_selected_gps_comparison_cat[comparison_cat_idx])
            std_selected_gps_comparison_cat_sample = (
                std_selected_gps_comparison_cat[comparison_cat_idx])

            # Calculate gene program log Bayes Factors for the category
            to_reduce = (
                - (mu_selected_gps_cat_sample -
                mu_selected_gps_comparison_cat_sample) /
                np.sqrt(2 * (std_selected_gps_cat_sample ** 2 +
                std_selected_gps_comparison_cat_sample ** 2)))
            to_reduce = 0.5 * erfc(to_reduce)
            p_h0 = np.mean(to_reduce, axis=0)
            p_h1 = 1.0 - p_h0
            epsilon = 1e-12
            log_bayes_factor = np.log(p_h0 + epsilon) - np.log(p_h1 + epsilon)
            zeros_mask = (
                (np.abs(mu_selected_gps_cat_sample).sum(0) == 0) | 
                (np.abs(mu_selected_gps_comparison_cat_sample).sum(0) == 0))
            p_h0[zeros_mask] = 0
            p_h1[zeros_mask] = 0
            log_bayes_factor[zeros_mask] = 0

            # Store differential gp test results
            zipped = zip(
                selected_gps,
                p_h0,
                p_h1,
                log_bayes_factor)
            cat_results = [{"category": cat,
                           "gene_program": gp,
                           "p_h0": p_h0,
                           "p_h1": p_h1,
                           "log_bayes_factor": log_bayes_factor}
                          for gp, p_h0, p_h1, log_bayes_factor in zipped]
            for result in cat_results:
                results.append(result)

        # Create test results dataframe and keep only enriched category gene
        # program pairs (log bayes factor above thresh)
        results = pd.DataFrame(results)
        results["abs_log_bayes_factor"] = np.abs(results["log_bayes_factor"])
        results = results[
            results["abs_log_bayes_factor"] > log_bayes_factor_thresh]
        results.sort_values(by="abs_log_bayes_factor",
                            ascending=False,
                            inplace=True)
        results.reset_index(drop=True, inplace=True)
        results.drop("abs_log_bayes_factor", axis=1, inplace=True)
        adata.uns[key_added] = results

        # Retrieve enriched gene programs
        enriched_gps = results["gene_program"].unique().tolist()
        enriched_gps_idx = [selected_gps.index(gp) for gp in enriched_gps]
        
        # Add gene program scores of enriched gene programs to adata
        enriched_gps_gp_scores = pd.DataFrame(
            mu_selected_gps[:, enriched_gps_idx],
            columns=enriched_gps,
            index=adata.obs.index)
        new_cols = [col for col in enriched_gps_gp_scores.columns if col not in
                    adata.obs.columns]
        if new_cols:
            adata.obs = pd.concat([adata.obs,
                                   enriched_gps_gp_scores[new_cols]], axis=1)

        return enriched_gps

    def compute_gp_gene_importances(
            self,
            selected_gp: str) -> pd.DataFrame:
        """
        Compute gene importances for the genes of a given gene program. Gene
        importances are determined by the normalized weights of the rna
        decoders.

        Parameters
        ----------
        selected_gp:
            Name of the gene program for which the gene importances should be
            retrieved.
     
        Returns
        ----------
        gp_gene_importances_df:
            Pandas DataFrame containing genes, gene weights, gene
            importances and an indicator whether the gene belongs to the
            communication source or target, stored in ´gene_entity´.
        """
        self._check_if_trained(warn=True)

        # Check if selected gene program is active
        active_gps = self.adata.uns[self.active_gp_names_key_]
        if selected_gp not in active_gps:
            print(f"GP '{selected_gp}' is not an active gene program. "
                  "Continuing anyways.")

        _, gp_gene_weights, _ = self.get_gp_data(selected_gps=selected_gp)

        # Normalize gp gene weights to get gp gene importances
        gp_gene_importances = np.where(
            np.abs(gp_gene_weights).sum(0) != 0,
            np.abs(gp_gene_weights) / np.abs(gp_gene_weights).sum(0),
            0)

        # Create result dataframe
        gp_gene_importances_df = pd.DataFrame()
        gp_gene_importances_df["gene"] = [
            gene for gene in self.adata.var_names.tolist()] * 2
        gp_gene_importances_df["gene_entity"] = (
            ["target"] * len(self.adata.var_names) +
            ["source"] * len(self.adata.var_names))
        gp_gene_importances_df["gene_weight"] = gp_gene_weights
        gp_gene_importances_df["gene_importance"] = gp_gene_importances
        gp_gene_importances_df = (gp_gene_importances_df
            [gp_gene_importances_df["gene_importance"] != 0])
        gp_gene_importances_df.sort_values(by="gene_importance",
                                           ascending=False,
                                           inplace=True)
        gp_gene_importances_df.reset_index(drop=True, inplace=True)
        return gp_gene_importances_df
    
    def compute_gp_peak_importances(
            self,
            selected_gp: str) -> pd.DataFrame:
        """
        Compute peak importances for the peaks of a given gene program. Peak
        importances are determined by the normalized weights of the atac
        decoders.

        Parameters
        ----------
        selected_gp:
            Name of the gene program for which the peak importances should be
            retrieved.
     
        Returns
        ----------
        gp_peak_importances_df:
            Pandas DataFrame containing peaks, peak weights, peak
            importances and an indicator whether the peak belongs to the
            communication source or target, stored in ´peak_entity´.
        """
        self._check_if_trained(warn=True)

        if not "atac" in self.modalities_:
            raise ValueError("The model training needs to include ATAC data, "
                             "otherwise peak importances cannot be retrieved.")

        # Check if selected gene program is active
        active_gps = self.adata.uns[self.active_gp_names_key_]
        if selected_gp not in active_gps:
            print(f"GP '{selected_gp}' is not an active gene program. "
                  "Continuing anyways.")

        _, gp_gene_weights, gp_peak_weights = self.get_gp_data(
            selected_gps=selected_gp)

        # Normalize gp peak weights to get gp peak importances
        gp_peak_importances = np.where(
            np.abs(gp_peak_weights).sum(0) != 0,
            np.abs(gp_peak_weights) / np.abs(gp_peak_weights).sum(0),
            0)

        # Create result dataframe
        gp_peak_importances_df = pd.DataFrame()
        gp_peak_importances_df["peak"] = [
            peak for peak in self.adata_atac.var_names.tolist()] * 2
        gp_peak_importances_df["peak_entity"] = (
            ["target"] * len(self.adata_atac.var_names) +
            ["source"] * len(self.adata_atac.var_names))
        gp_peak_importances_df["peak_weight"] = gp_peak_weights
        gp_peak_importances_df["peak_importance"] = gp_peak_importances
        gp_peak_importances_df = (gp_peak_importances_df
            [gp_peak_importances_df["peak_importance"] != 0])
        gp_peak_importances_df.sort_values(by="peak_importance",
                                           ascending=False,
                                           inplace=True)
        gp_peak_importances_df.reset_index(drop=True, inplace=True)
        return gp_peak_importances_df

    def get_gp_data(self,
                    selected_gps: Optional[Union[str, list]]=None,
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the index of selected gene programs as well as their omics decoder
        weights.

        Parameters:
        ----------
        selected_gps:
            Names of the selected gene programs for which data should be
            retrieved.

        Returns:
        ----------
        selected_gps_idx:
            Index of the selected gene programs (dim: n_selected_gps,)
        selected_gps_rna_decoder_weights:
            Gene weights of the rna decoders of the selected gene programs
            (dim: (2 * n_genes) x n_selected_gps).
        selected_gps_atac_decoder_weights:
            Peak weights of the atac decoders of the selected gene programs
            (dim: (2 * n_peaks) x n_selected_gps).
        """
        self._check_if_trained(warn=True)

        # Get selected gps and their index
        all_gps = list(self.adata.uns[self.gp_names_key_])
        if selected_gps is None:
            selected_gps = all_gps
        elif isinstance(selected_gps, str):
            selected_gps = [selected_gps]
        selected_gps_idx = np.array([all_gps.index(gp) for gp in selected_gps])

        # Get weights of selected gps
        all_gps_rna_decoder_weights = self.model.get_gp_weights()[0]
        selected_gps_rna_decoder_weights = (
            all_gps_rna_decoder_weights[:, selected_gps_idx]
            .cpu().detach().numpy())
        
        if "atac" in self.modalities_:
            all_gps_atac_decoder_weights = self.model.get_gp_weights()[1]
            selected_gps_atac_decoder_weights = (
                all_gps_atac_decoder_weights[:, selected_gps_idx]
                .cpu().detach().numpy())
        else:
            selected_gps_atac_decoder_weights = None

        return (selected_gps_idx,
                selected_gps_rna_decoder_weights,
                selected_gps_atac_decoder_weights)

    def get_cat_covariates_embeds(self) -> np.ndarray:
        """
        Get the categorical covariates embeddings.

        Returns:
        ----------
        cat_covariates_embeds:
            Categorical covariates embeddings.
        """
        self._check_if_trained(warn=True)
        
        cat_covariates_embeds = []
        for cat_covariate_embedder in self.model.cat_covariates_embedders:
            cat_covariates_embeds.append(
                cat_covariate_embedder.weight.cpu().detach().numpy())
        return cat_covariates_embeds

    def get_active_gps(self) -> np.ndarray:
        """
        Get active gene programs based on the gene expression decoder gene
        weights of gene programs. Active gene programs are gene programs
        whose absolute gene weights aggregated over all genes are greater than
        ´self.active_gp_thresh_ratio_´ times the absolute gene weights
        aggregation of the gene program with the maximum value across all gene 
        programs.

        Parameters
        ----------
        adata:
            AnnData object to get the active gene programs for. If ´None´, uses
            the adata object stored in the model instance.

        Returns
        ----------
        active_gps:
            Gene program names of active gene programs (dim: n_active_gps,)
        """
        self._check_if_trained(warn=True)
        
        device = next(self.model.parameters()).device

        active_gp_mask = self.model.get_active_gp_mask()
        active_gp_mask = active_gp_mask.detach().cpu().numpy()
        active_gps = self.adata.uns[self.gp_names_key_][active_gp_mask]
        return active_gps

    def get_latent_representation(
            self, 
            adata: Optional[AnnData]=None,
            adata_atac: Optional[AnnData]=None,
            counts_key: Optional[str]="counts",
            adj_key: str="spatial_connectivities",
            cat_covariates_keys: Optional[List[str]]=None,
            only_active_gps: bool=True,
            return_mu_std: bool=False,
            node_batch_size: int=64,
            dtype: type=np.float64,
            ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get the latent representation / gene program scores from a trained model.

        Parameters
        ----------
        adata:
            AnnData object to get the latent representation for. If ´None´, uses
            the adata object stored in the model instance.
        counts_key:
            Key under which the counts are stored in ´adata.layer´. If ´None´,
            uses ´adata.X´ as counts. 
        adj_key:
            Key under which the sparse adjacency matrix is stored in 
            ´adata.obsp´.
        cat_covariates_keys:
            Keys under which the categorical covariates are stored in ´adata.obs´.
        only_active_gps:
            If ´True´, return only the latent representation of active gps.              
        return_mu_std:
            If `True`, return ´mu´ and ´std´ instead of latent features ´z´.
        node_batch_size:
            Batch size used during data loading.
        dtype:
            Precision to store the latent representations.

        Returns
        ----------
        z:
            Latent space features (dim: n_obs x n_active_gps or n_obs x n_gps).
        mu:
            Expected values of the latent posterior (dim: n_obs x n_active_gps 
            or n_obs x n_gps).
        std:
            Standard deviations of the latent posterior (dim: n_obs x 
            n_active_gps or n_obs x n_gps).
        """
        self._check_if_trained(warn=False)
        
        device = next(self.model.parameters()).device

        if adata is None:
            adata = self.adata
        if (adata_atac is None) & hasattr(self, "adata_atac"):
            adata_atac = self.adata_atac

        # Create single dataloader containing entire dataset
        data_dict = prepare_data(
            adata=adata,
            cat_covariates_label_encoders=self.model.cat_covariates_label_encoders_,
            adata_atac=adata_atac,
            counts_key=counts_key,
            adj_key=adj_key,
            cat_covariates_keys=cat_covariates_keys,
            edge_val_ratio=0.,
            edge_test_ratio=0.,
            node_val_ratio=0.,
            node_test_ratio=0.)
        node_masked_data = data_dict["node_masked_data"]
        loader_dict = initialize_dataloaders(
            node_masked_data=node_masked_data,
            edge_train_data=None,
            edge_val_data=None,
            edge_batch_size=None,
            node_batch_size=node_batch_size,
            shuffle=False)
        node_loader = loader_dict["node_train_loader"]

        # Get number of gene programs
        if only_active_gps:
            n_gps = self.get_active_gps().shape[0]
        else:
            n_gps = (self.n_prior_gp_ + self.n_addon_gp_ )

        # Initialize latent vectors
        if return_mu_std:
            mu = np.empty(shape=(adata.shape[0], n_gps), dtype=dtype)
            std = np.empty(shape=(adata.shape[0], n_gps), dtype=dtype)
        else:
            z = np.empty(shape=(adata.shape[0], n_gps), dtype=dtype)

        # Get latent representation for each batch of the dataloader and put it
        # into latent vectors
        for i, node_batch in enumerate(node_loader):
            n_obs_before_batch = i * node_batch_size
            n_obs_after_batch = n_obs_before_batch + node_batch.batch_size
            node_batch = node_batch.to(device)
            if return_mu_std:
                mu_batch, std_batch = self.model.get_latent_representation(
                    node_batch=node_batch,
                    only_active_gps=only_active_gps,
                    return_mu_std=True)
                mu[n_obs_before_batch:n_obs_after_batch, :] = (
                    mu_batch.detach().cpu().numpy())
                std[n_obs_before_batch:n_obs_after_batch, :] = (
                    std_batch.detach().cpu().numpy())
            else:
                z_batch = self.model.get_latent_representation(
                    node_batch=node_batch,
                    only_active_gps=only_active_gps,
                    return_mu_std=False)
                z[n_obs_before_batch:n_obs_after_batch, :] = (
                    z_batch.detach().cpu().numpy())
        if return_mu_std:
            return mu, std
        else:
            return z
        
    def get_omics_decoder_outputs(
                self, 
                adata: Optional[AnnData]=None,
                adata_atac: Optional[AnnData]=None,
                only_active_gps: bool=True,
                node_batch_size: int=64,
                ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
            """
            Get the omics decoder outputs.

            Parameters
            ----------
            adata:
                AnnData object to get the latent representation for. If ´None´, uses
                the adata object stored in the model instance.
            counts_key:
                Key under which the counts are stored in ´adata.layer´. If ´None´,
                uses ´adata.X´ as counts. 
            adj_key:
                Key under which the sparse adjacency matrix is stored in 
                ´adata.obsp´.
            cat_covariates_keys:
                Keys under which the categorical covariates are stored in ´adata.obs´.
            only_active_gps:
                If ´True´, return only the latent representation of active gps.

            Returns
            ----------
            output:
                A dictionary containing the omics decoder outputs.
            """
            self._check_if_trained(warn=False)

            device = next(self.model.parameters()).device

            if adata is None:
                adata = self.adata
            if (adata_atac is None) & hasattr(self, "adata_atac"):
                adata_atac = self.adata_atac

            # Create single dataloader containing entire dataset
            data_dict = prepare_data(
                adata=adata,
                cat_covariates_label_encoders=self.model.cat_covariates_label_encoders_,
                adata_atac=adata_atac,
                counts_key=self.counts_key_,
                adj_key=self.adj_key_,
                cat_covariates_keys=self.cat_covariates_keys_,
                edge_val_ratio=0.,
                edge_test_ratio=0.,
                node_val_ratio=0.,
                node_test_ratio=0.)
            node_masked_data = data_dict["node_masked_data"]
            loader_dict = initialize_dataloaders(
                node_masked_data=node_masked_data,
                edge_train_data=None,
                edge_val_data=None,
                edge_batch_size=None,
                node_batch_size=node_batch_size,
                shuffle=False)
            node_loader = loader_dict["node_train_loader"]
            
            output = {}    
            output["target_rna_nb_means"] = np.empty(shape=(adata.shape[0], self.n_output_genes_))
            output["source_rna_nb_means"] = np.empty(shape=(adata.shape[0], self.n_output_genes_))
            if "atac" in self.modalities_:
                output["target_atac_nb_means"] = np.empty(shape=(adata.shape[0], self.n_output_peaks_))
                output["source_atac_nb_means"] = np.empty(shape=(adata.shape[0], self.n_output_peaks_))

            # Get latent representation for each batch of the dataloader and put it
            # into latent vectors
            for i, node_batch in enumerate(node_loader):
                n_obs_before_batch = i * node_batch_size
                n_obs_after_batch = n_obs_before_batch + node_batch.batch_size
                node_batch = node_batch.to(device)
                output_batch = self.model.get_omics_decoder_outputs(
                    node_batch=node_batch,
                    only_active_gps=only_active_gps)
                output["target_rna_nb_means"][n_obs_before_batch:n_obs_after_batch, :] = (
                    output_batch["target_rna_nb_means"].detach().cpu().numpy())
                output["source_rna_nb_means"][n_obs_before_batch:n_obs_after_batch, :] = (
                    output_batch["source_rna_nb_means"].detach().cpu().numpy())
                if "atac" in self.modalities_:
                    output["target_atac_nb_means"][n_obs_before_batch:n_obs_after_batch, :] = (
                        output_batch["target_atac_nb_means"].detach().cpu().numpy())
                    output["source_atac_nb_means"][n_obs_before_batch:n_obs_after_batch, :] = (
                        output_batch["source_atac_nb_means"].detach().cpu().numpy())
            return output
    
    @torch.no_grad()
    def get_recon_edge_probs(self,      
                             node_batch_size: int=2048,
                             device: Optional[str]=None,
                             edge_thresh: Optional[float]=None,
                             n_neighbors: Optional[int]=None,
                             return_edge_probs: bool=False
                             ) -> Union[sp.csr_matrix, torch.Tensor]:
        """
        Get the reconstructed adjacency matrix (or edge probability matrix if 
        ´return_edge_probs == True´ from a trained NicheCompass model.

        Parameters
        ----------
        node_batch_size:
            Batch size for batched decoder forward pass to alleviate memory
            consumption. Only relevant if ´return_edge_probs == False´.
        device:
            Device where the computation will be executed.
        edge_thresh:
            Probability threshold above or equal to which edge probabilities
            lead to a reconstructed edge. If ´None´, ´n_neighbors´ will be used
            to compute an independent edge threshold for each observation.
        n_neighbors:
            Number of neighbors used to compute an independent edge threshold
            for each observation (before the adjacency matrix is made
            symmetric).Only applies if ´edge_thresh is None´. In some occassions
            when multiple edges have the same probability, the number of
            reconstructed edges can slightly deviate from ´n_neighbors´. If
            ´None´, the number of neighbors in the original (symmetric) spatial
            graph stored in ´adata.obsp[self.adj_key_]´ are used to compute an
            independent edge threshold for each observation (in this case the
            adjacency matrix is not made symmetric). 
        return_edge_probs:
            If ´True´, return a matrix of edge probabilities instead of the
            reconstructed adjacency matrix. This will require a lot of memory
            as a dense tensor will be returned instead of a sparse matrix.

        Returns
        ----------
        adj_recon:
            Sparse scipy matrix containing reconstructed edges (dim: n_nodes x
            n_nodes).
        adj_recon_probs:
            Tensor containing edge probabilities (dim: n_nodes x n_nodes).
        """
        self._check_if_trained(warn=False)
        model_device = next(self.model.parameters()).device
        if device is None:
            # Get device from model
            device = model_device
        else:
            self.model.to(device)

        if edge_thresh is None:
            compute_edge_thresh = True
        
        # Get the latent representation for each observation
        if self.latent_key_ not in self.adata.obsm:
            raise ValueError("Please first store the latent representations in "
                             f"adata.obsm['{self.latent_key_}']. They can be "
                             "retrieved via "
                             "'model.get_latent_representation()'.")
        z = torch.tensor(self.adata.obsm[self.latent_key_], device=device)

        # Add 0s for inactive gps back to stored latent representation which
        # only contains active gps (model expects all gps with inactive ones
        # having 0 values)
        active_gp_mask = self.model.get_active_gp_mask()
        z_with_inactive = torch.zeros((z.shape[0], active_gp_mask.shape[0]),
                                      dtype=torch.float64, device=device)
        active_gp_idx = (active_gp_mask == 1).nonzero().t()
        active_gp_idx = active_gp_idx.repeat(z_with_inactive.shape[0], 1)
        z_with_inactive = z_with_inactive.scatter(1, active_gp_idx, z)

        if not return_edge_probs:
            # Initialize global reconstructed adjacency matrix
            adj_recon = sp.lil_matrix((len(self.adata), len(self.adata)))

            for i in range(0, len(self.adata), node_batch_size):
                # Get edge probabilities for current batch
                adj_recon_logits = self.model.graph_decoder(
                    z=z_with_inactive,
                    reduced_obs_start_idx=i,
                    reduced_obs_end_idx=i+node_batch_size)
                adj_recon_probs_batch = torch.sigmoid(adj_recon_logits)

                if compute_edge_thresh:
                    if n_neighbors is None:
                        # Get neighbors from spatial (input) adjacency matrix
                        n_neighs_adj = np.array(
                            self.adata.obsp[self.adj_key_][i: i+node_batch_size]
                            .sum(axis=1).astype(int)).flatten()
                    else:
                        n_neighs_adj = np.ones(
                            [adj_recon_probs_batch.shape[0]],
                            dtype=int) * n_neighbors
                    adj_recon_probs_batch_sorted = adj_recon_probs_batch.sort(
                        descending=True)[0]
                    edge_thresh = adj_recon_probs_batch_sorted[
                        np.arange(adj_recon_probs_batch_sorted.shape[0]),
                        n_neighs_adj-1]
                    edge_thresh = edge_thresh.view(-1, 1).expand_as(
                        adj_recon_probs_batch)

                # Convert edge probabilities to edges
                adj_recon_batch = (adj_recon_probs_batch >= edge_thresh).long()
                adj_recon_batch = adj_recon_batch.cpu().numpy()
                adj_recon[i:i+node_batch_size, :] = adj_recon_batch
        else:
            adj_recon_logits = self.model.graph_decoder(
                z=z_with_inactive)
            adj_recon_probs = torch.sigmoid(adj_recon_logits)

        if device is not None:
            # Move model back to original device
            self.model.to(model_device)

        if not return_edge_probs:
            adj_recon = adj_recon.tocsr(copy=False)
            if n_neighbors is not None:
                # Make adjacency matrix symmetric
                adj_recon = adj_recon.maximum(adj_recon.T)
            return adj_recon
        else:
            return adj_recon_probs

    @torch.no_grad()
    def get_neighbor_importances(
            self,      
            node_batch_size: Optional[int]=None) -> sp.csr_matrix:
        """
        Get the aggregation weights of the node label aggregator. The
        aggregation weights indicate how much importance each node / observation
        has attributed to its neighboring nodes / observations for the omics 
        reconstruction tasks. If ´one-hop-attention´ is used as node label
        method, the mean over all attention heads is used as aggregation
        weights.

        Parameters
        ----------
        node_batch_size:
            Batch size that is used by the node-level dataloader. If ´None´,
            uses the node batch size used during model training.

        Returns
        ----------
        agg_weights:
            A sparse scipy matrix containing the aggregation weights of the node
            label aggregator (dim: n_obs x n_obs). Row-wise entries will be
            neighbor importances for each observation. The matrix is not
            symmetric.
        """
        self._check_if_trained(warn=False)
        device = next(self.model.parameters()).device

        if node_batch_size is None:
            node_batch_size = self.node_batch_size_

        # Initialize global aggregation weights matrix
        agg_weights = sp.lil_matrix((len(self.adata), len(self.adata)))

        # Create single dataloader containing entire dataset
        data_dict = prepare_data(
            adata=self.adata,
            cat_covariates_label_encoders=self.model.cat_covariates_label_encoders_,
            adata_atac=self.adata_atac,
            counts_key=self.counts_key_,
            adj_key=self.adj_key_,
            cat_covariates_keys=self.cat_covariates_keys_,
            edge_val_ratio=0.,
            edge_test_ratio=0.,
            node_val_ratio=0.,
            node_test_ratio=0.)
        node_masked_data = data_dict["node_masked_data"]
        loader_dict = initialize_dataloaders(
            node_masked_data=node_masked_data,
            edge_train_data=None,
            edge_val_data=None,
            edge_batch_size=None,
            node_batch_size=node_batch_size,
            shuffle=False)
        node_loader = loader_dict["node_train_loader"]

        # Get aggregation weights for each node batch of the dataloader and put
        # them into the global aggregation weights matrix
        for i, node_batch in enumerate(node_loader):
            node_batch = node_batch.to(device)
            n_obs_before_batch = i * node_batch_size
            n_obs_after_batch = n_obs_before_batch + node_batch.batch_size

            _, alpha = (self.model.node_label_aggregator(
                x=node_batch.x,
                edge_index=node_batch.edge_index,
                return_agg_weights=True))

            # Filter global edge index and aggregation weights for nodes in
            # current batch (exclude sampled neighbors across dim 1)
            global_edge_index = node_batch.edge_attr.t()
            batch_mask = ((global_edge_index[1] >= n_obs_before_batch) & 
                          (global_edge_index[1] < n_obs_after_batch))
            global_edge_index = global_edge_index[:, batch_mask]
            if alpha.ndim > 1:
                # Compute mean over attention heads
                alpha = alpha.mean(dim=-1)
            alpha = alpha[batch_mask]

            # Insert aggregation weights from current node batch in global
            # aggregation weights matrix
            global_edge_index = global_edge_index.cpu().numpy()
            alpha = alpha.cpu().numpy()
            agg_weights[global_edge_index[1, :],
                        global_edge_index[0, :]] = alpha
        agg_weights = agg_weights.tocsr(copy=False)
        return agg_weights
    

    def get_gp_summary(self) -> pd.DataFrame:
        """
        Get summary information of gene programs and return it as a DataFrame.
        
        Returns
        ----------
        gp_summary_df:
            DataFrame with gene program summary information.
        """
        device = next(self.model.parameters()).device
        
        # Get source and target omics decoder weights
        _, gp_gene_weights, gp_peak_weights = self.get_gp_data()

        # Normalize gp weights to get gene importances
        gp_gene_importances = np.where(
            np.abs(gp_gene_weights).sum(0) != 0,
            np.abs(gp_gene_weights) / np.abs(gp_gene_weights).sum(0),
            0)      

        # Split gene weights and importances into source and target part
        gp_gene_weights = np.transpose(gp_gene_weights)
        gp_gene_importances = np.transpose(gp_gene_importances)
        gp_gene_weights_source = gp_gene_weights[
            :, (gp_gene_weights.shape[1] // 2):]
        gp_gene_weights_target = gp_gene_weights[
            :, :(gp_gene_weights.shape[1] // 2)]
        gp_gene_importances_source = gp_gene_importances[
            :, (gp_gene_weights.shape[1] // 2):]
        gp_gene_importances_target = gp_gene_importances[
            :, :(gp_gene_weights.shape[1] // 2)]
        
        # Get source and target gene masks
        gp_gene_mask_source = np.transpose(
            np.array(self.model.source_rna_decoder_mask).T != 0)
        gp_gene_mask_target = np.transpose(
            np.array(self.model.target_rna_decoder_mask).T != 0)
        
        # Add entries to gp mask for addon gps
        if self.n_addon_gp_ > 0:
            gp_gene_addon_mask_source = np.transpose(
            np.array(self.model.source_rna_decoder_addon_mask).T != 0)
            gp_gene_addon_mask_target = np.transpose(
            np.array(self.model.target_rna_decoder_addon_mask).T != 0)
            gp_gene_mask_source = np.concatenate(
                (gp_gene_mask_source, gp_gene_addon_mask_source), axis=0)
            gp_gene_mask_target = np.concatenate(
                (gp_gene_mask_target, gp_gene_addon_mask_target), axis=0)

        # Get active gp mask
        gp_active_status = (self.model.get_active_gp_mask().cpu().detach()
                            .numpy().tolist())

        active_gps = list(self.get_active_gps())
        all_gps = list(self.adata.uns[self.gp_names_key_])

        # Collect info for each gp in lists of lists
        gp_names = []
        active_gp_idx = [] # Index among active gene programs
        all_gp_idx = [] # Index among all gene programs
        n_source_genes = []
        n_non_zero_source_genes = []
        n_target_genes = []
        n_non_zero_target_genes = []
        gp_source_genes = []
        gp_target_genes = []
        gp_source_genes_weights = []
        gp_target_genes_weights = []
        gp_source_genes_importances = []
        gp_target_genes_importances = []
        for (name,
             gene_mask_source,
             gene_mask_target,
             gene_weights_source,
             gene_weights_target,
             gene_importances_source,
             gene_importances_target) in zip(
                all_gps,
                gp_gene_mask_source,
                gp_gene_mask_target,
                gp_gene_weights_source,
                gp_gene_weights_target,
                gp_gene_importances_source,
                gp_gene_importances_target):
            gp_names.append(name)
            active_gp_idx.append(active_gps.index(name)
                                 if name in active_gps else np.nan)
            all_gp_idx.append(all_gps.index(name))

            # Sort source genes according to absolute weights
            gene_weights_source_sorted = []
            gene_importances_source_sorted = []
            genes_source_sorted = []
            for _, weights, importances, genes in sorted(zip(
                np.abs(np.around(gene_weights_source[gene_mask_source],
                                 decimals=4)), # just for sorting
                np.around(gene_weights_source[gene_mask_source],
                          decimals=4),
                np.around(gene_importances_source[gene_mask_source],
                          decimals=4),        
                self.adata.var_names[gene_mask_source].tolist()), reverse=True):
                    genes_source_sorted.append(genes)
                    gene_weights_source_sorted.append(weights)
                    gene_importances_source_sorted.append(importances)
            
            # Sort target genes according to absolute weights
            geme_weights_target_sorted = []
            gene_importances_target_sorted = []
            genes_target_sorted = []
            for _, weights, importances, genes in sorted(zip(
                np.abs(np.around(gene_weights_target[gene_mask_target],
                                 decimals=4)), # just for sorting
                np.around(gene_weights_target[gene_mask_target],
                          decimals=4),                 
                np.around(gene_importances_target[gene_mask_target],
                          decimals=4),
                self.adata.var_names[gene_mask_target].tolist()), reverse=True):
                    genes_target_sorted.append(genes)
                    geme_weights_target_sorted.append(weights)
                    gene_importances_target_sorted.append(importances)                 
                
            n_source_genes.append(len(genes_source_sorted))
            n_non_zero_source_genes.append(len(np.array(
                gene_weights_source_sorted).nonzero()[0]))
            n_target_genes.append(len(genes_target_sorted))
            n_non_zero_target_genes.append(len(np.array(
                geme_weights_target_sorted).nonzero()[0]))
            gp_source_genes.append(genes_source_sorted)
            gp_target_genes.append(genes_target_sorted)
            gp_source_genes_weights.append(gene_weights_source_sorted)
            gp_target_genes_weights.append(geme_weights_target_sorted)
            gp_source_genes_importances.append(gene_importances_source_sorted)
            gp_target_genes_importances.append(gene_importances_target_sorted)
   
        gp_summary_df = pd.DataFrame(
            {"gp_name": gp_names,
             "all_gp_idx": all_gp_idx,
             "gp_active": gp_active_status,
             "active_gp_idx": active_gp_idx,
             "n_source_genes": n_source_genes,
             "n_non_zero_source_genes": n_non_zero_source_genes,
             "n_target_genes": n_target_genes,
             "n_non_zero_target_genes": n_non_zero_target_genes,
             "gp_source_genes": gp_source_genes,
             "gp_target_genes": gp_target_genes,
             "gp_source_genes_weights": gp_source_genes_weights,
             "gp_target_genes_weights": gp_target_genes_weights,
             "gp_source_genes_importances": gp_source_genes_importances,
             "gp_target_genes_importances": gp_target_genes_importances})
        
        gp_summary_df["active_gp_idx"] = (
            gp_summary_df["active_gp_idx"].astype("Int64"))
        
        if "atac" in self.modalities_:
            # Add peak info for each gp
            
            # Normalize gp weights to get gene importances
            gp_peak_importances = np.where(
                np.abs(gp_peak_weights).sum(0) != 0,
                np.abs(gp_peak_weights) / np.abs(gp_peak_weights).sum(0),
                0)
        
            # Split peak weights and importances into source and target part
            gp_peak_weights = np.transpose(gp_peak_weights)
            gp_peak_importances = np.transpose(gp_peak_importances)
            gp_peak_weights_source = gp_peak_weights[
                :, (gp_peak_weights.shape[1] // 2):]
            gp_peak_weights_target = gp_peak_weights[
                :, :(gp_peak_weights.shape[1] // 2)]
            gp_peak_importances_source = gp_peak_importances[
                :, (gp_peak_weights.shape[1] // 2):]
            gp_peak_importances_target = gp_peak_importances[
                :, :(gp_peak_weights.shape[1] // 2)]

            # Get source and target peak masks
            gp_peak_mask_source = np.transpose(
                np.array(
                    self.model.source_atac_decoder_mask.to_dense()).T != 0)
            gp_peak_mask_target = np.transpose(
                np.array(
                    self.model.target_atac_decoder_mask.to_dense()).T != 0)

            # Add entries to gp mask for addon gps
            if self.n_addon_gp_ > 0:
                gp_peak_addon_mask_source = np.transpose(
                np.array(self.model.source_atac_decoder_addon_mask).T != 0)
                gp_peak_addon_mask_target = np.transpose(
                np.array(self.model.target_atac_decoder_addon_mask).T != 0)
                gp_peak_mask_source = np.concatenate(
                    (gp_peak_mask_source, gp_peak_addon_mask_source), axis=0)
                gp_peak_mask_target = np.concatenate(
                    (gp_peak_mask_target, gp_peak_addon_mask_target), axis=0)

            # Collect info for each gp in lists of lists
            n_source_peaks = []
            n_non_zero_source_peaks = []
            n_target_peaks = []
            n_non_zero_target_peaks = []
            gp_source_peaks = []
            gp_target_peaks = []
            gp_source_peaks_weights = []
            gp_target_peaks_weights = []
            gp_source_peaks_importances = []
            gp_target_peaks_importances = []
            for (gp_source_peaks_idx,
                 gp_target_peaks_idx,
                 gp_source_peaks_weights_arr,
                 gp_target_peaks_weights_arr,
                 gp_source_peaks_importances_arr,
                 gp_target_peaks_importances_arr) in zip(
                    gp_peak_mask_source,
                    gp_peak_mask_target,
                    gp_peak_weights_source,
                    gp_peak_weights_target,
                    gp_peak_importances_source,
                    gp_peak_importances_target):
                # Sort source peaks according to absolute weights
                peak_weights_source_sorted = []
                peak_importances_source_sorted = []
                peaks_source_sorted = []
                for _, weights, importances, peaks in sorted(zip(
                    np.abs(np.around(gp_source_peaks_weights_arr[gp_source_peaks_idx],
                                    decimals=4)), # just for sorting
                    np.around(gp_source_peaks_weights_arr[gp_source_peaks_idx],
                            decimals=4),
                    np.around(gp_source_peaks_importances_arr[gp_source_peaks_idx],
                            decimals=4),        
                    self.adata_atac.var_names[gp_source_peaks_idx].tolist()),reverse=True):
                        peaks_source_sorted.append(peaks)
                        peak_weights_source_sorted.append(weights)
                        peak_importances_source_sorted.append(importances)
                
                # Sort target peaks according to absolute weights
                peak_weights_target_sorted = []
                peak_importances_target_sorted = []
                peaks_target_sorted = []
                for _, weights, importances, peaks in sorted(zip(
                    np.abs(np.around(gp_target_peaks_weights_arr[gp_target_peaks_idx],
                                    decimals=4)),
                    np.around(gp_target_peaks_weights_arr[gp_target_peaks_idx],
                            decimals=4),                 
                    np.around(gp_target_peaks_importances_arr[gp_target_peaks_idx],
                            decimals=4),
                    self.adata_atac.var_names[gp_target_peaks_idx].tolist()), reverse=True):
                        peaks_target_sorted.append(peaks)
                        peak_weights_target_sorted.append(weights)
                        peak_importances_target_sorted.append(importances)                 
                    
                n_source_peaks.append(len(peaks_source_sorted))
                n_non_zero_source_peaks.append(len(np.array(
                    peak_weights_source_sorted).nonzero()[0]))
                n_target_peaks.append(len(peaks_target_sorted))
                n_non_zero_target_peaks.append(len(np.array(
                    peak_weights_target_sorted).nonzero()[0]))
                gp_source_peaks.append(peaks_source_sorted)
                gp_target_peaks.append(peaks_target_sorted)
                gp_source_peaks_weights.append(peak_weights_source_sorted)
                gp_target_peaks_weights.append(peak_weights_target_sorted)
                gp_source_peaks_importances.append(peak_importances_source_sorted)
                gp_target_peaks_importances.append(peak_importances_target_sorted)

            gp_summary_df["n_source_peaks"] = n_source_peaks
            gp_summary_df["n_non_zero_source_peaks"] = n_non_zero_source_peaks
            gp_summary_df["n_target_peaks"] = n_target_peaks
            gp_summary_df["n_non_zero_target_peaks"] = n_non_zero_target_peaks
            gp_summary_df["gp_source_peaks"] = gp_source_peaks
            gp_summary_df["gp_target_peaks"] = gp_target_peaks
            gp_summary_df["gp_source_peaks_weights"] = gp_source_peaks_weights
            gp_summary_df["gp_target_peaks_weights"] = gp_target_peaks_weights
            gp_summary_df["gp_source_peaks_importances"] = gp_source_peaks_importances
            gp_summary_df["gp_target_peaks_importances"] = gp_target_peaks_importances
            gp_summary_df["gp_source_peaks_importances"] = (
                gp_summary_df["gp_source_peaks_importances"].replace(np.nan, 0.))
            gp_summary_df["gp_target_peaks_importances"] = (
                gp_summary_df["gp_target_peaks_importances"].replace(np.nan, 0.))

        return gp_summary_df
    

    def add_active_gp_scores_to_obs(self) -> None:
        """
        Add the expression of all active gene programs to ´adata.obs´.      
        """
        # Get active gene program names
        active_gp_names = self.get_active_gps()
        
        # Create active gene program df
        active_gp_df = pd.DataFrame(self.adata.obsm[self.latent_key_],
                                    columns=active_gp_names)
        active_gp_df = active_gp_df.set_index(self.adata.obs.index)

        # Drop columns if they are already in ´adata.obs´
        for col in active_gp_df.columns:
            if col in self.adata.obs:
                self.adata.obs.drop(col, axis=1, inplace=True)

        # Concatenate active gene program df horizontally to ´adata.obs´
        self.adata.obs = pd.concat([self.adata.obs, active_gp_df], axis=1)
        