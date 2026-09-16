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
from nichecompass.train.distributed import (cleanup_distributed,
                                            is_main_process)
from .basemodelmixin import BaseModelMixin
from .gpanalysismixin import GPAnalysisMixin


def _mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
    """
    Read a decoder mask into host memory as a numpy array.

    The masks are registered buffers, so ´Module.to´ moves them onto whichever
    device the model is on, and ´np.array´ refuses a CUDA tensor with

        TypeError: can't convert cuda:0 device type tensor to numpy

    They used to be plain attributes that ´Module.to´ never touched, which is
    why reading them directly worked for as long as it did. Sparse masks are
    densified first, as they were before.
    """
    if mask.is_sparse:
        mask = mask.to_dense()
    return mask.detach().cpu().numpy()


class NicheCompass(GPAnalysisMixin, BaseModelMixin):
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
        If ´True´, applies batch normalization to the shared fully connected hidden
        representation. Only has an effect when there are two fully
        connected encoder layers.
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
                 encoder_use_bn: bool=True,
                 dropout_rate_encoder: float=0.,
                 dropout_rate_graph_decoder: float=0.,
                 cat_covariates_cats: Optional[List[List]]=None,
                 n_addon_gp: int=100,
                 cat_covariates_embeds_nums: Optional[List[int]]=None,
                 include_edge_kl_loss: bool=True,
                 use_cuda_if_available: bool=True,
                 seed: int=0,
                 **kwargs):
        self.gp_analysis_default_orientation_ = "canonical"
        self.adata = adata
        self.adata_atac = adata_atac
        self.gp_analysis_feature_names_ = {"rna": list(map(str, adata.var_names))}
        if adata_atac is not None:
            self.gp_analysis_feature_names_["atac"] = list(map(str, adata_atac.var_names))
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
              multi_gpu: bool=False,
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
        multi_gpu:
            If `True`, split training across all processes of a distributed
            job, so that every process holds a copy of the model and works on a
            disjoint part of the training edges and nodes. Gradients are
            averaged across processes, so ´edge_batch_size´ and
            ´node_batch_size´ keep their meaning as the global batch sizes and
            the number of optimizer steps per epoch is unchanged; the speedup
            comes from each process computing a ´world_size´-th of every batch.
            Requires the script to be launched as a distributed job, for
            example with ´torchrun --nproc_per_node=4 your_script.py´, and
            raises otherwise. Leaving this at `False` reproduces the previous
            behavior exactly.
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
            multi_gpu=multi_gpu,
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
        
        # The per process batch size is not the batch size the user asked for,
        # so the global one is kept for everything that follows training
        # The caller's own number, not the effective global batch: the
        # inference pass below is single process, so under
        # ´batch_size_scaling="per_process"´ the effective batch would be
        # ´world_size´ times too large for it.
        self.node_batch_size_ = self.trainer.requested_node_batch_size_
        
        self.is_trained_ = True
        self.model.eval()

        # Everything below writes into ´adata´ and runs over the whole dataset.
        # Every process holds its own copy of ´adata´, so letting all of them
        # do it would duplicate the work and leave the copies of every process
        # other than the main one unused. The other processes wait inside
        # ´cleanup_distributed´, which synchronizes before it releases the
        # process group, so that no process tears the group down while another
        # is still inside a collective.
        if not is_main_process():
            cleanup_distributed()
            return

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
        if hasattr(self.model.target_rna_decoder.nb_means_normalized_decoder, "masked_l"):
            self.prepare_gp_analysis(overwrite=True)

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

        cleanup_distributed()

    def run_differential_gp_tests(
            self, cat_key: str, selected_cats=None, comparison_cats="rest",
            selected_gps=None, n_sample: int=10000,
            log_bayes_factor_thresh: float=2.3,
            key_added: str="nichecompass_differential_gp_test_results",
            seed: int=0, adata: Optional[AnnData]=None,
            orientation=None, direction: str="both", return_all: bool=False,
            adata_atac: Optional[AnnData]=None):
        """Compare GP activities between cell distributions.

        For randomly sampled cell pairs, analytically integrate the Gaussian
        posterior probability that the focal activity exceeds the comparison
        activity. Its log odds is retained as ``log_bayes_factor`` for API
        compatibility. This is not a test of population means or a donor-level
        effect, and the cutoff is not FDR control.

        Parameters
        ----------
        cat_key:
            Observation column defining the compared populations. Missing
            labels are excluded, including from the rest population.
        selected_cats:
            Focal category labels; by default all nonmissing categories.
        comparison_cats:
            A category/list of categories or the sentinel ``"rest"``. To use
            a literal category called rest as comparator pass ``["rest"]``.
        selected_gps:
            GP names, active programs by default.
        n_sample:
            Number of independent cell pairs sampled with replacement.
        log_bayes_factor_thresh:
            Absolute log-odds cutoff for the filtered result.
        key_added:
            Filtered table key in ``adata.uns``. All tested contrasts are
            stored at ``key_added + "_all"`` and provenance at ``+ "_params"``.
        seed:
            Local random-generator seed; does not modify NumPy's global RNG.
        adata:
            Data to encode; defaults to the model data.
        adata_atac:
            Aligned ATAC observations for an external multimodal analysis.
        orientation:
            ``canonical``, ``raw``, or the model's convention if None.
        direction:
            Filter to ``higher``, ``lower``, or ``both`` directions.
        return_all:
            If True return the complete DataFrame; otherwise return the list
            of GP names passing the threshold/direction filter (legacy API).

        Notes
        -----
        GP columns in obs are refreshed from the same posterior used for the
        test. The orientation ID is stored with results so plotting can reject
        stale results instead of combining different conventions.
        """
        self._check_if_trained(warn=False)
        if not isinstance(n_sample, (int, np.integer)) or n_sample <= 0:
            raise ValueError("n_sample must be a positive integer.")
        if not np.isfinite(log_bayes_factor_thresh) or log_bayes_factor_thresh < 0:
            raise ValueError("log_bayes_factor_thresh must be finite and nonnegative.")
        if direction not in ("both", "higher", "lower"):
            raise ValueError("direction must be 'both', 'higher', or 'lower'.")
        adata = self.adata if adata is None else adata
        names, _ = self._gp_selection(selected_gps, active=True)
        if not names:
            raise ValueError("No gene programs selected for testing.")
        labels = adata.obs[cat_key].astype(object)
        valid = labels.notna().to_numpy()
        cats = list(pd.unique(labels[valid]))
        def as_list(value):
            return [value] if pd.api.types.is_scalar(value) else list(value)
        focal = cats if selected_cats is None else as_list(selected_cats)
        rest = isinstance(comparison_cats, str) and comparison_cats == "rest"
        comparison = [] if rest else as_list(comparison_cats)
        if not set(focal).issubset(cats) or not set(comparison).issubset(cats):
            raise ValueError("Selected and comparison categories must exist in the data.")
        contrasts = []
        for cat in focal:
            if not rest and cat in comparison:
                continue
            mask = (labels == cat).to_numpy() & valid
            other = (~mask & valid) if rest else labels.isin(comparison).to_numpy() & valid
            if not mask.any() or not other.any():
                raise ValueError(f"Both populations must be nonempty for category {cat!r}.")
            contrasts.append((cat, mask, other))
        if not contrasts:
            raise ValueError("No non-overlapping category contrasts selected.")
        orientation = self._gp_orientation(orientation)
        mu, std = self.get_gp_activities(names, adata=adata, adata_atac=adata_atac,
                                          return_std=True, orientation=orientation)
        if not (np.isfinite(mu).all() and np.isfinite(std).all()) or (std < 0).any():
            raise ValueError("Posterior means/std must be finite and standard deviations nonnegative.")
        if orientation == "canonical":
            quality = self._gp_analysis_table().set_index("gp_name").loc[names]
            signs = quality.orientation_sign.to_numpy()
            statuses = quality.orientation_status.to_numpy()
            gp_ids = quality.gp_id.to_numpy()
            orientation_id = self.gp_analysis_["orientation_id"]
        else:
            signs = np.ones(len(names), dtype=int)
            statuses = np.repeat("raw", len(names))
            gp_ids = np.array(names)
            orientation_id = "raw"
        rng = np.random.default_rng(seed)
        rows = []
        for cat, mask, other in contrasts:
            a = rng.choice(np.flatnonzero(mask), n_sample)
            b = rng.choice(np.flatnonzero(other), n_sample)
            # Bound temporary posterior arrays even for large GP dictionaries.
            probability_sum = np.zeros(len(names), dtype=np.float64)
            for start in range(0, n_sample, 1024):
                aa, bb = a[start:start+1024], b[start:start+1024]
                diff = mu[aa] - mu[bb]
                denominator = np.sqrt(2 * (std[aa]**2 + std[bb]**2))
                standardized = np.divide(-diff, denominator, out=np.zeros_like(diff), where=denominator > 0)
                probability = 0.5 * erfc(standardized)
                probability = np.where(denominator > 0, probability,
                                       np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5)))
                probability_sum += probability.sum(axis=0)
            p_higher = np.clip(probability_sum / n_sample, 0, 1)
            p_lower = 1 - p_higher
            statistic = np.log(p_higher + 1e-12) - np.log(p_lower + 1e-12)
            effect = mu[mask].mean(0) - mu[other].mean(0)
            comparison_label = "rest" if rest else str(comparison)
            for k, name in enumerate(names):
                rows.append({"category": cat, "comparison": comparison_label, "gene_program": name,
                             "gp_id": gp_ids[k], "p_h0": p_higher[k], "p_h1": p_lower[k],
                             "p_higher": p_higher[k], "p_lower": p_lower[k],
                             "log_bayes_factor": statistic[k], "mean_difference": effect[k],
                             "direction": "higher" if statistic[k] > 0 else "lower" if statistic[k] < 0 else "equal",
                             "n_focal": int(mask.sum()), "n_comparison": int(other.sum()),
                             "orientation_sign": int(signs[k]), "orientation_status": statuses[k],
                             "orientation": orientation, "orientation_id": orientation_id})
        all_results = pd.DataFrame(rows)
        order = np.argsort(-np.abs(all_results.log_bayes_factor.to_numpy()), kind="stable")
        all_results = all_results.iloc[order].reset_index(drop=True)
        keep = all_results.log_bayes_factor.abs() > log_bayes_factor_thresh
        if direction != "both":
            keep &= all_results.direction == direction
        results = all_results[keep].reset_index(drop=True)
        adata.uns[key_added] = results
        adata.uns[key_added + "_all"] = all_results
        adata.uns[key_added + "_params"] = {
            "schema_version": 1, "method": "cell_pair_posterior_superiority",
            "orientation": orientation, "orientation_id": orientation_id,
            "n_sample": n_sample, "seed": seed, "direction": direction,
            "log_bayes_factor_thresh": log_bayes_factor_thresh}
        adata.uns[key_added + "_params"].update({
            "cat_key": cat_key,
            "input_fingerprint": self._gp_input_fingerprint(adata, cat_key, adata_atac)})
        # Assignment replaces stale or manually flipped columns, never toggles.
        self._write_gp_scores(adata, names, mu, orientation)
        return all_results.copy() if return_all else results.gene_program.unique().tolist()

    def compute_gp_gene_importances(self, selected_gp: str, orientation=None) -> pd.DataFrame:
        """Return full-precision RNA loadings and absolute loading importance.

        ``orientation=None`` uses the model analysis convention; ``raw`` opts
        out. Importances are normalized across source and target together.
        Signed loadings describe logits, not necessarily expression effects.
        """
        return self._gp_importances(selected_gp, "rna", "gene", orientation)

    def compute_gp_peak_importances(self, selected_gp: str, orientation=None) -> pd.DataFrame:
        """Return ATAC loadings using the same RNA-derived orientation."""
        if "atac" not in self.modalities_:
            raise ValueError("Peak importances require a model with ATAC data.")
        return self._gp_importances(selected_gp, "atac", "peak", orientation)

    def _gp_importances(self, selected_gp, modality, unit, orientation):
        table = self.get_gp_feature_table(selected_gp, orientation=orientation)
        table = table[(table.modality == modality) & (table.importance != 0)].copy()
        table = table.rename(columns={"feature": unit, "entity": f"{unit}_entity",
                                      "loading": f"{unit}_weight",
                                      "raw_loading": f"{unit}_weight_raw",
                                      "importance": f"{unit}_importance"})
        return table.sort_values([f"{unit}_importance", unit, f"{unit}_entity"],
                                 ascending=[False, True, True], kind="stable").reset_index(drop=True)

    def get_gp_data(self,
                    selected_gps: Optional[Union[str, list]]=None,
                    orientation: str="raw",
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
        selected_gps, selected_gps_idx = self._gp_selection(selected_gps)

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

        signs = self._gp_signs(selected_gps, orientation)
        if self._gp_orientation(orientation) == "canonical":
            effective_weights, _ = self._gp_arrays()
            selected_gps_rna_decoder_weights = effective_weights["rna"][:, selected_gps_idx].copy()
            if "atac" in self.modalities_:
                selected_gps_atac_decoder_weights = effective_weights["atac"][:, selected_gps_idx].copy()
        selected_gps_rna_decoder_weights *= signs
        if selected_gps_atac_decoder_weights is not None:
            selected_gps_atac_decoder_weights *= signs

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
            selected_gps=None,
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
        selected_gps:
            Optional GP names in output-column order. Selection occurs before
            copying each batch to the host, avoiding a full atlas-by-GP array.
            When only_active_gps is True, selected names must all be active.

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

        columns = slice(None)
        if selected_gps is not None:
            selected_names, all_indices = self._gp_selection(selected_gps)
            if only_active_gps:
                active_names = list(self.get_active_gps())
                if not set(selected_names).issubset(active_names):
                    raise ValueError("Selected GPs must be active when only_active_gps=True.")
                columns = [active_names.index(name) for name in selected_names]
            else:
                columns = all_indices.tolist()
            n_gps = len(selected_names)

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
                    mu_batch[:, columns].detach().cpu().numpy())
                std[n_obs_before_batch:n_obs_after_batch, :] = (
                    std_batch[:, columns].detach().cpu().numpy())
            else:
                z_batch = self.model.get_latent_representation(
                    node_batch=node_batch,
                    only_active_gps=only_active_gps,
                    return_mu_std=False)
                z[n_obs_before_batch:n_obs_after_batch, :] = (
                    z_batch[:, columns].detach().cpu().numpy())
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
    

    def get_gp_summary(self, orientation=None) -> pd.DataFrame:
        """Return full-precision GP summaries in the model analysis convention.

        ``orientation="raw"`` returns un-oriented weights. Round only for
        presentation; orientation and nonzero counts use full precision.
        """
        return self._gp_summary(orientation=orientation)

    def add_active_gp_scores_to_obs(self, orientation=None, use_cached=False) -> None:
        """Replace GP columns with consistently oriented activities.

        Fresh inference is the safe default for migrating manually edited
        notebooks. Set ``use_cached=True`` only for a known raw latent cache.
        Repeated calls never multiply previously oriented values.
        """
        names = list(self.get_active_gps())
        scores = self.get_gp_activities(names, orientation=orientation, use_cached=use_cached)
        self._write_gp_scores(self.adata, names, scores, orientation)
