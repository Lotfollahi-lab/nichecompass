"""
This module contains the Autotalker model. Different analysis capabilities are
integrated directly into the model API for easy use.
"""

import warnings
from typing import Literal, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import torch
from anndata import AnnData
from scipy.special import erfc

from autotalker.data import SpatialAnnTorchDataset
from autotalker.data import initialize_dataloaders, prepare_data
from autotalker.modules import VGPGAE
from autotalker.train import Trainer
from autotalker.utils import compute_graph_connectivities
from .basemodelmixin import BaseModelMixin


class Autotalker(BaseModelMixin):
    """
    Autotalker model class.

    Parameters
    ----------
    adata:
        AnnData object with raw counts stored in ´adata.layers[counts_key]´, 
        sparse adjacency matrix stored in ´adata.obsp[adj_key]´, gene program
        names stored in ´adata.uns[gp_names_key]´, and binary gene program 
        targets and sources masks stored in ´adata.varm[gp_targets_mask_key]´ 
        and ´adata.varm[gp_sources_mask_key]´ respectively (unless gene program 
        masks are passed explicitly to the model via parameters 
        ´gp_targets_mask´ and ´gp_sources_mask´, in which case this will have
        prevalence).
    counts_key:
        Key under which the raw counts are stored in ´adata.layer´.
    adj_key:
        Key under which the sparse adjacency matrix is stored in ´adata.obsp´.
    gp_names_key:
        Key under which the gene program names are stored in ´adata.uns´.
    active_gp_names_key:
        Key under which the active gene program names will be stored in 
        ´adata.uns´.
    gp_targets_mask_key:
        Key under which the gene program targets mask is stored in ´adata.varm´. 
        This mask will only be used if no ´gp_targets_mask´ is passed explicitly
        to the model.
    gp_sources_mask_key:
        Key under which the gene program sources mask is stored in ´adata.varm´. 
        This mask will only be used if no ´gp_sources_mask´ is passed explicitly
        to the model.
    latent_key:
        Key under which the latent / gene program representation of active gene
        programs will be stored in ´adata.obsm´ after model training.
    condition_key:
        Key under which the conditions are stored in ´adata.obs´.
    cond_embed_key:
        Key under which the conditional embeddings will be stored in
        ´adata.uns´.
    genes_idx_key:
        Key in ´adata.uns´ where the index of a concatenated vector of target
        and source genes that are in the gene program masks are stored.
    recon_adj_key:
        Key in ´adata.obsp´ where the reconstructed adjacency matrix edge
        probabilities will be stored.
    agg_alpha_key:
        Key in ´adata.obsp´ where the attention weights of the gene expression
        node label aggregator will be stored.
    include_edge_recon_loss:
        If `True`, includes the edge reconstruction loss in the loss 
        optimization.
    include_gene_expr_recon_loss:
        If `True`, includes the gene expression reconstruction loss in the loss
        optimization.
    gene_expr_recon_dist:
        The distribution used for gene expression reconstruction. If `nb`, uses
        a negative binomial distribution. If `zinb`, uses a zero-inflated
        negative binomial distribution.
    log_variational:
        If ´True´, transforms x by log(x+1) prior to encoding for numerical 
        stability (not for normalization).
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
        active. More information can be found in 
        ´self.model.get_active_gp_mask()´.
    n_hidden_encoder:
        Number of nodes in the encoder hidden layer.
    conv_layer_encoder:
        Convolutional layer used in the graph encoder.
    encoder_n_attention_heads:
        Only relevant if ´conv_layer_encoder == gatv2conv´. Number of attention
        heads used in the graph encoder.
    dropout_rate_encoder:
        Probability that nodes will be dropped in the encoder during training.
    dropout_rate_graph_decoder:
        Probability that nodes will be dropped in the graph decoder during 
        training.
    gp_targets_mask:
        Gene program targets mask that is directly passed to the model (if not 
        ´None´, this mask will have prevalence over a gene program targets mask
        stored in ´adata.varm[gp_targets_mask_key]´).
    gp_sources_mask:
        Gene program sources mask that is directly passed to the model (if not 
        ´None´, this mask will have prevalence over a gene program sources mask
        stored in ´adata.varm[gp_sources_mask_key]´).
    conditions:
        Condition names to get the right encoding when used after reloading.
    n_addon_gps:
        Number of addon gene programs (i.e. gene programs that are not included
        in masks but can be learned de novo).
    n_cond_embed:
        Number of conditional embedding nodes.
    """
    def __init__(self,
                 adata: AnnData,
                 counts_key: str="counts",
                 adj_key: str="spatial_connectivities",
                 gp_names_key: str="autotalker_gp_names",
                 active_gp_names_key: str="autotalker_active_gp_names",
                 gp_targets_mask_key: str="autotalker_gp_targets",
                 gp_sources_mask_key: str="autotalker_gp_sources",
                 latent_key: str="autotalker_latent",
                 condition_key: Optional[str]=None,
                 cond_embed_key: Optional[str]="autotalker_cond_embed",
                 cond_embed_injection: Optional[list]=["encoder",
                                                       "graph_decoder",
                                                       "gene_expr_decoder"],
                 genes_idx_key: str="autotalker_genes_idx",
                 recon_adj_key: str="autotalker_recon_adj",
                 agg_alpha_key: str="autotalker_agg_alpha",
                 include_edge_recon_loss: bool=True,
                 include_gene_expr_recon_loss: bool=True,
                 gene_expr_recon_dist: Literal["nb", "zinb"]="nb",
                 log_variational: bool=True,
                 node_label_method: Literal[
                    "self",
                    "one-hop-sum",
                    "one-hop-norm",
                    "one-hop-attention"]="one-hop-attention",
                 active_gp_thresh_ratio: float=0.03,
                 n_layers_encoder: int=2,
                 n_hidden_encoder: int=256,
                 conv_layer_encoder: Literal["gcnconv", "gatv2conv"]="gcnconv",
                 encoder_n_attention_heads: int=8,
                 dropout_rate_encoder: float=0.,
                 dropout_rate_graph_decoder: float=0.,
                 gp_targets_mask: Optional[Union[np.ndarray, list]]=None,
                 gp_sources_mask: Optional[Union[np.ndarray, list]]=None,
                 conditions: Optional[list]=None, 
                 n_addon_gps: int=0,
                 n_cond_embed: int=10):
        self.adata = adata
        self.counts_key_ = counts_key
        self.adj_key_ = adj_key
        self.gp_names_key_ = gp_names_key
        self.active_gp_names_key_ = active_gp_names_key
        self.gp_targets_mask_key_ = gp_targets_mask_key
        self.gp_sources_mask_key_ = gp_sources_mask_key
        self.latent_key_ = latent_key
        self.condition_key_ = condition_key
        self.cond_embed_key_ = cond_embed_key
        self.cond_embed_injection_ = cond_embed_injection
        self.genes_idx_key_ = genes_idx_key
        self.recon_adj_key_ = recon_adj_key
        self.agg_alpha_key_ = agg_alpha_key
        self.include_edge_recon_loss_ = include_edge_recon_loss
        self.include_gene_expr_recon_loss_ = include_gene_expr_recon_loss
        self.gene_expr_recon_dist_ = gene_expr_recon_dist
        self.log_variational_ = log_variational
        self.node_label_method_ = node_label_method
        self.active_gp_thresh_ratio_ = active_gp_thresh_ratio
        self.n_input_ = adata.n_vars
        self.n_output_ = adata.n_vars
        if node_label_method != "self":
            self.n_output_ *= 2
        self.n_layers_encoder_ = n_layers_encoder
        self.n_hidden_encoder_ = n_hidden_encoder
        self.conv_layer_encoder_ = conv_layer_encoder
        if conv_layer_encoder == "gatv2conv":
            self.encoder_n_attention_heads_ = encoder_n_attention_heads
        else:
            self.encoder_n_attention_heads_ = 0
        self.dropout_rate_encoder_ = dropout_rate_encoder
        self.dropout_rate_graph_decoder_ = dropout_rate_graph_decoder

        # Retrieve gene program masks
        if gp_targets_mask is None:
            if gp_targets_mask_key in adata.varm:
                gp_targets_mask = adata.varm[gp_targets_mask_key].T
            else:
                raise ValueError("Please explicitly provide a ´gp_targets_mask´"
                                 " to the model or specify an adequate "
                                 "´gp_targets_mask_key´ for your adata object. "
                                 "If you do not want to mask gene expression "
                                 "reconstruction, you can create a mask of 1s "
                                 "that allows all gene program latent nodes "
                                 "to reconstruct all genes by passing a mask "
                                 "created with ´mask = "
                                 "np.ones((n_latent, n_output))´).")
        self.gp_mask_ = torch.tensor(gp_targets_mask, dtype=torch.float32)
        if node_label_method != "self":
            if gp_sources_mask is None:
                if gp_sources_mask_key in adata.varm:
                    gp_sources_mask = adata.varm[gp_sources_mask_key].T
                else:
                    raise ValueError("Please explicitly provide a "
                                     "´gp_sources_mask´ to the model or specify"
                                     " an adequate ´gp_sources_mask_key´ for "
                                     "your adata object.")
            # Horizontally concatenate targets and sources masks
            self.gp_mask_ = torch.cat(
                (self.gp_mask_, torch.tensor(gp_sources_mask, 
                dtype=torch.float32)), dim=1)
        self.n_nonaddon_gps_ = len(self.gp_mask_)
        self.n_addon_gps_ = n_addon_gps
        self.n_cond_embed_ = n_cond_embed
        
        # Retrieve index of genes in gp mask
        self.genes_idx_ = adata.uns[genes_idx_key]

        # Retrieve conditions
        if conditions is None:
            if condition_key is not None:
                self.conditions_ = adata.obs[condition_key].unique().tolist()
            else:
                self.conditions_ = []
        else:
            self.conditions_ = conditions
        
        # Validate counts layer key and counts values
        if counts_key not in adata.layers:
            raise ValueError("Please specify an adequate ´counts_key´. "
                             "By default the raw counts are assumed to be "
                             f"stored in adata.layers['counts'].")
        if include_gene_expr_recon_loss or log_variational:
            if (adata.layers[counts_key] < 0).sum() > 0:
                raise ValueError("Please make sure that "
                                 "´adata.layers[counts_key]´ contains the"
                                 " raw counts (not log library size "
                                 "normalized) if ´include_gene_expr_recon_loss´"
                                 " is ´True´ or ´log_variational´ is ´True´.")

        # Validate adjacency key
        if adj_key not in adata.obsp:
            raise ValueError("Please specify an adequate ´adj_key´. "
                             "By default the adjacency matrix is assumed to be "
                             "stored in adata.obsm['spatial_connectivities'].")

        # Validate gp key
        if gp_names_key not in adata.uns:
            raise ValueError("Please specify an adequate ´gp_names_key´. "
                             "By default the gene program names are assumed to "
                             "be stored in adata.uns['autotalker_gp_names'].")

        # Validate condition key
        if condition_key is not None and condition_key not in adata.obs:
            raise ValueError("Please specify an adequate ´condition_key´. "
                             "By default the conditions are assumed to "
                             "be stored in adata.obs['sample'].")
        
        # Initialize model with Variational Gene Program Graph Autoencoder 
        # neural network module
        self.model = VGPGAE(
            n_input=self.n_input_,
            n_layers_encoder=self.n_layers_encoder_,
            n_hidden_encoder=self.n_hidden_encoder_,
            n_nonaddon_gps=self.n_nonaddon_gps_,
            n_addon_gps=self.n_addon_gps_,
            n_cond_embed=self.n_cond_embed_,
            n_output=self.n_output_,
            gene_expr_decoder_mask=self.gp_mask_,
            genes_idx=self.genes_idx_,
            conditions=self.conditions_,
            conv_layer_encoder=self.conv_layer_encoder_,
            encoder_n_attention_heads=self.encoder_n_attention_heads_,
            dropout_rate_encoder=self.dropout_rate_encoder_,
            dropout_rate_graph_decoder=self.dropout_rate_graph_decoder_,
            include_edge_recon_loss=self.include_edge_recon_loss_,
            include_gene_expr_recon_loss=self.include_gene_expr_recon_loss_,
            gene_expr_recon_dist=self.gene_expr_recon_dist_,
            node_label_method=self.node_label_method_,
            active_gp_thresh_ratio=self.active_gp_thresh_ratio_,
            log_variational=self.log_variational_,
            cond_embed_injection=self.cond_embed_injection_)

        self.is_trained_ = False
        # Store init params for saving and loading
        self.init_params_ = self._get_init_params(locals())

    def train(self,
              n_epochs: int=20,
              n_epochs_all_gps: int=10,
              n_epochs_no_edge_recon: int=0,
              lr: float=0.001,
              weight_decay: float=0.,
              lambda_edge_recon: Optional[float]=0.01,
              lambda_gene_expr_recon: float=0.0033,
              lambda_group_lasso: float=0.,
              lambda_l1_masked: float=0.,
              lambda_l1_addon: float=0.,
              edge_val_ratio: float=0.1,
              node_val_ratio: float=0.1,
              edge_batch_size: int=128,
              node_batch_size: int=16,
              mlflow_experiment_id: Optional[str]=None,
              **trainer_kwargs):
        """
        Train the Autotalker model.
        
        Parameters
        ----------
        n_epochs:
            Number of epochs.
        n_epochs_all_gps:
            Number of epochs during which all gene programs are used for model
            training. After that only active gene programs are retained.
        n_epochs_no_edge_recon:
            Number of epochs without edge reconstruction loss for gene
            expression decoder pretraining.
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
        lambda_group_lasso:
            Lambda (weighting factor) for the group lasso regularization loss of
            gene programs. If ´>0´, this will enforce sparsity of gene programs.
        lambda_l1_masked:
            Lambda (weighting factor) for the L1 regularization loss of genes in
            masked gene programs. If ´>0´, this will enforce sparsity of genes
            in masked gene programs.        
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
            Batch size for the node-level dataloaders.
        mlflow_experiment_id:
            ID of the Mlflow experiment used for tracking training parameters
            and metrics.
        trainer_kwargs:
            Kwargs for the model Trainer.
        """
        self.trainer = Trainer(
            adata=self.adata,
            model=self.model,
            counts_key=self.counts_key_,
            adj_key=self.adj_key_,
            condition_key=self.condition_key_,
            edge_val_ratio=edge_val_ratio,
            node_val_ratio=node_val_ratio,
            edge_batch_size=edge_batch_size,
            node_batch_size=node_batch_size,
            **trainer_kwargs)

        self.trainer.train(n_epochs=n_epochs,
                           n_epochs_no_edge_recon=n_epochs_no_edge_recon,
                           n_epochs_all_gps=n_epochs_all_gps,
                           lr=lr,
                           weight_decay=weight_decay,
                           lambda_edge_recon=lambda_edge_recon,
                           lambda_gene_expr_recon=lambda_gene_expr_recon,
                           lambda_group_lasso=lambda_group_lasso,
                           lambda_l1_masked=lambda_l1_masked,
                           lambda_l1_addon=lambda_l1_addon,
                           mlflow_experiment_id=mlflow_experiment_id)
        
        self.is_trained_ = True
        self.adata.obsm[self.latent_key_], _ = self.get_latent_representation(
           adata=self.adata,
           counts_key=self.counts_key_,
           adj_key=self.adj_key_,
           condition_key=self.condition_key_,
           only_active_gps=True,
           return_mu_std=True)
        self.adata.uns[self.active_gp_names_key_] = self.get_active_gps()

        if len(self.conditions_) > 0:
            self.adata.uns[self.cond_embed_key_] = self.get_cond_embeddings()

        # self.adata.obsp[self.recon_adj_key_] = self.get_recon_edge_probs()

        if self.node_label_method_ == "one-hop-attention":
            self.adata.obsp[self.agg_alpha_key_] = (
                self.get_gene_expr_agg_att_weights(
                    node_batch_size=node_batch_size))

        if mlflow_experiment_id is not None:
            mlflow.log_metric("n_active_gps",
                              len(self.adata.uns[self.active_gp_names_key_]))

    def compute_differential_gp_scores(
            self,
            cat_key: str,
            selected_gps: Optional[Union[str,list]]=None,
            selected_cats: Optional[Union[str,list]]=None,
            gp_scores_weight_normalization: bool=False,
            comparison_cats: Union[str, list]="rest",
            n_sample: int=10000,
            key_added: str="autotalker_differential_gp_scores",
            n_top_up_gps_retrieved: int=10,
            n_top_down_gps_retrieved: int=10,
            seed: int=0,
            adata: Optional[AnnData]=None) -> list:
        """
        Compute differential gene program / latent scores between a category and
        specified comparison categories for all categories in ´selected_cats´
        (by default all categories in ´adata.obs[cat_key]´). Differential gp
        scores are measured through the log Bayes Factor between the hypothesis
        h0 that the (normalized) gene program / latent scores of the category
        under consideration (z0) are higher than the (normalized) gene program /
        latent score of the comparison categories (z1) versus the alternative
        hypothesis h1 that the (normalized) gene program / latent scores of the
        comparison categories (z1) are higher or equal to the (normalized) gene
        program / latent scores of the category under consideration (z0). The
        log Bayes Factors per category are stored in a pandas DataFrame under
        ´adata.uns[key_added]´. The DataFrame also stores p_h0, the probability
        that z0 > z1 and p_h1, the probability that z1 >= z0. The rows are
        ordered by the log Bayes Factor. In addition, the (normalized) gene
        program / latent scores of the ´n_top_up_gps_retrieved´ top upregulated
        gene programs and ´n_top_down_gps_retrieved´ top downregulated gene
        programs will be stored in ´adata.obs´.

        Parts of the implementation are inspired by
        https://github.com/theislab/scarches/blob/master/scarches/models/expimap/expimap_model.py#L429
        (24.11.2022).

        Parameters
        ----------
        cat_key:
            Key under which the categories and comparison categories are stored
            in ´adata.obs´.
        selected_gps:
            List of gene program names for which differential gp scores will be
            computed. If ´None´, uses all active gene programs.
        selected_cats:
            List of category labels for which differential gp scores will be
            computed. If ´None´, uses all category labels from
            ´adata.obs[cat_key]´.
        gp_scores_weight_normalization:
            If ´True´, normalize the gp scores by the nb means gene expression
            decoder weights. If ´False´, normalize the gp scores by the signs of
            the summed nb means gene expression decoder weights.
        comparison_cats:
            Categories used as comparison group. If ´rest´, all categories other
            than the category under consideration are used as comparison group.
        n_sample:
            Number of observations to be drawn from the category and comparison
            categories for the log Bayes Factor computation.
        key_added:
            Key under which the differential gp scores pandas DataFrame is
            stored in ´adata.uns´.
        n_top_up_gps_retrieved:
            Number of top upregulated gene programs which will be returned and
            whose (normalized) gp scores will be stored in ´adata.obs´.
        n_top_down_gps_retrieved:
            Number of top downregulated gene programs which will be returned and
            whose (normalized) gp scores will be stored in ´adata.obs´.
        seed:
            Random seed for reproducible sampling.
        adata:
            AnnData object to be used. If ´None´, uses the adata object stored
            in the model instance.

        Returns
        ----------
        top_unique_gps:
            Names of ´n_top_up_gps_retrieved´ upregulated and
            ´n_top_down_gps_retrieved´ downregulated unique differential gene
            programs across all categories (duplicate gene programs that appear
            for multiple catgories are only considered once).
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
        selected_gps_idx, selected_gps_weights = self.get_gp_data(
            selected_gps=selected_gps,
            adata=adata)

        # Get gp / latent scores for selected gps
        mu, std = self.get_latent_representation(
            adata=adata,
            counts_key=self.counts_key_,
            adj_key=self.adj_key_,
            condition_key=self.condition_key_,
            only_active_gps=False,
            return_mu_std=True)
        mu_selected_gps = mu[:, selected_gps_idx]
        std_selected_gps = std[:, selected_gps_idx]

        # Normalize gp scores using the gene expression negative binomial means
        # decoder weights (if ´gp_scores_weight_normaliztion == True´), and, in
        # addition, correct them for zero inflation probabilities if
        # ´self.gene_expr_recon_dist == zinb´. Alternatively (if
        # ´gp_scores_weight_normalization == False´), just use the signs of the
        # summed gene expression negative binomial means decoder weights for
        # normalization. This normalization only causes a difference if a ´zinb´
        # gene expression reconstruction distribution is used as zero inflation
        # probabilities differ for different observations / cells and, as a
        # result also between a category and the comparison categories. The
        # effect of normalizing mu and std by the gene expression negative
        # binomial means decoder weight is cancelled out in the calculation of
        # the log bayes factors
        if gp_scores_weight_normalization:
            norm_factors = selected_gps_weights # dim: (2 x n_genes,
            # n_selected_gps)
            if self.gene_expr_recon_dist_ == "nb":
                mu_norm_factors = norm_factors.mean(0) # sum over genes; dim:
                # (n_selected_gps,)
                std_norm_factors = np.abs(norm_factors.mean(0)) # sum over
                # genes; dim: (n_selected_gps,); proportional increase of std
                # but no negative std
            elif self.gene_expr_recon_dist_ == "zinb":
                # Get zero inflation probabilities
                _, zi_probs = self.get_gene_expr_dist_params(
                    adata=adata,
                    counts_key=self.counts_key_,
                    adj_key=self.adj_key_)
                non_zi_probs = 1 - zi_probs # dim: (n_obs, 2 x n_genes)
                non_zi_probs_rep = np.repeat(non_zi_probs[:, :, np.newaxis],
                                             len(selected_gps),
                                             axis=2) # dim: (n_obs, 2 x n_genes,
                                             # n_selected_gps)
                norm_factors = np.repeat(norm_factors[np.newaxis, :],
                                         mu.shape[0],
                                         axis=0) # dim: (n_obs, 2 x n_genes,
                                         # n_selected_gps)
                norm_factors *= non_zi_probs_rep
        else:
            gp_weights_sum = selected_gps_weights.sum(0) # sum over genes
            gp_signs = np.zeros_like(gp_weights_sum)
            gp_signs[gp_weights_sum>0] = 1. # keep sign of gp score
            gp_signs[gp_weights_sum<0] = -1. # reverse sign of gp score
            norm_factors = gp_signs # dim: (n_selected_gps,)
            mu_norm_factors = norm_factors
            std_norm_factors = 1 # no negative std

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

        # Compute differential gp scores for all selected categories that are
        # not part of the comparison categories
        scores = []
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

            # Aggregate normalization factors
            if norm_factors.ndim == 1 or norm_factors.ndim == 2:
                mu_norm_factors_cat = mu_norm_factors
                mu_norm_factors_comparison_cat =  mu_norm_factors
                std_norm_factors_cat = std_norm_factors
                std_norm_factors_comparison_cat = std_norm_factors
            elif norm_factors.ndim == 3:
                # Compute mean of normalization factors across genes for the
                # category under consideration and the comparison categories
                # respectively
                mu_norm_factors_cat = norm_factors[cat_mask].mean(1)
                std_norm_factors_cat = np.abs(norm_factors[cat_mask].mean(1))
                mu_norm_factors_comparison_cat = (
                    norm_factors[comparison_cat_mask].mean(1)) # dim:
                    # (n_selected_gps,)
                std_norm_factors_comparison_cat = np.abs(
                    norm_factors[comparison_cat_mask].mean(1)) # dim:
                    # (n_selected_gps,)             

            # Normalize gp scores
            mu_selected_gps_cat = (
                mu_selected_gps[cat_mask] * mu_norm_factors_cat)
            std_selected_gps_cat = (
                std_selected_gps[cat_mask] * std_norm_factors_cat)
            mu_selected_gps_comparison_cat = (
                mu_selected_gps[comparison_cat_mask] *
                mu_norm_factors_comparison_cat)
            std_selected_gps_comparison_cat = (
                std_selected_gps[comparison_cat_mask] *
                std_norm_factors_comparison_cat)

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

            # Store differential gp scores
            zipped = zip(
                selected_gps,
                p_h0,
                p_h1,
                log_bayes_factor)
            cat_scores = [{"category": cat,
                           "gene_program": gp,
                           "p_h0": p_h0,
                           "p_h1": p_h1,
                           "log_bayes_factor": log_bayes_factor}
                          for gp, p_h0, p_h1, log_bayes_factor in zipped]
            for score in cat_scores:
                scores.append(score)

        # Create result dataframe
        scores = pd.DataFrame(scores)
        scores.sort_values(by="log_bayes_factor", ascending=False, inplace=True)
        scores.reset_index(drop=True, inplace=True)
        adata.uns[key_added] = scores

        # Retrieve top unique gps and (normalized) gp scores
        top_unique_gps = []
        if n_top_up_gps_retrieved > 0 or n_top_down_gps_retrieved > 0:
            if mu_norm_factors.ndim == 2:
                mu_norm_factors = mu_norm_factors.mean(0) # mean over genes,
                # dim: (n_selected_gps,)
            elif norm_factors.ndim == 3:
                mu_norm_factors = norm_factors.mean(1) # mean over genes,
                # dim: (n_obs, n_selected_gps)
            mu_selected_gps *= mu_norm_factors # use broadcasting

            # Store ´n_top_up_gps_retrieved´ top upregulated gene program scores
            # in ´adata.obs´
            if n_top_up_gps_retrieved > 0:
                # Get unique top up gene programs while maintaining order
                top_up_gps = scores["gene_program"]
                _, top_up_gps_sort_idx = np.unique(top_up_gps,
                                                   return_index=True)
                top_up_gps = top_up_gps[np.sort(top_up_gps_sort_idx)]
                top_up_gps = top_up_gps[:n_top_up_gps_retrieved].to_list()
                top_up_gps_idx = [selected_gps.index(gp) for gp in top_up_gps]
                for gp, gp_idx in zip(top_up_gps, top_up_gps_idx):
                    adata.obs[gp] = mu_selected_gps[:, gp_idx]
                top_unique_gps.extend(top_up_gps)
            
            # Store ´n_top_down_gps_retrieved´ top downregulated gene program
            # scores in ´adata.obs´
            if n_top_down_gps_retrieved > 0:
                # Get unique top down gene programs while maintaining order
                top_down_gps = scores["gene_program"][::-1]
                top_down_gps.reset_index(inplace=True, drop=True)
                _, top_down_gps_sort_idx = np.unique(top_down_gps,
                                                     return_index=True)
                top_down_gps = top_down_gps[np.sort(top_down_gps_sort_idx)]
                top_down_gps = top_down_gps[:n_top_down_gps_retrieved].to_list()
                top_down_gps_idx = [selected_gps.index(gp) for
                                    gp in top_down_gps]
                for gp, gp_idx in zip(top_down_gps, top_down_gps_idx):
                    adata.obs[gp] = mu_selected_gps[:, gp_idx]
                top_unique_gps.extend(top_down_gps)
        top_unique_gps = list(set(top_unique_gps))
        return top_unique_gps

    def compute_gp_gene_importances(
            self,
            selected_gp: str,
            adata: Optional[AnnData]=None) -> pd.DataFrame:
        """
        Compute gene importances for the genes of a given gene program. Gene
        importances are determined by the normalized weights of the gene
        expression decoder, corrected for gene expression zero inflation in the
        case of ´self.edge_recon_dist == zinb´.

        Parameters
        ----------
        selected_gp:
            Name of the gene program for which the gene importances should be
            retrieved.
        adata:
            AnnData object to be used. If ´None´, uses the adata object stored
            in the model instance.
     
        Returns
        ----------
        gp_gene_importances_df:
            Pandas DataFrame containing genes, sign-corrected gene weights, gene
            importances and an indicator whether the gene belongs to the
            communication source or target, stored in ´gene_entity´.
        """
        self._check_if_trained(warn=True)
        
        if adata is None:
            adata = self.adata

        # Check if selected gene program is active
        active_gps = adata.uns[self.active_gp_names_key_]
        if selected_gp not in active_gps:
            print(f"GP '{selected_gp}' is not an active gene program. "
                  "Continuing anyways.")

        _, gp_weights = self.get_gp_data(selected_gps=selected_gp,
                                         adata=adata)

        # Correct signs of gp weights to be aligned with (normalized) gp scores
        if gp_weights.sum(0) < 0:
            gp_weights *= -1

        if self.gene_expr_recon_dist_ == "zinb":
            # Correct for zero inflation probabilities
            _, zi_probs = self.get_gene_expr_dist_params(
                adata=adata,
                counts_key=self.counts_key_,
                adj_key=self.adj_key_)
            non_zi_probs = 1 - zi_probs
            gp_weights_zi = gp_weights * non_zi_probs.sum(0) # sum over all obs
            # Normalize gp weights to get gene importances
            gp_gene_importances = np.abs(gp_weights_zi / np.abs(gp_weights_zi).sum(0))
        elif self.gene_expr_recon_dist_ == "nb":
            # Normalize gp weights to get gene importances
            gp_gene_importances = np.abs(gp_weights / np.abs(gp_weights).sum(0))

        # Create result dataframe
        gp_gene_importances_df = pd.DataFrame()
        gp_gene_importances_df["gene"] = [gene for gene in
                                          adata.var_names.tolist()] * 2
        gp_gene_importances_df["gene_entity"] = (["target"] *
                                                 len(adata.var_names) +
                                                 ["source"] *
                                                 len(adata.var_names))
        gp_gene_importances_df["gene_weight_sign_corrected"] = gp_weights
        gp_gene_importances_df["gene_importance"] = gp_gene_importances
        gp_gene_importances_df = (gp_gene_importances_df
            [gp_gene_importances_df["gene_importance"] != 0])
        gp_gene_importances_df.sort_values(by="gene_importance",
                                           ascending=False,
                                           inplace=True)
        gp_gene_importances_df.reset_index(drop=True, inplace=True)
        return gp_gene_importances_df

    def compute_latent_graph_connectivities(
            self,
            n_neighbors: int=15,
            mode: Literal["knn", "umap"]="knn",
            seed: int=42,
            adata: Optional[AnnData]=None):
        """
        Compute latent graph connectivities.

        Parameters
        ----------
        n_neighbors:
            Number of neighbors for graph connectivities computation.
        mode:
            Mode to be used for graph connectivities computation.
        seed:
            Random seed for reproducible computation.
        adata:
            AnnData object to be used. If ´None´, uses the adata object stored 
            in the model instance.
        """
        self._check_if_trained(warn=True)

        if adata is None:
            adata = self.adata

        # Validate that latent representation exists
        if self.latent_key_ not in adata.obsm:
            raise ValueError(f"Key '{self.latent_key_}' not found in "
                              "'adata.obsm'. Please make sure to first train "
                              "the model and store the latent representation in"
                              " 'adata.obsm'.")

        # Compute latent connectivities
        sc.pp.neighbors(adata=adata,
                        use_rep=self.latent_key_,
                        n_neighbors=n_neighbors,
                        random_state=seed,
                        key_added="latent")

    def get_gp_data(self,
                    selected_gps: Optional[Union[str, list]]=None,
                    adata: Optional[AnnData]=None
                    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the index of selected gene programs as well as their gene weights of
        the gene expression negative binomial means decoder.

        Parameters:
        ----------
        selected_gps:
            Names of the selected gene programs for which data should be
            retrieved.
        adata:
            AnnData object to be used. If ´None´, uses the adata object stored
            in the model instance.

        Returns:
        ----------
        selected_gps_idx:
            Index of the selected gene programs (dim: n_selected_gps,)
        selected_gp_weights:
            Gene expression decoder gene weights of the selected gene programs
            (dim: (n_genes, n_gps) if ´self.node_label_method == self´ or
            (2 x n_genes, n_gps) otherwise).
        """
        self._check_if_trained(warn=True)
        
        if adata is None:
            adata = self.adata

        # Get selected gps and their index
        all_gps = list(adata.uns[self.gp_names_key_])
        if selected_gps is None:
            selected_gps = all_gps
        elif isinstance(selected_gps, str):
            selected_gps = [selected_gps]
        selected_gps_idx = np.array([all_gps.index(gp) for gp in selected_gps])

        # Get gene weights of selected gps
        gp_weights = self.model.get_gp_weights()
        selected_gps_weights = (gp_weights[:, selected_gps_idx].cpu().detach()
                                .numpy())
        return selected_gps_idx, selected_gps_weights

    def get_cond_embeddings(self) -> np.ndarray:
        """
        Get the conditional embeddings.

        Returns:
        ----------
        cond_embed:
            Conditional embeddings.
        """
        self._check_if_trained(warn=True)
        
        cond_embed = self.model.cond_embedder.weight.cpu().detach().numpy()
        return cond_embed

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

        active_gp_mask = self.model.get_active_gp_mask()
        active_gp_mask = active_gp_mask.detach().cpu().numpy()
        active_gps = self.adata.uns[self.gp_names_key_][active_gp_mask]
        return active_gps

    def get_latent_representation(
            self, 
            adata: Optional[AnnData]=None,
            counts_key: str="counts",
            adj_key: str="spatial_connectivities",
            condition_key: Optional[str]=None,
            only_active_gps: bool=True,
            return_mu_std: bool=False,
            node_batch_size: int=64,
            ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get the latent / gene program representation from a trained model.

        Parameters
        ----------
        adata:
            AnnData object to get the latent representation for. If ´None´, uses
            the adata object stored in the model instance.
        counts_key:
            Key under which the raw counts are stored in ´adata.layer´.
        adj_key:
            Key under which the sparse adjacency matrix is stored in 
            ´adata.obsp´.
        condition_key:
            Key under which the conditions are stored in ´adata.obs´.              
            only_active_gps:
            If ´True´, return only the latent representation of active gps.            
        return_mu_std:
            If `True`, return ´mu´ and ´std´ instead of latent features ´z´.

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

        # Create single dataloader containing entire dataset
        data_dict = prepare_data(
            adata=adata,
            condition_label_encoder=self.model.condition_label_encoder_,
            counts_key=counts_key,
            adj_key=adj_key,
            condition_key=condition_key,
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
            n_gps = (self.n_nonaddon_gps_ + self.n_addon_gps_ )

        # Initialize latent vectors
        if return_mu_std:
            mu = np.empty(shape=(adata.shape[0], n_gps))
            std = np.empty(shape=(adata.shape[0], n_gps))
        else:
            z = np.empty(shape=(adata.shape[0], n_gps))

        # Get latent representation for each batch of the dataloader and put it
        # into latent vectors
        for i, node_batch in enumerate(node_loader):
            n_obs_before_batch = i * node_batch_size
            n_obs_after_batch = n_obs_before_batch + node_batch.batch_size
            node_batch = node_batch.to(device)
            x = node_batch.x
            edge_index = node_batch.edge_index
            conditions = (node_batch.conditions if "conditions" in node_batch
                          else None)
            if self.model.log_variational_:
                x = torch.log(1 + x)

            if return_mu_std:
                mu_batch, std_batch = self.model.get_latent_representation(
                    x=x,
                    edge_index=edge_index,
                    conditions=conditions,
                    only_active_gps=only_active_gps,
                    return_mu_std=True)
                mu[n_obs_before_batch:n_obs_after_batch, :] = (
                    mu_batch[:node_batch.batch_size].detach().cpu().numpy())
                std[n_obs_before_batch:n_obs_after_batch, :] = (
                    std_batch[:node_batch.batch_size].detach().cpu().numpy())
            else:
                z_batch = self.model.get_latent_representation(
                    x=x,
                    edge_index=edge_index,
                    conditions=conditions,
                    only_active_gps=only_active_gps,
                    return_mu_std=False)
                z[n_obs_before_batch:n_obs_after_batch, :] = (
                    z_batch[:node_batch.batch_size].detach().cpu().numpy())
        if return_mu_std:
            return mu, std
        else:
            return z
    
    @torch.no_grad()
    def get_recon_adj(self,      
                      node_batch_size: int=2048,
                      device: Optional[str]=None,
                      edge_thresh: float=None,
                      return_edge_probs: bool=False
                      ) -> Union[sp.csr_matrix, torch.Tensor]:
        """
        Get the reconstructed adjacency matrix (or edge probability matrix if 
        ´return_edge_probs == True´ from a trained Autotalker model.

        Parameters
        ----------
        node_batch_size:
            Batch size for batched decoder forward pass to alleviate memory
            consumption. Only relevant if ´return_edge_probs == False´.
        device:
            Device where the computation will be executed.
        edge_thresh:
            Probability threshold above which edge probabilities lead to a
            reconstructed edge. If ´None´, uses the best balanced accuracy
            threshold determined by the model with the validation dataset
            (stored in ´adata.uns['autotalker_recon_adj_best_acc_threshold']´).
        return_edge_probs:
            If ´True´, return a matrix of edge probabilities instead of the
            reconstructed adjacency matrix. This will require a lot of memory
            as a dense matrix will be returned instead of a sparse matrix.

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
            edge_thresh = (
                self.adata.uns["autotalker_recon_adj_best_acc_threshold"])

        # Get conditional embeddings for each observation
        if (len(self.conditions_) > 0) & \
        ("graph_decoder" in self.cond_embed_injection_):
            if self.cond_embed_key_ not in self.adata.uns:
                raise ValueError("Please first store the conditional embeddings"
                                f" in adata.uns['{self.cond_embed_key_}']. They"
                                " can be retrieved via "
                                "'model.get_cond_embeddings()'.")
            cond_labels = self.adata.obs[self.condition_key_]
            cond_label_encodings = cond_labels.map(
                self.model.condition_label_encoder_).values
            cond_embed = torch.tensor(
                self.adata.uns[self.cond_embed_key_][cond_label_encodings],
                device=device)
        else:
            cond_embed = None
        
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
                    cond_embed=cond_embed,
                    reduced_obs_start_idx=i,
                    reduced_obs_end_idx=i+node_batch_size)
                adj_recon_probs_batch = torch.sigmoid(adj_recon_logits)

                # Convert edge probabilities to edges
                adj_recon_batch = (adj_recon_probs_batch > edge_thresh).long()
                adj_recon_batch = adj_recon_batch.cpu().numpy()
                adj_recon[i:i+node_batch_size, :] = adj_recon_batch
        else:
            adj_recon_logits = self.model.graph_decoder(
                z=z_with_inactive,
                cond_embed=cond_embed)
            adj_recon_probs = torch.sigmoid(adj_recon_logits)

        if device is not None:
            # Move model back to original device
            self.model.to(model_device)

        if not return_edge_probs:
            adj_recon = adj_recon.tocsr(copy=False)
            return adj_recon
        else:
            return adj_recon_probs

    @torch.no_grad()
    def get_gene_expr_agg_att_weights(
            self,      
            node_batch_size: int=64) -> sp.csr_matrix:
        """
        Get the mean attention weights (over all heads) of the gene expression
        node label aggregator. The attention weights indicate how much
        importance each node / observation has attributed to its neighboring
        nodes / observations for the gene expression reconstruction task.

        Parameters
        ----------
        node_batch_size:
            Batch size that is used by the dataloader.

        Returns
        ----------
        agg_alpha:
            A sparse scipy matrix containing the mean attention weights over all
            heads of the gene expression node label aggregator (dim: n_obs x
            n_obs). Row-wise sums will be 1 for each observation. The matrix is
            not symmetric.
        """
        self._check_if_trained(warn=False)
        device = next(self.model.parameters()).device

        if self.node_label_method_ != "one-hop-attention":
            raise ValueError("The node label aggregator attention weights can "
                             " only be retrieved if 'one-hop-attention' has "
                             "been used as node label method.")

        # Initialize global attention weights matrix
        agg_alpha = sp.csr_matrix((len(self.adata), len(self.adata)))

        # Create single dataloader containing entire dataset
        data_dict = prepare_data(
            adata=self.adata,
            condition_label_encoder=self.model.condition_label_encoder_,
            counts_key=self.counts_key_,
            adj_key=self.adj_key_,
            condition_key=self.condition_key_,
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

        # Get attention weights for each node batch of the dataloader and put
        # them into the global attention weights matrix
        for i, node_batch in enumerate(node_loader):
            node_batch = node_batch.to(device)
            n_obs_before_batch = i * node_batch_size
            n_obs_after_batch = n_obs_before_batch + node_batch.batch_size

            _, alpha = (self.model.gene_expr_node_label_aggregator(
                x=node_batch.x,
                edge_index=node_batch.edge_index,
                batch_size=node_batch.batch_size,
                return_attention_weights=True))

            # Get edge index and attention weights of current node batch only
            # (exclude sampled neighbors)
            alpha_edge_index = node_batch.edge_attr[
                (node_batch.edge_attr[:, 1] >= n_obs_before_batch) &
                (node_batch.edge_attr[:, 1] < n_obs_after_batch)]
            alpha = alpha[:alpha_edge_index.shape[0]]

            # Compute mean over attention heads
            mean_alpha = alpha.mean(dim=-1)

            # Insert attention weights from current node batch in global
            # attention weights matrix
            alpha_edge_index = alpha_edge_index.cpu().numpy()
            mean_alpha = mean_alpha.cpu().numpy()
            agg_alpha[alpha_edge_index[:, 1],
                      alpha_edge_index[:, 0]] = mean_alpha
        return agg_alpha
    

    def get_gene_expr_dist_params(
            self, 
            adata: Optional[AnnData]=None,
            counts_key: str="counts",
            adj_key: str="spatial_connectivities",
            ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get the gene expression distribution parameters from a trained model. 
        This is either (´nb_means´, ´zi_probs´) if a zero-inflated negative 
        binomial is used to model gene expression or ´nb_means´ if a negative 
        binomial is used to model gene expression.

        Parameters
        ----------
        adata:
            AnnData object to get the gene expression distribution parameters
            for. If ´None´, uses the adata object stored in the model instance.
        counts_key:
            Key under which the raw counts are stored in ´adata.layer´.    
        adj_key:
            Key under which the sparse adjacency matrix is stored in 
            ´adata.obsp´.       

        Returns
        ----------
        nb_means:
            Expected values of the negative binomial distribution (dim: n_obs x
            n_genes).
        zi_probs:
            Zero-inflation probabilities of the zero-inflated negative binomial
            distribution (dim: n_obs x n_genes).
        """
        self._check_if_trained(warn=False)

        device = next(self.model.parameters()).device
        
        if adata is None:
            adata = self.adata

        dataset = SpatialAnnTorchDataset(adata=adata,
                                         counts_key=counts_key,
                                         adj_key=adj_key)
        x = dataset.x.to(device)
        edge_index = dataset.edge_index.to(device)
        log_library_size = torch.log(x.sum(1)).unsqueeze(1)
        if self.model.log_variational_:
            x = torch.log(1 + x)

        mu, _ = self.model.get_latent_representation(
            x=x,
            edge_index=edge_index,
            only_active_gps=False,
            return_mu_std=True)

        if self.gene_expr_recon_dist_ == "nb":
            nb_means = self.model.get_gene_expr_dist_params(
                z=mu,
                log_library_size=log_library_size)
            nb_means = nb_means.detach().cpu().numpy()
            return nb_means
        if self.gene_expr_recon_dist_ == "zinb":
            nb_means, zi_prob_logits = self.model.get_gene_expr_dist_params(
                z=mu,
                log_library_size=log_library_size)
            zi_probs = torch.sigmoid(zi_prob_logits)
            nb_means = nb_means.detach().cpu().numpy()
            zi_probs = zi_probs.detach().cpu().numpy()
            return nb_means, zi_probs

    def get_gp_summary(self) -> pd.DataFrame:
        """
        Returns
        ----------
        gp_summary_df:
            DataFrame with gene program summary information.

        TO DO: extend to addon gps.
        """
        # Get source and target (sign corrected) gene weights
        _, gp_weights = self.get_gp_data()
        gp_weights_sum = gp_weights.sum(0) # sum over genes
        gp_weights_signs = np.zeros_like(gp_weights_sum)
        gp_weights_signs[gp_weights_sum>0] = 1. # keep sign of gp score
        gp_weights_signs[gp_weights_sum<0] = -1. # reverse sign of gp score
        gp_weights *= gp_weights_signs

        if self.gene_expr_recon_dist_ == "zinb":
            # Correct for zero inflation probabilities
            _, zi_probs = self.get_gene_expr_dist_params(
                adata=self.adata,
                counts_key=self.counts_key_,
                adj_key=self.adj_key_)
            non_zi_probs = 1 - zi_probs
            gp_weights_zi = gp_weights * non_zi_probs.sum(0) # sum over all obs
            # Normalize gp weights to get gene importances
            gp_gene_importances = np.abs(gp_weights_zi / np.abs(gp_weights_zi).sum(0))
        elif self.gene_expr_recon_dist_ == "nb":
            # Normalize gp weights to get gene importances
            gp_gene_importances = np.abs(gp_weights / np.abs(gp_weights).sum(0))

        gp_weights = np.transpose(gp_weights)
        gp_gene_importances = np.transpose(gp_gene_importances)
        gp_source_genes_weights_all_arr = gp_weights[:, int(gp_weights.shape[1]/2):]
        gp_target_genes_weights_all_arr = gp_weights[:, :int(gp_weights.shape[1]/2)]
        gp_source_gene_importances_all_arr = gp_gene_importances[:, int(gp_weights.shape[1]/2):]
        gp_target_gene_importances_all_arr = gp_gene_importances[:, :int(gp_weights.shape[1]/2)]

        # Get source and target genes
        gp_source_genes_mask = np.transpose(
            self.adata.varm[self.gp_sources_mask_key_] != 0)
        gp_target_genes_mask = np.transpose(
            self.adata.varm[self.gp_targets_mask_key_] != 0)
        
        # Add entries to gp mask for addon gps
        if self.n_addon_gps_ > 0:
            addon_gp_source_genes_mask = np.ones((self.n_addon_gps_,
                                                  self.adata.n_vars), dtype=bool)
            addon_gp_target_genes_mask = np.ones((self.n_addon_gps_,
                                                  self.adata.n_vars), dtype=bool)
            gp_source_genes_mask = np.concatenate(
                (gp_source_genes_mask, addon_gp_source_genes_mask), axis=0)
            gp_target_genes_mask = np.concatenate(
                (gp_target_genes_mask, addon_gp_target_genes_mask), axis=0)

        # Get active gp mask
        gp_active_status = np.array(self.model.get_active_gp_mask()).tolist()

        active_gps = list(self.get_active_gps())
        all_gps = list(self.adata.uns[self.gp_names_key_])

        # Collect info for each gp in lists of lists
        gp_names = []
        active_gp_idx = [] # Index among active gene programs
        all_gp_idx = [] # Index among all gene programs
        n_source_genes = []
        n_target_genes = []
        gp_source_genes = []
        gp_target_genes = []
        gp_source_genes_weights = []
        gp_target_genes_weights = []
        gp_source_genes_importances = []
        gp_target_genes_importances = []
        for (gp_name,
             gp_source_genes_idx,
             gp_target_genes_idx,
             gp_source_genes_weights_arr,
             gp_target_genes_weights_arr,
             gp_source_genes_importances_arr,
             gp_target_genes_importances_arr) in zip(
                self.adata.uns[self.gp_names_key_],
                gp_source_genes_mask,
                gp_target_genes_mask,
                gp_source_genes_weights_all_arr,
                gp_target_genes_weights_all_arr,
                gp_source_gene_importances_all_arr,
                gp_target_gene_importances_all_arr):
            gp_names.append(gp_name)
            active_gp_idx.append(active_gps.index(gp_name)
                                 if gp_name in active_gps else np.nan)
            all_gp_idx.append(all_gps.index(gp_name))

            # Sort source genes according to absolute weights
            sorted_source_genes_weights = []
            sorted_source_genes_importances = []
            sorted_source_genes = []
            for _, weights, importances, genes in sorted(zip(
                np.abs(np.around(gp_source_genes_weights_arr[gp_source_genes_idx],
                                 decimals=4)),
                np.around(gp_source_genes_weights_arr[gp_source_genes_idx],
                          decimals=4),
                np.around(gp_source_genes_importances_arr[gp_source_genes_idx],
                          decimals=4),        
                self.adata.var_names[gp_source_genes_idx].tolist()),reverse=True):
                    sorted_source_genes.append(genes)
                    sorted_source_genes_weights.append(weights)
                    sorted_source_genes_importances.append(importances)
            
            # Sort target genes according to absolute weights
            sorted_target_genes_weights = []
            sorted_target_genes_importances = []
            sorted_target_genes = []
            for _, weights, importances, genes in sorted(zip(
                np.abs(np.around(gp_target_genes_weights_arr[gp_target_genes_idx],
                                 decimals=4)),
                np.around(gp_target_genes_weights_arr[gp_target_genes_idx],
                          decimals=4),                 
                np.around(gp_target_genes_importances_arr[gp_target_genes_idx],
                          decimals=4),
                self.adata.var_names[gp_target_genes_idx].tolist()), reverse=True):
                    sorted_target_genes.append(genes)
                    sorted_target_genes_weights.append(weights)
                    sorted_target_genes_importances.append(importances)                    
                
            n_source_genes.append(len(sorted_source_genes))
            n_target_genes.append(len(sorted_target_genes))
            gp_source_genes.append(sorted_source_genes)
            gp_target_genes.append(sorted_target_genes)
            gp_source_genes_weights.append(sorted_source_genes_weights)
            gp_target_genes_weights.append(sorted_target_genes_weights)
            gp_source_genes_importances.append(sorted_source_genes_importances)
            gp_target_genes_importances.append(sorted_target_genes_importances)
   
        gp_summary_df = pd.DataFrame(
            {"gp_name": gp_names,
             "all_gp_idx": all_gp_idx,
             "gp_active": gp_active_status,
             "active_gp_idx": active_gp_idx,
             "n_source_genes": n_source_genes,
             "n_target_genes": n_target_genes,
             "gp_source_genes": gp_source_genes,
             "gp_target_genes": gp_target_genes,
             "gp_source_genes_weights_sign_corrected": gp_source_genes_weights,
             "gp_target_genes_weights_sign_corrected": gp_target_genes_weights,
             "gp_source_genes_importances": gp_source_genes_importances,
             "gp_target_genes_importances": gp_target_genes_importances})

        gp_summary_df["active_gp_idx"] = (
            gp_summary_df["active_gp_idx"].astype("Int64"))
        
        return gp_summary_df