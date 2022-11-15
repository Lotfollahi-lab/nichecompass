from typing import Literal, Optional, Union

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from numpy import ndarray
from scipy.special import erfc

from .basemodelmixin import BaseModelMixin
from .vgaemodelmixin import VGAEModelMixin
from autotalker.modules import VGPGAE
from autotalker.train import Trainer
from autotalker.utils import _compute_graph_connectivities


class Autotalker(BaseModelMixin, VGAEModelMixin):
    """
    Autotalker model class.

    Parameters
    ----------
    adata:
        AnnData object with raw counts stored in 
        ´adata.layers[counts_layer_key]´, sparse adjacency matrix stored in 
        ´adata.obsp[adj_key]´ and binary gene program targets and (optionally) 
        sources masks stored in ´adata.varm[gp_targets_mask_key]´ and 
        ´adata.varm[gp_sources_mask_key]´ respectively (unless gene program 
        masks are passed explicitly to the model via parameters 
        ´gp_targets_mask_key´ and ´gp_sources_mask_key´).
    counts_layer_key:
        Key under which the raw counts are stored in ´adata.layer´.
    adj_key:
        Key under which the sparse adjacency matrix is stored in ´adata.obsp´.
    gp_targets_mask_key:
        Key under which the gene program targets mask is stored in ´adata.varm´. 
        This mask will only be used if no ´gp_targets_mask_key´ is passed 
        explicitly to the model.
    gp_sources_mask_key:
        Key under which the gene program sources mask is stored in ´adata.varm´. 
        This mask will only be used if no ´gp_sources_mask_key´ is passed 
        explicitly to the model.    
    include_edge_recon_loss:
        If `True`, include the edge reconstruction loss in the loss 
        optimization of the model.
    include_gene_expr_recon_loss:
        If `True`, include the gene expression reconstruction loss in the 
        loss optimization.
    log_variational:
        If ´True´, transform x by log(x+1) prior to encoding for numerical 
        stability. Not normalization.
    node_label_method:
        Node label method that will be used for gene expression reconstruction. 
        If ´self´, use only the input features of the node itself as node labels
        for gene expression reconstruction. If ´one-hop-sum´, use a 
        concatenation of the node's input features with the sum of the input 
        features of all nodes in the node's one-hop neighborhood. If 
        ´one-hop-norm´, use a concatenation of the node`s input features with
        the node's one-hop neighbors input features normalized as per Kipf, T. 
        N. & Welling, M. Semi-Supervised Classification with Graph Convolutional
        Networks. arXiv [cs.LG] (2016))
    n_hidden_encoder:
        Number of nodes in the encoder hidden layer.
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
    """
    def __init__(self,
                 adata: AnnData,
                 counts_layer_key="counts",
                 adj_key: str="spatial_connectivities",
                 gp_targets_mask_key: str="autotalker_gp_targets",
                 gp_sources_mask_key: str="autotalker_gp_sources",
                 include_edge_recon_loss: bool=True,
                 include_gene_expr_recon_loss: bool=True,
                 log_variational: bool=True,
                 node_label_method: Literal["self",
                                            "one-hop-sum",
                                            "one-hop-norm",
                                            "one-hop-attention"]="one-hop-attention",
                 n_hidden_encoder: int=256,
                 dropout_rate_encoder: float=0.0,
                 dropout_rate_graph_decoder: float=0.0,
                 gp_targets_mask: Optional[Union[ndarray, list]]=None,
                 gp_sources_mask: Optional[Union[ndarray, list]]=None,
                 n_addon_gps: int=0):
        self.adata = adata
        self.counts_layer_key_ = counts_layer_key
        self.adj_key_ = adj_key
        self.gp_targets_mask_key_ = gp_targets_mask_key
        self.gp_sources_mask_key_ = gp_sources_mask_key
        self.include_edge_recon_loss_ = include_edge_recon_loss
        self.include_gene_expr_recon_loss_ = include_gene_expr_recon_loss
        self.log_variational_ = log_variational
        self.node_label_method_ = node_label_method
        self.n_input_ = adata.n_vars
        self.n_output_ = adata.n_vars
        if node_label_method != "self":
            self.n_output_ *= 2
        self.n_hidden_encoder_ = n_hidden_encoder
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
        
        self.n_gps_ = len(self.gp_mask_)
        self.n_addon_gps_ = n_addon_gps
        
        # Validate counts layer key and counts values
        if counts_layer_key not in adata.layers:
            raise ValueError("Please specify an adequate ´counts_layer_key´. "
                             "By default the raw counts are assumed to be "
                             f"stored in adata.layers['counts'].")
        if include_gene_expr_recon_loss or log_variational:
            if (adata.layers[counts_layer_key] < 0).sum() > 0:
                raise ValueError("Please make sure that "
                                 "´adata.layers[counts_layer_key]´ contains the"
                                 " raw counts (not log library size "
                                 "normalized) if ´include_gene_expr_recon_loss´"
                                 " is ´True´ or ´log_variational´ is ´True´.")

        # Validate adjacency key
        if adj_key not in adata.obsp:
            raise ValueError("Please specify an adequate ´adj_key´.")
        
        # Initialize model with module
        self.model = VGPGAE(
            n_input=self.n_input_,
            n_hidden_encoder=self.n_hidden_encoder_,
            n_latent=self.n_gps_,
            n_addon_latent=self.n_addon_gps_,
            n_output=self.n_output_,
            gene_expr_decoder_mask=self.gp_mask_,
            dropout_rate_encoder=self.dropout_rate_encoder_,
            dropout_rate_graph_decoder=self.dropout_rate_graph_decoder_,
            include_edge_recon_loss=self.include_edge_recon_loss_,
            include_gene_expr_recon_loss=self.include_gene_expr_recon_loss_,
            node_label_method=self.node_label_method_,
            log_variational=self.log_variational_)

        self.is_trained_ = False
        # Store init params for saving and loading
        self.init_params_ = self._get_init_params(locals())

    def train(self,
              n_epochs: int=200,
              lr: float=0.01,
              weight_decay: float=0,
              edge_val_ratio: float=0.1,
              edge_test_ratio: float=0.05,
              node_val_ratio: float=0.1,
              edge_batch_size: int=64,
              node_batch_size: int=64,
              mlflow_experiment_id: Optional[str]=None,
              **trainer_kwargs):
        """
        Train the Autotalker model.
        
        Parameters
        ----------
        n_epochs:
            Number of epochs for model training.
        lr:
            Learning rate for model training.
        weight_decay:
            Weight decay (L2 penalty) for model training.
        edge_val_ratio:
            Fraction of the data that is used as validation set on edge-level.
            The rest of the data will be used as training or test set (as 
            defined in edge_test_ratio) on edge-level.
        edge_test_ratio:
            Fraction of the data that is used as test set on edge-level.
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
            counts_layer_key=self.counts_layer_key_,
            adj_key=self.adj_key_,
            node_label_method=self.node_label_method_,
            edge_val_ratio=edge_val_ratio,
            edge_test_ratio=edge_test_ratio,
            node_val_ratio=node_val_ratio,
            node_test_ratio=0.0,
            edge_batch_size=edge_batch_size,
            node_batch_size=node_batch_size,
            **trainer_kwargs)

        self.trainer.train(n_epochs=n_epochs,
                           lr=lr,
                           weight_decay=weight_decay,
                           mlflow_experiment_id=mlflow_experiment_id,)
        
        self.is_trained_ = True


    def add_gps_from_gp_dict_to_adata(
            self,
            gp_dict: dict,
            adata: Optional[AnnData]=None,
            genes_uppercase: bool=True,
            gp_targets_varm_key: str="autotalker_gp_targets",
            gp_sources_varm_key: str="autotalker_gp_sources",
            gp_names_uns_key: str="autotalker_gp_names",
            min_genes_per_gp: int=0,
            max_genes_per_gp: Optional[int]=None):
        """
        Add gene programs defined in a gene program dictionary to an AnnData object
        by converting the gene program lists of gene program target and source genes
        to binary masks and aligning the masks with genes for which gene expression
        is available in the AnnData object. Inspired by
        https://github.com/theislab/scarches/blob/master/scarches/utils/annotations.py#L5.
    
        Parameters
        ----------
        adata:
            AnnData object to which the gene programs will be added. If ´None´, uses
            the adata object stored in the model.
        gp_dict:
            Nested dictionary containing the gene programs with keys being gene 
            program names and values being dictionaries with keys ´targets´ and 
            ´sources´, where ´targets´ contains a list of the names of genes in the
            gene program for the reconstruction of the gene expression of the node
            itself (receiving node) and ´sources´ contains a list of the names of
            genes in the gene program for the reconstruction of the gene expression
            of the node's neighbors (transmitting nodes).
        genes_uppercase:
            If `True`, convert the gene names in adata to uppercase for comparison
            with the gene program dictionary (e.g. if adata contains mouse data).
        gp_targets_varm_key:
            Key in adata.varm where the binary gene program mask for target genes
            of a gene program will be stored (target genes are used for the 
            reconstruction of the gene expression of the node itself (receiving node
            )).
        gp_sources_varm_key:
            Key in adata.varm where the binary gene program mask for source genes
            of a gene program will be stored (source genes are used for the 
            reconstruction of the gene expression of the node'sneighbors 
            (transmitting nodes).
        gp_names_uns_key:
            Key in adata.uns where the gene program names will be stored.
        min_genes_per_gp:
            Minimum number of genes in a gene program inluding both target and 
            source genes that need to be available in the adata (gene expression has
            been probed) for a gene program not to be discarded.
        max_genes_per_gp:
            Maximum number of genes in a gene program including both target and 
            source genes that can be available in the adata (gene expression has 
            been probed) for a gene program not to be discarded.
        """
        if adata is None:
            adata = self.adata
            
        # Retrieve probed genes from adata
        adata_genes = (adata.var_names.str.upper() if genes_uppercase 
                                                   else adata.var_names)
    
        # Create binary gene program masks considering only probed genes
        gp_targets_mask = [[int(gene in gp_genes_dict["targets"]) 
                   for _, gp_genes_dict in gp_dict.items()]
                   for gene in adata_genes]
        gp_targets_mask = np.asarray(gp_targets_mask, dtype="int32")
    
        gp_sources_mask = [[int(gene in gp_genes_dict["sources"]) 
                   for _, gp_genes_dict in gp_dict.items()]
                   for gene in adata_genes]
        gp_sources_mask = np.asarray(gp_sources_mask, dtype="int32")
        
        gp_mask = np.concatenate((gp_sources_mask, gp_targets_mask), axis=0)
    
        # Filter gene programs
        gp_mask_filter = gp_mask.sum(0) > min_genes_per_gp
        if max_genes_per_gp is not None:
            gp_mask_filter &= gp_mask.sum(0) < max_genes_per_gp
        gp_targets_mask = gp_targets_mask[:, gp_mask_filter]
        gp_sources_mask = gp_sources_mask[:, gp_mask_filter]
    
        # Add binary gene program masks to adata.varm
        adata.varm[gp_sources_varm_key] = gp_sources_mask
        adata.varm[gp_targets_varm_key] = gp_targets_mask
    
        # Add gene program names of gene programs that passed filter to adata.uns
        removed_gp_idx = np.where(~gp_mask_filter)[0]
        adata.uns[gp_names_uns_key] = [gp_name for i, (gp_name, _) in 
                              enumerate(gp_dict.items()) if i not in removed_gp_idx]

        self.gp_key_ = gp_names_uns_key


    def compute_differential_gp_scores(
            self,
            cat_key: str,
            adata: Optional[AnnData]=None,
            selected_gps: Optional[Union[str,list]]=None,
            gp_scores_weight_normalization: bool=True,
            gp_scores_zi_normalization: bool=True,
            comparison_cats: Union[str, list]="rest",
            n_sample: int=10000,
            key_added: str="gp_enrichment_scores",
            n_top_up_gps_retrieved: int=10,
            n_top_down_gps_retrieved: int=10,
            seed: int=42) -> list:
        """
        Compute differential gene program / latent scores between a category and 
        specified comparison categories for all categories in 
        ´adata.obs[cat_key]´. Differential gp scores are measured through the 
        log Bayes Factor between the hypothesis h0 that the (normalized) gene 
        program / latent scores of the category under consideration (z0) are 
        higher than the (normalized) gene program / latent score of the 
        comparison categories (z1) versus the alternative hypothesis h1 that the
        (normalized) gene program / latent scores of the comparison categories 
        (z1) are higher or equal to the (normalized) gene program / latent 
        scores of the category under consideration (z0). The log Bayes Factors 
        per category are stored in a pandas DataFrame under 
        ´adata.uns[key_added]´. The DataFrame also stores p_h0, the probability
        that z0 > z1 and p_h1, the probability that z1 >= z0. The rows are 
        ordered by the log Bayes Factor. In addition, the (normalized) gene 
        program / latent scores of the ´n_top_up_gps_retrieved´ top upregulated
        gene programs and ´n_top_down_gps_retrieved´ top downregulated gene 
        programs will be stored in ´adata.obs´.

        Parts of the implementation are inspired by
        https://github.com/theislab/scarches/blob/master/scarches/models/expimap/expimap_model.py#L429.

        Parameters
        ----------
        cat_key:
            Key under which the categories and comparison categories are stored 
            in ´adata.obs´.
        adata:
            AnnData object to be used. If ´None´, uses the adata object stored 
            in the model.
        selected_gps:
            List of gene program names for which differential gp scores will be
            computed. If ´None´, uses all gene programs.
        gp_scores_weight_normalization:
            If ´True´, normalize the gp scores by the nb means decoder weights.
        gp_scores_zi_normalization:
            If ´True´, normalize the gp scores by the zero inflation 
            probabilities.        
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

        Returns
        ----------
        top_gps:
            Names of ´n_top_up_gps_retrieved´ upregulated and 
            ´n_top_down_gps_retrieved´ downregulated differential gene programs.
        """
        np.random.seed(seed)

        if selected_gps is None:
            selected_gps = adata.uns[self.gp_key_]
            selected_gps_idx = np.arange(len(selected_gps))
        else: 
            if isinstance(selected_gps, str):
                selected_gps = [selected_gps]
            selected_gps_idx = [adata.uns[self.gp_key_].index(gp) 
                                for gp in selected_gps]

        if adata is None:
            adata = self.adata

        # Get gene program / latent posterior parameters of selected gps
        mu, std = self.get_latent_representation(
            adata=adata,
            counts_layer_key=self.counts_layer_key_,
            adj_key=self.adj_key_,
            return_mu_std=True)
        mu = mu[:, selected_gps_idx].cpu().numpy()
        std = std[:, selected_gps_idx].cpu().numpy()

        # Normalize gp scores using nb means decoder weights or signs of summed 
        # weights and, if specified, zero inflation to accurately reflect up- &
        # downregulation directionalities and strengths across gene programs 
        # (naturally the gp scores do not necessarily correspond to up- & 
        # downregulation directionalities and strengths as nb means decoder 
        # weights are different for different genes and gene programs and zero 
        # inflation is different for different observations / cells and genes)
        gp_weights = (self.model.gene_expr_decoder
                      .nb_means_normalized_decoder.masked_l.weight.data)
        if self.n_addon_gps_ > 0:
            gp_weights = torch.cat(
                [gp_weights, 
                (self.model.gene_expr_decoder
                .nb_means_normalized_decoder.addon_l.weight.data)])
        gp_weights = gp_weights[:, selected_gps_idx].cpu().numpy()

        if gp_scores_weight_normalization:
            norm_factors = gp_weights
        else:
            gp_weights_sum = gp_weights.sum(0) # sum over genes
            gp_signs = np.zeros_like(gp_weights_sum)
            gp_signs[gp_weights_sum>0] = 1. # keep sign of gp score
            gp_signs[gp_weights_sum<0] = -1. # reverse sign of gp score
            norm_factors = gp_signs

        if gp_scores_zi_normalization:
            # Get zero inflation probabilities
            _, zi_probs = self.get_zinb_gene_expr_params(
                adata=adata,
                counts_layer_key=self.counts_layer_key_,
                adj_key=self.adj_key_)
            zi_probs = zi_probs.cpu().numpy()
            non_zi_probs = 1 - zi_probs
            non_zi_probs_rep = np.repeat(
                non_zi_probs[:, :, np.newaxis],
                len(selected_gps),
                axis=2) # dim: (n_obs, 2 x n_genes, n_selected_gps)
            if norm_factors.ndim == 1:               
               norm_factors = np.repeat(norm_factors[np.newaxis, :],
                                        2*len(adata.var_names),
                                        axis=0)
            norm_factors = np.repeat(norm_factors[np.newaxis, :],
                                     mu.shape[0],
                                     axis=0) # dim: (n_obs, 2 x n_genes, n_selected_gps)
            norm_factors *= non_zi_probs_rep

        # Retrieve category values for each observation as well as all existing 
        # categories
        cat_values = adata.obs[cat_key]
        cats = cat_values.unique()

        # Check specified comparison categories
        if comparison_cats != "rest" and isinstance(comparison_cats, str):
            comparison_cats = [comparison_cats]
        if comparison_cats != "rest" and not set(comparison_cats).issubset(cats):
            raise ValueError("Comparison categories should be 'rest' (for "
                             "comparison with all other categories) or contain "
                             "existing categories")

        # Compute scores for all categories that are not part of the comparison
        # categories
        scores = []
        for cat in cats:
            if cat in comparison_cats:
                continue

            # Filter gp scores and normalization factors for the category under
            # consideration and comparison categories
            cat_mask = cat_values == cat
            if comparison_cats == "rest":
                comparison_cat_mask = ~cat_mask
            else:
                comparison_cat_mask = cat_values.isin(comparison_cats)

            if norm_factors.ndim == 1:
                norm_factors_cat = norm_factors_comparison_cat = norm_factors
            # Compute mean of normalization factors across genes
            elif norm_factors.ndim == 2:
                norm_factors_cat = norm_factors_comparison_cat = norm_factors.mean(0)
            # Compute mean of normalization factors across genes for the category
            # under consideration and the comparison categories respectively            
            elif norm_factors.ndim == 3:
                norm_factors_cat = norm_factors[cat_mask].mean(1)
                norm_factors_comparison_cat = norm_factors[comparison_cat_mask].mean(1)

            mu_cat = mu[cat_mask] * norm_factors_cat
            std_cat = std[cat_mask] * norm_factors_cat
            mu_comparison_cat = (mu[comparison_cat_mask] * 
                                 norm_factors_comparison_cat)
            std_comparison_cat = (std[comparison_cat_mask] *
                                  norm_factors_comparison_cat)

            # Generate random samples of category and comparison categories 
            # observations with equal size
            cat_idx = np.random.choice(cat_mask.sum(),
                                       n_sample)
            comparison_cat_idx = np.random.choice(comparison_cat_mask.sum(),
                                                  n_sample)
            mu_cat_sample = mu_cat[cat_idx]
            std_cat_sample = std_cat[cat_idx]
            mu_comparison_cat_sample = mu_comparison_cat[comparison_cat_idx]
            std_comparison_cat_sample = std_comparison_cat[comparison_cat_idx]

            # Calculate gene program log Bayes Factors for the category
            to_reduce = (-(mu_cat_sample - mu_comparison_cat_sample) / 
                         np.sqrt(2 * (std_cat_sample**2 + 
                                      std_comparison_cat_sample**2)))
            to_reduce = 0.5 * erfc(to_reduce)
            p_h0 = np.mean(to_reduce, axis=0)
            p_h1 = 1.0 - p_h0
            epsilon = 1e-12
            log_bayes_factor = np.log(p_h0 + epsilon) - np.log(p_h1 + epsilon)
            zeros_mask = ((np.abs(mu_cat_sample).sum(0) == 0) | 
                          (np.abs(mu_comparison_cat_sample).sum(0) == 0))
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

        scores = pd.DataFrame(scores)
        scores.sort_values(by="log_bayes_factor", ascending=False, inplace=True)
        scores.reset_index(drop=True, inplace=True)
        adata.uns[key_added] = scores

        # Retrieve top gps and (normalized) gene program / latent scores
        top_gps = []
        if n_top_up_gps_retrieved > 0 or n_top_down_gps_retrieved > 0:
            if norm_factors.ndim == 1:
                norm_factors = norm_factors
            elif norm_factors.ndim == 2:
                norm_factors = norm_factors.mean(0)
            elif norm_factors.ndim == 3:
                norm_factors = norm_factors.mean(1)
            mu *= norm_factors
    
            # Store ´n_top_up_gps_retrieved´ top upregulated gene program scores
            # in ´adata.obs´
            if n_top_up_gps_retrieved > 0:
                top_up_gps = scores["gene_program"][:n_top_up_gps_retrieved].to_list()
                top_up_gps_idx = [selected_gps.index(gp) for gp in top_up_gps]
                for gp, gp_idx in zip(top_up_gps, top_up_gps_idx):
                    adata.obs[gp] = mu[:, gp_idx]
                top_gps.extend(top_up_gps)
            
            # Store ´n_top_down_gps_retrieved´ top downregulated gene program 
            # scores in ´adata.obs´
            if n_top_down_gps_retrieved > 0:
                top_down_gps = scores["gene_program"][-n_top_down_gps_retrieved:].to_list()
                top_down_gps_idx = [selected_gps.index(gp) for gp in top_down_gps]
                for gp, gp_idx in zip(top_down_gps, top_down_gps_idx):
                    adata.obs[gp] = mu[:, gp_idx]
                top_gps.extend(top_down_gps)

        return top_gps


    def compute_gp_gene_importancesx(
            self,
            gp_name: str,
            gp_key: str):
        """
        Compute gene importances of a given gene program. Gene importances are 
        determined by the normalized absolute weights of the gene expression 
        decoder. Adapted from 
        https://github.com/theislab/scarches/blob/master/scarches/models/expimap/expimap_model.py#L305.
        Parameters
        ----------
        gp_name:
            Name of the gene program for which the gene importances should be
            retrieved.
        gp_key:
            Key under which a list of all gene programs is stored in ´adata.uns´.       
        Returns
        ----------
        gp_gene_importances_df:
            Pandas DataFrame with genes stored in ´gene´ and gene expression
            decoder weights stored in ´weights_nb_means_normalized´ and 
            ´weights_zi_prob_logits´ ordered by ´weight_based_importance´, which
            is calculated as an average of the normalized gene expression 
            decoder weights. Genes can belong to the communication source or 
            target as indicated in ´gene_entity´.
        """
        # Retrieve gene program names
        gp_list = list(self.adata.uns[gp_key])
        if len(gp_list) == self.n_gps_:
            if self.n_addon_gps_ > 0:
                gp_list += ["addon_GP_" + str(i) for i in range(
                            self.n_addon_gps_)]
        
        # Validate that all gene programs are contained
        n_latent_w_addon = self.n_gps_ + self.n_addon_gps_
        if len(gp_list) != n_latent_w_addon:
            raise ValueError(f"The number of gene programs ({len(gp_list)}) "
                             "must equal the number of latent dimensions "
                             f"({n_latent_w_addon})!")

        # Retrieve gene-expression-decoder-weight-based importance scores
        gp_idx = gp_list.index(gp_name)
        if gp_idx < self.n_gps_:
            weights_nb_means_normalized = (
                self.model.gene_expr_decoder.nb_means_normalized_decoder
                .masked_l.weight[:, gp_idx].data.cpu().numpy())
            weights_zi_prob_logits = (
                self.model.gene_expr_decoder.zi_prob_logits_decoder
                .masked_l.weight[:, gp_idx].data.cpu().numpy())
        elif gp_idx >= self.n_gps_:
            weights_nb_means_normalized = (
                self.model.gene_expr_decoder.nb_means_normalized_decoder
                .addon_l.weight[:, gp_idx].data.cpu().numpy())
            weights_zi_prob_logits = (
                self.model.gene_expr_decoder.zi_prob_logits_decoder
                .addon_l.weight[:, gp_idx].data.cpu().numpy())
        abs_weights_nb_means_normalized = np.abs(weights_nb_means_normalized)
        abs_weights_zi_prob_logits = np.abs(weights_zi_prob_logits) 
        normalized_abs_weights_nb_means_normalized = (
            abs_weights_nb_means_normalized / 
            abs_weights_nb_means_normalized.sum())
        normalized_abs_weights_zi_prob_logits = (
            abs_weights_zi_prob_logits / 
            abs_weights_zi_prob_logits.sum())
        weight_based_importance = (normalized_abs_weights_nb_means_normalized + 
                                   normalized_abs_weights_zi_prob_logits / 2)
        srt_idx = (np.argsort(weight_based_importance)[::-1]
                   [:(weight_based_importance > 0).sum()])

        # Split into communication target and source idx
        target_srt_idx = srt_idx[srt_idx < len(self.adata.var_names)]
        source_srt_idx = (srt_idx[srt_idx > len(self.adata.var_names)] - 
                          len(self.adata.var_names))

        # Build gene importances df
        gp_gene_importances_df = pd.DataFrame()
        gp_gene_importances_df["gene"] = (
            [gene for gene in self.adata.var_names[target_srt_idx].tolist()] +
            [gene for gene in self.adata.var_names[source_srt_idx].tolist()])
        gp_gene_importances_df["gene_entity"] = (
            ["target" for _ in self.adata.var_names[target_srt_idx].tolist()] +
            ["source" for _ in self.adata.var_names[source_srt_idx].tolist()])
        gp_gene_importances_df["weights_nb_means_normalized"] = (
           weights_nb_means_normalized[srt_idx])
        gp_gene_importances_df["weights_zi_prob_logits"] = (
           weights_zi_prob_logits[srt_idx])
        gp_gene_importances_df["weight_based_importance"] = (
            weight_based_importance[srt_idx])

        return gp_gene_importances_df

    def compute_gp_gene_importances(
            self,
            selected_gps: Union[str, list],
            gp_key: str,
            adata: Optional[AnnData]=None):
        """
        Compute gene importances of a given gene program. Gene importances are 
        determined by the normalized absolute weights of the gene expression 
        decoder. Implementation is inspired by
        https://github.com/theislab/scarches/blob/master/scarches/models/expimap/expimap_model.py#L305.

        Parameters
        ----------
        gp_name:
            Name of the gene program for which the gene importances should be
            retrieved.
        gp_key:
            Key under which a list of all gene programs is stored in ´adata.uns´.       

        Returns
        ----------
        gp_gene_importances_df:
            Pandas DataFrame with genes stored in ´gene´ and gene expression
            decoder weights stored in ´weights_nb_means_normalized´ and 
            ´weights_zi_prob_logits´ ordered by ´weight_based_importance´, which
            is calculated as an average of the normalized gene expression 
            decoder weights. Genes can belong to the communication source or 
            target as indicated in ´gene_entity´.
        """
        if adata is None:
            adata = self.adata

        if isinstance(selected_gps, str):
            selected_gps = [selected_gps]

        # Get latent scores for all observations / cells
        gp_scores, _ = self.get_latent_representation(
            adata=adata,
            counts_layer_key=self.counts_layer_key_,
            adj_key=self.adj_key_,
            return_mu_std=True)

        # categories
        cat_values = adata.obs["celltype_mapped_refined"]
        cat_mask = cat_values == "Cardiomyocytes"

        selected_gps_idx = [adata.uns[gp_key].index(gp) for gp in selected_gps]
        #gp_idx = gp_list.index(gp_name)
        # gp_scores = mu[:, gp_idx]
        selected_gps_scores = gp_scores[cat_mask, selected_gps_idx]
        selected_gps_scores_obs_sum = selected_gps_scores.sum(0).cpu().numpy()
        print(selected_gps_scores_obs_sum)
        #gp_sign 
        #if gp_sign > 0:
        #    gp_sign = 1
        #elif gp_sign < 0:
        #    gp_sign = -1
        #gp_sign = gp_scores_obs_sum[gp_scores_obs_sum>0] = 1.
        #gp_sign = gp_scores_obs_sum[gp_scores_obs_sum<0] = -1.

        # Get zero inflation probabilities
        _, zi_probs = self.get_zinb_gene_expr_params(
            adata=adata,
            counts_layer_key=self.counts_layer_key_,
            adj_key=self.adj_key_)
        zi_probs_obs_mean = zi_probs.mean(0).cpu().numpy()
        non_zi_probs_obs_mean = (1 - zi_probs_obs_mean).reshape(-1, 1)

        gp_weights = (self.model.gene_expr_decoder
                      .nb_means_normalized_decoder.masked_l.weight.data)
        if self.n_addon_gps_ > 0:
            gp_weights = torch.cat(
                [gp_weights, 
                (self.model.gene_expr_decoder
                .nb_means_normalized_decoder.addon_l.weight.data)])
        selected_gps_weights = gp_weights[:, selected_gps_idx].cpu().numpy()
        weight_based_gene_importances = selected_gps_weights
        print(weight_based_gene_importances.shape)

        # print(weights_nb_means_normalized)
        """
        if gp_idx < self.n_gps_:
            weights_nb_means_normalized = (
                self.model.gene_expr_decoder.nb_means_normalized_decoder
                .masked_l.weight[:, gp_idx].data.cpu().numpy())
        elif gp_idx >= self.n_gps_:
            weights_nb_means_normalized = (
                self.model.gene_expr_decoder.nb_means_normalized_decoder
                .addon_l.weight[:, gp_idx].data.cpu().numpy())
        """

        # Adjust weights by zero inflation probabilities and gp sign
        #weight_based_gene_importances = (selected_gps_weights *
        #                                 selected_gps_scores_obs_sum *
        #                                 non_zi_probs_obs_mean)
        
        gp_gene_importances = []
        for i, gp in enumerate(selected_gps):
            # Store gp enrichment scores
            zipped = zip(
                [gene for gene in adata.var_names.tolist()] * 2,
                ["target"] * len(adata.var_names) + ["source"] * len(adata.var_names),
                 weight_based_gene_importances[:, i])
            cat_scores = [{"gene_program": gp,
                           "gene": gene,
                           "gene_entity": gene_entity,
                           "weight_based_gene_importance": weight_based_gene_importance}
                          for gene, gene_entity, weight_based_gene_importance in zipped]
            for dictiii in cat_scores:
                gp_gene_importances.append(dictiii)

        gp_gene_importances_df = pd.DataFrame(gp_gene_importances)
        gp_gene_importances_df.sort_values(by="weight_based_gene_importance",
                                           ascending=False,
                                           inplace=True)
        gp_gene_importances_df.reset_index(drop=True, inplace=True)

        """
        # Split into communication target and source idx
        target_srt_idx = srt_idx[srt_idx < len(self.adata.var_names)]
        source_srt_idx = (srt_idx[srt_idx > len(self.adata.var_names)] - 
                          len(self.adata.var_names))

        # Build gene importances df
        gp_gene_importances_df = pd.DataFrame()
        gp_gene_importances_df["gene"] = (
            [gene for gene in self.adata.var_names[target_srt_idx].tolist()] +
            [gene for gene in self.adata.var_names[source_srt_idx].tolist()])
        gp_gene_importances_df["gene_entity"] = (
            ["target" for _ in self.adata.var_names[target_srt_idx].tolist()] +
            ["source" for _ in self.adata.var_names[source_srt_idx].tolist()])
        gp_gene_importances_df["weight_based_importance"] = (
            weight_based_importance[srt_idx])
        """

        return gp_gene_importances_df


    def compute_latent_graph_connectivities(
            self,
            adata: Optional[AnnData]=None,
            latent_key: str="latent_autotalker_fc_gps",
            n_neighbors: int=15,
            mode: Literal["knn", "umap"]="knn",
            seed: int=42):
        """
        
        """
        if adata is None:
            adata = self.adata

        if latent_key not in adata.obsm:
            raise ValueError(f"Key '{latent_key}' not found in 'adata.obsm'. "
                             "Please make sure to first train the model and "
                             "store the latent representation in 'adata.obsm'.")

        # Compute latent connectivities
        adata.obsp["latent_connectivities"] = _compute_graph_connectivities(
            adata=adata,
            feature_key=latent_key,
            n_neighbors=n_neighbors,
            mode=mode,
            seed=seed)