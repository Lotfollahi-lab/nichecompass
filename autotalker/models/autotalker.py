from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
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
    gp_targets_mask_key:
        Key under which the gene program targets mask is stored in ´adata.varm´. 
        This mask will only be used if no ´gp_targets_mask´ is passed explicitly
        to the model.
    gp_sources_mask_key:
        Key under which the gene program sources mask is stored in ´adata.varm´. 
        This mask will only be used if no ´gp_sources_mask´ is passed explicitly
        to the model.
    latent_key:
        Key under which the latent representation will be stored in ´adata.obsm´
        after model training. 
    include_edge_recon_loss:
        If `True`, include the edge reconstruction loss in the loss 
        optimization.
    include_gene_expr_recon_loss:
        If `True`, include the gene expression reconstruction loss in the loss
        optimization.
    log_variational:
        If ´True´, transform x by log(x+1) prior to encoding for numerical 
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
    n_addon_gps:
        Number of addon gene programs (i.e. gene programs that are not included
        in masks but can be learned de novo).
    """
    def __init__(self,
                 adata: AnnData,
                 counts_key="counts",
                 adj_key: str="spatial_connectivities",
                 gp_names_key: str="autotalker_gp_names",
                 gp_targets_mask_key: str="autotalker_gp_targets",
                 gp_sources_mask_key: str="autotalker_gp_sources",
                 latent_key: str="autotalker_latent",
                 include_edge_recon_loss: bool=True,
                 include_gene_expr_recon_loss: bool=True,
                 log_variational: bool=True,
                 node_label_method: Literal[
                    "self",
                    "one-hop-sum",
                    "one-hop-norm",
                    "one-hop-attention"]="one-hop-attention",
                 n_hidden_encoder: int=256,
                 dropout_rate_encoder: float=0.0,
                 dropout_rate_graph_decoder: float=0.0,
                 gp_targets_mask: Optional[Union[np.ndarray, list]]=None,
                 gp_sources_mask: Optional[Union[np.ndarray, list]]=None,
                 n_addon_gps: int=0):
        self.adata = adata
        self.counts_key_ = counts_key
        self.adj_key_ = adj_key
        self.gp_names_key_ = gp_names_key
        self.gp_targets_mask_key_ = gp_targets_mask_key
        self.gp_sources_mask_key_ = gp_sources_mask_key
        self.latent_key_ = latent_key
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
        self.n_nonaddon_gps_ = len(self.gp_mask_)
        self.n_addon_gps_ = n_addon_gps
        
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
                             "be stored in adata.uns['autotalker_gps'].")
        
        # Initialize model with Variational Gene Program Graph Autoencoder 
        # module
        self.model = VGPGAE(
            n_input=self.n_input_,
            n_hidden_encoder=self.n_hidden_encoder_,
            n_latent=self.n_nonaddon_gps_,
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
              lambda_l1_addon: float=0.,
              lambda_group_lasso: float=0.,
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
            Number of epochs.
        lr:
            Learning rate.
        weight_decay:
            Weight decay (L2 penalty).
        lambda_l1_addon:
            Lambda (weighting) parameter for the L1 regularization of genes in addon
            gene programs.
        lambda_group_lasso:
            Lambda (weighting) parameter for the group lasso regularization of gene
            programs.
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
            counts_key=self.counts_key_,
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
                           lambda_l1_addon=lambda_l1_addon,
                           lambda_group_lasso=lambda_group_lasso
                           mlflow_experiment_id=mlflow_experiment_id)
        
        self.is_trained_ = True

        self.adata.obsm[self.latent_key_] = self.get_latent_representation()

    def compute_differential_gp_scores(
            self,
            cat_key: str,
            adata: Optional[AnnData]=None,
            selected_gps: Optional[Union[str,list]]=None,
            selected_cats: Optional[Union[str,list]]=None,
            gp_scores_weight_normalization: bool=True,
            gp_scores_zi_normalization: bool=True,
            comparison_cats: Union[str, list]="rest",
            n_sample: int=1000,
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
            If ´True´, normalize the gp scores by the nb means gene expression 
            decoder weights.
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
        top_unique_gps:
            Names of ´n_top_up_gps_retrieved´ upregulated and 
            ´n_top_down_gps_retrieved´ downregulated unique differential gene 
            programs (duplicate gene programs that appear for multiple catgories
            are not considered).
        """
        np.random.seed(seed)

        if selected_gps is None:
            selected_gps = adata.uns[self.gp_names_key_]
            selected_gps_idx = np.arange(len(selected_gps))
        else: 
            if isinstance(selected_gps, str):
                selected_gps = [selected_gps]
            selected_gps_idx = [adata.uns[self.gp_names_key_].index(gp) 
                                for gp in selected_gps]

        if adata is None:
            adata = self.adata

        # Get gene program / latent posterior parameters of selected gps
        mu, std = self.get_latent_representation(
            adata=adata,
            counts_key=self.counts_key_,
            adj_key=self.adj_key_,
            return_mu_std=True)
        mu = mu[:, selected_gps_idx].cpu().numpy()
        std = std[:, selected_gps_idx].cpu().numpy()

        # Normalize gp scores using nb means gene expression decoder weights or 
        # signs of summed weights and, if specified, gene expression zero 
        # inflation to accurately reflect up- & downregulation directionalities
        # and strengths across gene programs (naturally the gp scores do not 
        # necessarily correspond to up- & downregulation directionalities and 
        # strengths as nb means gene expression decoder weights are different 
        # for different genes and gene programs and gene expression zero 
        # inflation is different for different observations / cells and genes)
        gp_weights = (self.model.gene_expr_decoder
                      .nb_means_normalized_decoder.masked_l.weight.data)
        if self.n_addon_gps_ > 0:
            gp_weights = torch.cat(
                [gp_weights, 
                (self.model.gene_expr_decoder
                .nb_means_normalized_decoder.addon_l.weight.data)])
        gp_weights = gp_weights[:, selected_gps_idx].cpu().numpy().copy()

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
                counts_key=self.counts_key_,
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
        if selected_cats is None:
            selected_cats = cats
        elif isinstance(selected_cats, str):
            selected_cats = [selected_cats]

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

            if norm_factors.ndim == 1:
                norm_factors_cat = norm_factors_comparison_cat = norm_factors
            elif norm_factors.ndim == 2:
                # Compute mean of normalization factors across genes
                norm_factors_cat = norm_factors_comparison_cat = norm_factors.mean(0)       
            elif norm_factors.ndim == 3:
                # Compute mean of normalization factors across genes for the 
                # category under consideration and the comparison categories 
                # respectively     
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

        # Retrieve top unique gps and (normalized) gene program / latent scores
        top_unique_gps = []
        if n_top_up_gps_retrieved > 0 or n_top_down_gps_retrieved > 0:
            if norm_factors.ndim == 1:
                norm_factors = norm_factors
            elif norm_factors.ndim == 2:
                norm_factors = norm_factors.mean(0) # mean over genes
            elif norm_factors.ndim == 3:
                norm_factors = norm_factors.mean(1) # mean over genes
            mu *= norm_factors

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
                    adata.obs[gp] = mu[:, gp_idx]
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
                top_down_gps_idx = [selected_gps.index(gp) for gp in top_down_gps]
                for gp, gp_idx in zip(top_down_gps, top_down_gps_idx):
                    adata.obs[gp] = mu[:, gp_idx]
                top_unique_gps.extend(top_down_gps)
        return top_unique_gps

    def compute_gp_gene_importances(
            self,
            selected_gp: str,
            adata: Optional[AnnData]=None,
            gene_importances_zi_normalization: bool=True) -> pd.DataFrame:
        """
        Compute gene importances for the genes of a given gene program. Gene
        importances are determined by the normalized weights of the NB means 
        gene expression decoder, optionally corrected for gene expression zero
        inflation.

        Parameters
        ----------
        selected_gp:
            Name of the gene program for which the gene importances should be
            retrieved.
        adata:
            AnnData object to be used. If ´None´, uses the adata object stored 
            in the model.
        gene_importances_zi_normalization:
            If ´True´, normalize the gene importances by the zero inflation 
            probabilities.        
     
        Returns
        ----------
        gp_gene_importances_df:
            Pandas DataFrame containing genes, gene weights, gene importances 
            and an indicator whether the gene belongs to the communication 
            source or target, stored in ´gene_entity´.
        """
        if adata is None:
            adata = self.adata

        # Retrieve NB means gene expression decoder weights
        selected_gp_idx = adata.uns[self.gp_names_key_].index(selected_gp)
        if selected_gp_idx < self.n_nonaddon_gps_: # non-addon gp
            gp_weights = (
                self.model.gene_expr_decoder.nb_means_normalized_decoder
                .masked_l.weight[:, selected_gp_idx].data.cpu().numpy().copy())
        elif selected_gp_idx >= self.n_nonaddon_gps_: # addon gp
            selected_gp_idx -= self.n_nonaddon_gps_
            gp_weights = (
                self.model.gene_expr_decoder.nb_means_normalized_decoder
                .addon_l.weight[:, selected_gp_idx].data.cpu().numpy().copy())

        # Correct signs of gp weights to be aligned with (normalized) gp scores
        if gp_weights.sum(0) < 0:
            gp_weights *= -1
        
        if gene_importances_zi_normalization:
            # Get zero inflation probabilities
            _, zi_probs = self.get_zinb_gene_expr_params(
                adata=adata,
                counts_key=self.counts_key_,
                adj_key=self.adj_key_)
            zi_probs = zi_probs.cpu().numpy()
            non_zi_probs = 1 - zi_probs
            gp_weights_zi = gp_weights * non_zi_probs.sum(0) # sum over all obs / cells
            # Normalize gp weights to get gene importances
            gp_gene_importances = gp_weights_zi / gp_weights_zi.sum(0)
        else:
            # Normalize gp weights to get gene importances
            gp_gene_importances = gp_weights / gp_weights.sum(0)

        gp_gene_importances_df = pd.DataFrame()
        gp_gene_importances_df["gene"] = [gene for gene in 
                                          adata.var_names.tolist()] * 2
        gp_gene_importances_df["gene_entity"] = (["target"] * len(adata.var_names) + 
                                                ["source"] * len(adata.var_names))
        gp_gene_importances_df["gene_weight"] = gp_weights
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
            adata: Optional[AnnData]=None,
            n_neighbors: int=15,
            mode: Literal["knn", "umap"]="knn",
            seed: int=42):
        """
        Compute latent graph connectivities.

        Parameters
        ----------
        adata:
            AnnData object to be used. If ´None´, uses the adata object stored 
            in the model.
        n_neighbors:
            Number of neighbors for graph connectivities computation.
        mode:
            Mode to be used for graph connectivities computation.
        seed:
            Random seed for reproducible computation.
        """
        if adata is None:
            adata = self.adata

        if self.latent_key_ not in adata.obsm:
            raise ValueError(f"Key '{self.latent_key_}' not found in "
                              "'adata.obsm'. Please make sure to first train "
                              "the model and store the latent representation in"
                              " 'adata.obsm'.")

        # Compute latent connectivities
        adata.obsp["latent_connectivities"] = _compute_graph_connectivities(
            adata=adata,
            feature_key=self.latent_key_,
            n_neighbors=n_neighbors,
            mode=mode,
            seed=seed)