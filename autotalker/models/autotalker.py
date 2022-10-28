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
            **trainer_kwargs)

        self.trainer.train(n_epochs=n_epochs,
                           lr=lr,
                           weight_decay=weight_decay,
                           mlflow_experiment_id=mlflow_experiment_id,)
        
        self.is_trained_ = True


    def get_gp_gene_importances(
            self,
            gp_name: str,
            gp_key: str):
        """
        Get gene importances of a given gene program. Gene importances are 
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


    def calculate_gp_enrichment_scores(
            self,
            cat_key: str,
            gp_key: Optional[str]=None,
            adata: Optional[AnnData]=None,
            comparison_cats: Union[str, list]="rest",
            selected_gps: Optional[list]=None,
            n_sample: int=1000,
            key_added: str="gp_enrichment_scores"):
        """
        Calculate gene program (latent) enrichment scores between a category and 
        comparison categories for multiple categories. The enrichment scores are
        log Bayes Factors between the hypothesis h0 that the gene program / 
        latent score of the category under consideration (z0) is higher than the
        gene program / latent score of the comparison categories (z1) versus the
        alternative hypothesis h1 that the gene program / latent score of the 
        comparison categories (z1) is higher or equal to the gene program / 
        latent score of the category under consideration (z0). The gene program
        enrichment scores (log Bayes Factors) per category are stored in a 
        pandas DataFrame under ´adata.uns[key_added]´. The DataFrame also stores
        p_h0, the probability that z0 > z1 and p_h1, the probability that 
        z1 >= z0. The rows are ordered by the log Bayes Factor. 
        Adapted from 
        https://github.com/theislab/scarches/blob/master/scarches/models/expimap/expimap_model.py#L429.

        Parameters
        ----------
        cat_key:
            Key under which the categories and comparison categories used for
            enrichment score calculation are stored in ´adata.obs´.
        gp_key:
            Key under which a list of all gene programs is stored in ´adata.uns´.
        adata:
            AnnData object to be used for enrichment score calculation. If 
            ´None´, uses the adata object stored in the model.
        comparison_cats:
            Categories used as comparison group. If ´rest´, all categories other
            than the category under consideration are used as comparison group.
        selected_gps:
            List of gene program names to be selected for the enrichment score
            calculation. If ´None´, uses all gene programs.
        n_sample:
            Number of observations to be drawn from the category and comparison
            categories for the enrichment score calculation.
        key_added:
            Key under which the enrichment score pandas DataFrame is stored in 
            ´adata.uns´.           
        """
        if adata is None:
            adata = self.adata

        # Retrieve the category values for each observation and the unique 
        # categories
        cat_values = adata.obs[cat_key]
        cats = cat_values.unique()

        # Validate comparison categories
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

            # Generate random samples of category and comparison categories 
            # observations with equal size
            cat_mask = cat_values == cat
            if comparison_cats == "rest":
                comparison_cat_mask = ~cat_mask
            else:
                comparison_cat_mask = cat_values.isin(comparison_cats)

            cat_idx = np.random.choice(cat_mask.sum(), n_sample)
            comparison_cat_idx = np.random.choice(comparison_cat_mask.sum(), n_sample)

            adata_cat = adata[cat_mask][cat_idx]
            adata_comparison_cat = adata[comparison_cat_mask][comparison_cat_idx]

            # Get gene program (latent) posterior parameters
            mu_cat, std_cat = self.get_latent_representation(
                adata=adata_cat,
                counts_layer_key="counts",
                adj_key="spatial_connectivities",
                return_mu_std=True)
            mu_comparison_cat, std_comparison_cat = self.get_latent_representation(
                adata=adata_comparison_cat,
                counts_layer_key="counts",
                adj_key="spatial_connectivities",
                return_mu_std=True)

            # Align signs of latent values with up- & downregulation
            # directionality
            gp_weights = (self.model.gene_expr_decoder
                          .nb_means_normalized_decoder.masked_l.weight.data)
            if self.n_addon_gps_ > 0:
                gp_weights = torch.cat(
                    [gp_weights, 
                    (self.model.gene_expr_decoder
                    .nb_means_normalized_decoder.addon_l.weight.data)])

            gp_signs = gp_weights.sum(0).cpu().numpy()
            gp_signs[gp_signs>0] = 1.
            gp_signs[gp_signs<0] = -1.
            mu_cat *= gp_signs
            mu_comparison_cat *= gp_signs
    
            # Filter for selected gene programs only
            if selected_gps is not None:
                if gp_key is None:
                    raise ValueError("Please specify a 'gp_key' or set "
                                     "selected_gps to 'None'")
                selected_gps_idx = [adata.uns[gp_key].index(gp) for gp in selected_gps]
                mu_cat = mu_cat[:, selected_gps_idx]
                mu_comparison_cat = mu_comparison_cat[:, selected_gps_idx]
                std_cat = std_cat[:, selected_gps_idx]
                std_comparison_cat = std_comparison_cat[:, selected_gps_idx]
            else:
                selected_gps_idx = np.arange(len(adata.uns[gp_key]))

            # Calculate gene program log Bayes Factors for the category
            to_reduce = (-(mu_cat - mu_comparison_cat) / 
                         np.sqrt(2 * (std_cat**2 + std_comparison_cat**2)))
            to_reduce = 0.5 * erfc(to_reduce)
            p_h0 = np.mean(to_reduce.cpu().numpy(), axis=0)
            p_h1 = 1.0 - p_h0
            epsilon = 1e-12
            log_bayes_factor = np.log(p_h0 + epsilon) - np.log(p_h1 + epsilon)

            zeros_mask = (np.abs(mu_cat).sum(0) == 0) | (np.abs(mu_comparison_cat).sum(0) == 0)
            p_h0[zeros_mask] = 0
            p_h1[zeros_mask] = 0
            log_bayes_factor[zeros_mask] = 0

            zipped = zip(
                [adata.uns[gp_key][i] for i in selected_gps_idx],
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

    def latent_directions(self):
        """
        
        """



        return signs