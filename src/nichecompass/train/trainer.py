"""
This module contains the Trainer to train an NicheCompass model.
"""

import copy
import itertools
import math
import time
import warnings
from collections import defaultdict
from typing import List, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from anndata import AnnData

from nichecompass.data import initialize_dataloaders, prepare_data
from .basetrainermixin import BaseTrainerMixin
from .metrics import eval_metrics, plot_eval_metrics
from .utils import (_cycle_iterable,
                    plot_loss_curves,
                    print_progress,
                    EarlyStopping)


class Trainer(BaseTrainerMixin):
    """
    Trainer class. Encapsulates all logic for NicheCompass model training 
    preparation and model training.
    
    Parts of the implementation are inspired by 
    https://github.com/theislab/scarches/blob/master/scarches/trainers/trvae/trainer.py#L13
    (01.10.2022)
    
    Parameters
    ----------
    adata:
        AnnData object with counts stored in ´adata.layers[counts_key]´ or
        ´adata.X´ depending on ´counts_key´ and sparse adjacency matrix stored
        in ´adata.obsp[adj_key]´.
    adata_atac:
        Additional optional AnnData object with paired spatial ATAC data.
    model:
        An NicheCompass module model instance.
    counts_key:
        Key under which the counts are stored in ´adata.layer´. If ´None´, uses
        ´adata.X´ as counts.
    adj_key:
        Key under which the sparse adjacency matrix is stored in ´adata.obsp´.
    cat_covariates_keys:
        Keys under which the categorical covariates are stored in ´adata.obs´.
    gp_targets_mask_key:
        Key under which the gene program targets mask is stored in ´model.adata.varm´. 
        This mask will only be used if no ´gp_targets_mask´ is passed explicitly
        to the model.
    gp_sources_mask_key:
        Key under which the gene program sources mask is stored in ´model.adata.varm´. 
        This mask will only be used if no ´gp_sources_mask´ is passed explicitly
        to the model.
    edge_val_ratio:
        Fraction of the data that is used as validation set on edge-level. The
        rest of the data will be used as training set on edge-level.
    node_val_ratio:
        Fraction of the data that is used as validation set on node-level. The
        rest of the data will be used as training set on edge-level.
    edge_batch_size:
        Batch size for the edge-level dataloaders.
    node_batch_size:
        Batch size for the node-level dataloaders.
    n_sampled_neighbors:
        Number of neighbors that are sampled during model training from the spatial
        neighborhood graph.
    use_early_stopping:
        If `True`, the EarlyStopping class is used to prevent overfitting.
    reload_best_model:
        If `True`, the best state of the model with respect to the early
        stopping criterion is reloaded at the end of training.
    early_stopping_kwargs:
        Kwargs for the EarlyStopping class.
    use_cuda_if_available:
        If `True`, use cuda if available.
    seed:
        Random seed to get reproducible results.
    monitor:
        If ´True´, the progress of training will be printed after each epoch.
    verbose:
        If ´True´, print out detailed training progress of individual losses.
    """
    def __init__(self,
                 adata: AnnData,
                 model: nn.Module,
                 adata_atac: Optional[AnnData]=None,
                 counts_key: Optional[str]="counts",
                 adj_key: str="spatial_connectivities",
                 cat_covariates_keys: Optional[List[str]]=None,
                 gp_targets_mask_key: str="nichecompass_gp_targets",
                 gp_sources_mask_key: str="nichecompass_gp_sources",                 
                 edge_val_ratio: float=0.1,
                 node_val_ratio: float=0.1,
                 edge_batch_size: int=512,
                 node_batch_size: Optional[int]=None,
                 n_sampled_neighbors: int=-1,
                 use_early_stopping: bool=True,
                 reload_best_model: bool=True,
                 early_stopping_kwargs: Optional[dict]=None,
                 use_cuda_if_available: bool=True,
                 seed: int=0,
                 monitor: bool=True,
                 verbose: bool=False,
                 **kwargs):
        self.adata = adata
        self.adata_atac = adata_atac
        self.model = model
        self.counts_key = counts_key
        self.adj_key = adj_key
        self.cat_covariates_keys = cat_covariates_keys
        if self.cat_covariates_keys is None:
            self.n_cat_covariates = 0
        else:
            self.n_cat_covariates = len(self.cat_covariates_keys)
        self.gp_targets_mask_key = gp_targets_mask_key
        self.gp_sources_mask_key = gp_sources_mask_key
        self.edge_train_ratio_ = 1 - edge_val_ratio
        self.edge_val_ratio_ = edge_val_ratio
        self.node_train_ratio_ = 1 - node_val_ratio
        self.node_val_ratio_ = node_val_ratio
        self.edge_batch_size_ = edge_batch_size
        self.node_batch_size_ = node_batch_size
        self.n_sampled_neighbors_ = n_sampled_neighbors
        self.use_early_stopping_ = use_early_stopping
        self.reload_best_model_ = reload_best_model
        self.early_stopping_kwargs_ = (early_stopping_kwargs if 
            early_stopping_kwargs else {})
        if not "early_stopping_metric" in self.early_stopping_kwargs_:
            if edge_val_ratio > 0 and node_val_ratio > 0:
                self.early_stopping_kwargs_["early_stopping_metric"] = (
                    "val_global_loss")
            else:
                self.early_stopping_kwargs_["early_stopping_metric"] = (
                    "train_global_loss")
        self.early_stopping = EarlyStopping(**self.early_stopping_kwargs_)
        self.seed_ = seed
        self.monitor_ = monitor
        self.verbose_ = verbose
        self.loaders_n_hops_ = kwargs.pop("loaders_n_hops", 1)
        self.grad_clip_value_ = kwargs.pop("grad_clip_value", 0.)
        self.epoch = -1
        self.training_time = 0
        self.optimizer = None
        self.best_epoch = None
        self.best_model_state_dict = None

        print("\n--- INITIALIZING TRAINER ---")
        
        # Set seed and use GPU if available
        np.random.seed(self.seed_)
        if use_cuda_if_available & torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed_)
            torch.manual_seed(self.seed_)
            self.device = torch.device("cuda")
        else:
            torch.manual_seed(self.seed_)
            self.device = torch.device("cpu")
        self.model.to(self.device)

        # Prepare data and get node-level and edge-level training and validation
        # splits
        data_dict = prepare_data(
            adata=adata,
            cat_covariates_label_encoders=self.model.cat_covariates_label_encoders_,
            adata_atac=adata_atac,
            counts_key=self.counts_key,
            adj_key=self.adj_key,
            cat_covariates_keys=self.cat_covariates_keys,
            edge_val_ratio=self.edge_val_ratio_,
            edge_test_ratio=0.,
            node_val_ratio=self.node_val_ratio_,
            node_test_ratio=0.)
        self.node_masked_data = data_dict["node_masked_data"]
        self.edge_train_data = data_dict["edge_train_data"]
        self.edge_val_data = data_dict["edge_val_data"]
        self.n_nodes_train = self.node_masked_data.train_mask.sum().item()
        self.n_nodes_val = self.node_masked_data.val_mask.sum().item()
        self.n_edges_train = self.edge_train_data.edge_label_index.size(1)
        self.n_edges_val = self.edge_val_data.edge_label_index.size(1)
        print(f"Number of training nodes: {self.n_nodes_train}")
        print(f"Number of validation nodes: {self.n_nodes_val}")
        print(f"Number of training edges: {self.n_edges_train}")
        print(f"Number of validation edges: {self.n_edges_val}")

        # Determine node batch size automatically if not specified
        if self.node_batch_size_ is None:
            self.node_batch_size_ = int(self.edge_batch_size_ / math.floor(
                self.n_edges_train / self.n_nodes_train))
        
        print(f"Edge batch size: {edge_batch_size}")
        print(f"Node batch size: {node_batch_size}")

        # Initialize node-level and edge-level dataloaders
        loader_dict = initialize_dataloaders(
            node_masked_data=self.node_masked_data,
            edge_train_data=self.edge_train_data,
            edge_val_data=self.edge_val_data,
            edge_batch_size=self.edge_batch_size_,
            node_batch_size=self.node_batch_size_,
            n_direct_neighbors=self.n_sampled_neighbors_,
            n_hops=self.loaders_n_hops_,
            edges_directed=False,
            neg_edge_sampling_ratio=1.)
        self.edge_train_loader = loader_dict["edge_train_loader"]
        self.edge_val_loader = loader_dict.pop("edge_val_loader", None)
        self.node_train_loader = loader_dict["node_train_loader"]
        self.node_val_loader = loader_dict.pop("node_val_loader", None)

    def train(self,
              n_epochs: int=100,
              n_epochs_all_gps: int=25,
              n_epochs_no_edge_recon: int=0,
              n_epochs_no_cat_covariates_contrastive: int=5,
              lr: float=0.001,
              weight_decay: float=0.,
              lambda_edge_recon: Optional[float]=500000.,
              lambda_cat_covariates_contrastive: Optional[float]=0.,
              contrastive_logits_pos_ratio: Optional[float]=0.125,
              contrastive_logits_neg_ratio: Optional[float]=0.125,
              lambda_gene_expr_recon: float=100.,
              lambda_chrom_access_recon: float=10.,
              lambda_group_lasso: float=0.,
              lambda_l1_masked: float=0.,
              l1_targets_mask: Optional[torch.Tensor]=None,
              l1_sources_mask: Optional[torch.Tensor]=None,
              lambda_l1_addon: float=0.,
              mlflow_experiment_id: Optional[str]=None):
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
        l1_targets_mask:
            Boolean gene program gene mask that is True for all gene program target
            genes to which the L1 regularization loss should be applied (dim:
            n_genes, n_gps).
        l1_sources_mask:
            Boolean gene program gene mask that is True for all gene program source
            genes to which the L1 regularization loss should be applied (dim:
            n_genes, n_gps).
        lambda_l1_addon:
            Lambda (weighting factor) for the L1 regularization loss of genes in
            addon gene programs. If ´>0´, this will enforce sparsity of genes in
            addon gene programs.
        mlflow_experiment_id:
            ID of the mlflow experiment that will be used for tracking.
        """
        self.n_epochs_ = n_epochs
        self.n_epochs_all_gps_ = n_epochs_all_gps
        self.n_epochs_no_edge_recon_ = n_epochs_no_edge_recon
        self.n_epochs_no_cat_covariates_contrastive_ = (
            n_epochs_no_cat_covariates_contrastive)
        self.lr_ = lr
        self.weight_decay_ = weight_decay
        self.lambda_edge_recon_ = lambda_edge_recon
        self.lambda_gene_expr_recon_ = lambda_gene_expr_recon
        self.lambda_chrom_access_recon_ = lambda_chrom_access_recon
        self.lambda_cat_covariates_contrastive_ = (
            lambda_cat_covariates_contrastive)
        self.contrastive_logits_pos_ratio_ = contrastive_logits_pos_ratio
        self.contrastive_logits_neg_ratio_ = contrastive_logits_neg_ratio
        self.lambda_group_lasso_ = lambda_group_lasso
        self.lambda_l1_masked_ = lambda_l1_masked
        self.l1_targets_mask = l1_targets_mask
        self.l1_sources_mask = l1_sources_mask
        self.lambda_l1_addon_ = lambda_l1_addon
        self.mlflow_experiment_id = mlflow_experiment_id

        print("\n--- MODEL TRAINING ---")
        
        # Log hyperparameters
        if self.mlflow_experiment_id is not None:
            for attr, attr_value in self._get_public_attributes().items():
                mlflow.log_param(attr, attr_value)
            self.model.log_module_hyperparams_to_mlflow()

        start_time = time.time()
        self.epoch_logs = defaultdict(list)
        self.model.train()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params,
                                          lr=lr,
                                          weight_decay=weight_decay)

        for self.epoch in range(n_epochs):
            if self.epoch < self.n_epochs_no_edge_recon_:
                self.edge_recon_active = False
            else:
                self.edge_recon_active = True
            if self.epoch < self.n_epochs_all_gps_:
                self.use_only_active_gps = False
            else:
                self.use_only_active_gps = True
            if self.epoch < self.n_epochs_no_cat_covariates_contrastive_:
                self.cat_covariates_contrastive_active = False
            else:
                self.cat_covariates_contrastive_active = True

            self.iter_logs = defaultdict(list)
            self.iter_logs["n_train_iter"] = 0
            self.iter_logs["n_val_iter"] = 0
            
            # Jointly loop through edge- and node-level batches, repeating node-
            # level batches until edge-level batches are complete
            for edge_train_data_batch, node_train_data_batch in zip(
                    self.edge_train_loader,
                    _cycle_iterable(self.node_train_loader)): # itertools.cycle
                                                              # resulted in
                                                              # memory leak
                # Forward pass node-level batch
                node_train_data_batch = node_train_data_batch.to(self.device)
                node_train_model_output = self.model(
                    data_batch=node_train_data_batch,
                    decoder="omics",
                    use_only_active_gps=self.use_only_active_gps)

                # Forward pass edge-level batch
                edge_train_data_batch = edge_train_data_batch.to(self.device)
                edge_train_model_output = self.model(
                    data_batch=edge_train_data_batch,
                    decoder="graph",
                    use_only_active_gps=self.use_only_active_gps)

                # Calculate training loss
                train_loss_dict = self.model.loss(
                    edge_model_output=edge_train_model_output,
                    node_model_output=node_train_model_output,
                    lambda_edge_recon=self.lambda_edge_recon_,
                    lambda_gene_expr_recon=self.lambda_gene_expr_recon_,
                    lambda_chrom_access_recon=self.lambda_chrom_access_recon_,
                    lambda_cat_covariates_contrastive=self.lambda_cat_covariates_contrastive_,
                    contrastive_logits_pos_ratio=self.contrastive_logits_pos_ratio_,
                    contrastive_logits_neg_ratio=self.contrastive_logits_neg_ratio_,
                    lambda_group_lasso=self.lambda_group_lasso_,
                    lambda_l1_masked=self.lambda_l1_masked_,
                    l1_targets_mask=self.l1_targets_mask,
                    l1_sources_mask=self.l1_sources_mask,
                    lambda_l1_addon=self.lambda_l1_addon_,
                    edge_recon_active=self.edge_recon_active,
                    cat_covariates_contrastive_active=self.cat_covariates_contrastive_active)
                
                train_global_loss = train_loss_dict["global_loss"]
                train_optim_loss = train_loss_dict["optim_loss"]

                if self.verbose_:
                    for key, value in train_loss_dict.items():
                        self.iter_logs[f"train_{key}"].append(value.item())
                else:
                    self.iter_logs["train_global_loss"].append(
                        train_global_loss.item())   
                    self.iter_logs["train_optim_loss"].append(
                        train_optim_loss.item())
                self.iter_logs["n_train_iter"] += 1
                # Optimize for training loss
                self.optimizer.zero_grad()
                
                train_optim_loss.backward()
                # Clip gradients
                if self.grad_clip_value_ > 0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(),
                                                    self.grad_clip_value_)
                self.optimizer.step()

            # Validate model
            if (self.edge_val_loader is not None and 
                self.node_val_loader is not None):
                    self.eval_epoch()
            elif (self.edge_val_loader is None and 
            self.node_val_loader is not None):
                warnings.warn("You have specified a node validation set but no "
                              "edge validation set. Skipping validation...")
            elif (self.edge_val_loader is not None and 
            self.node_val_loader is None):
                warnings.warn("You have specified an edge validation set but no"
                              " node validation set. Skipping validation...")
    
            # Convert iteration level logs into epoch level logs
            for key in self.iter_logs:
                if key.startswith("train"):
                    self.epoch_logs[key].append(
                        np.array(self.iter_logs[key]).sum() / 
                        self.iter_logs["n_train_iter"])
                if key.startswith("val"):
                    self.epoch_logs[key].append(
                        np.array(self.iter_logs[key]).sum() /
                        self.iter_logs["n_val_iter"])

            # Monitor epoch level logs
            if self.monitor_:
                print_progress(self.epoch, self.epoch_logs, self.n_epochs_)

            # Check early stopping
            if self.use_early_stopping_:
                if self.is_early_stopping():
                    break

        # Track training time and load best model
        self.training_time += (time.time() - start_time)
        minutes, seconds = divmod(self.training_time, 60)
        print(f"Model training finished after {int(minutes)} min {int(seconds)}"
               " sec.")
        if self.best_model_state_dict is not None and self.reload_best_model_:
            print("Using best model state, which was in epoch "
                  f"{self.best_epoch + 1}.")
            self.model.load_state_dict(self.best_model_state_dict)

        self.model.eval()

        """
        # Track losses and eval metrics
        losses = {"train_global_loss": self.epoch_logs["train_global_loss"],
                  "train_optim_loss": self.epoch_logs["train_optim_loss"],
                  "val_global_loss": self.epoch_logs["val_global_loss"],
                  "val_optim_loss": self.epoch_logs["val_optim_loss"]}
        val_eval_metrics_over_epochs = {
            "auroc": self.epoch_logs["val_auroc_score"],
            "auprc": self.epoch_logs["val_auprc_score"],
            "best_acc": self.epoch_logs["val_best_acc_score"],
            "best_f1": self.epoch_logs["val_best_f1_score"]}

        fig = plot_loss_curves(losses)
        if self.mlflow_experiment_id is not None:
            mlflow.log_figure(fig, "loss_curves.png")
        fig = plot_eval_metrics(val_eval_metrics_over_epochs) 
        if self.mlflow_experiment_id is not None:
            mlflow.log_figure(fig, "val_eval_metrics.png")
        """

        # Calculate after training validation metrics
        if self.edge_val_loader is not None:
            self.eval_end()

    @torch.no_grad()
    def eval_epoch(self):
        """
        Epoch evaluation logic of NicheCompass model used during training.
        """
        self.model.eval()

        edge_recon_probs_val_accumulated = np.array([])
        edge_recon_labels_val_accumulated = np.array([])
        edge_same_cat_covariates_cat_val_accumulated = [
            np.array([]) for _ in range(self.n_cat_covariates)]
        edge_incl_val_accumulated = np.array([])

        # Jointly loop through edge- and node-level batches, repeating node-
        # level batches until edge-level batches are complete
        for edge_val_data_batch, node_val_data_batch in zip(
                self.edge_val_loader, _cycle_iterable(self.node_val_loader)):
            # Forward pass node level batch
            node_val_data_batch = node_val_data_batch.to(self.device)
            node_val_model_output = self.model(
                data_batch=node_val_data_batch,
                decoder="omics",
                use_only_active_gps=self.use_only_active_gps)

            # Forward pass edge level batch
            edge_val_data_batch = edge_val_data_batch.to(self.device)
            edge_val_model_output = self.model(
                data_batch=edge_val_data_batch,
                decoder="graph",
                use_only_active_gps=self.use_only_active_gps)

            # Calculate validation loss
            val_loss_dict = self.model.loss(
                    edge_model_output=edge_val_model_output,
                    node_model_output=node_val_model_output,
                    lambda_edge_recon=self.lambda_edge_recon_,
                    lambda_gene_expr_recon=self.lambda_gene_expr_recon_,
                    lambda_chrom_access_recon=self.lambda_chrom_access_recon_,
                    lambda_cat_covariates_contrastive=self.lambda_cat_covariates_contrastive_,
                    contrastive_logits_pos_ratio=self.contrastive_logits_pos_ratio_,
                    contrastive_logits_neg_ratio=self.contrastive_logits_neg_ratio_,
                    lambda_group_lasso=self.lambda_group_lasso_,
                    lambda_l1_masked=self.lambda_l1_masked_,
                    l1_targets_mask=self.l1_targets_mask,
                    l1_sources_mask=self.l1_sources_mask,
                    lambda_l1_addon=self.lambda_l1_addon_,
                    edge_recon_active=True)

            val_global_loss = val_loss_dict["global_loss"]
            val_optim_loss = val_loss_dict["optim_loss"]
            if self.verbose_:
                for key, value in val_loss_dict.items():
                    self.iter_logs[f"val_{key}"].append(value.item())
            else:
                self.iter_logs["val_global_loss"].append(val_global_loss.item())
                self.iter_logs["val_optim_loss"].append(val_optim_loss.item())  
            self.iter_logs["n_val_iter"] += 1
            
            # Calculate evaluation metrics
            edge_recon_probs_val = torch.sigmoid(
                edge_val_model_output["edge_recon_logits"])
            edge_recon_labels_val = edge_val_model_output["edge_recon_labels"]
            edge_same_cat_covariates_cat_val = edge_val_model_output["edge_same_cat_covariates_cat"]
            edge_incl_val = edge_val_model_output["edge_incl"]
            edge_recon_probs_val_accumulated = np.append(
                edge_recon_probs_val_accumulated,
                edge_recon_probs_val.detach().cpu().numpy())
            edge_recon_labels_val_accumulated = np.append(
                edge_recon_labels_val_accumulated,
                edge_recon_labels_val.detach().cpu().numpy())
            if edge_same_cat_covariates_cat_val is not None:
                for i, edge_same_cat_covariate_cat_val in enumerate(edge_same_cat_covariates_cat_val):
                    edge_same_cat_covariates_cat_val_accumulated[i] = np.append(
                        edge_same_cat_covariates_cat_val_accumulated[i],
                        edge_same_cat_covariate_cat_val.detach().cpu().numpy())
            if edge_incl_val is not None:
                edge_incl_val_accumulated = np.append(
                    edge_incl_val_accumulated,
                    edge_incl_val.detach().cpu().numpy())
            else:
                edge_same_cat_covariates_cat_val_accumulated = None
                edge_incl_val_accumulated = None
        val_eval_dict = eval_metrics(
            edge_recon_probs=edge_recon_probs_val_accumulated,
            edge_labels=edge_recon_labels_val_accumulated,
            edge_same_cat_covariates_cat=edge_same_cat_covariates_cat_val_accumulated,
            edge_incl=edge_incl_val_accumulated)
        if self.verbose_:
            self.epoch_logs["val_auroc_score"].append(
                val_eval_dict["auroc_score"])
            self.epoch_logs["val_auprc_score"].append(
                val_eval_dict["auprc_score"])
            self.epoch_logs["val_best_acc_score"].append(
                val_eval_dict["best_acc_score"])
            self.epoch_logs["val_best_f1_score"].append(
                val_eval_dict["best_f1_score"])
        
        self.model.train()

    @torch.no_grad()
    def eval_end(self):
        """
        End evaluation logic of NicheCompass model used after model training.
        """
        self.model.eval()

        # Get edge-level ground truth and predictions
        edge_recon_probs_val_accumulated = np.array([])
        edge_recon_labels_val_accumulated = np.array([])
        edge_same_cat_covariates_cat_val_accumulated = [
            np.array([]) for _ in range(self.n_cat_covariates)]
        edge_incl_val_accumulated = np.array([])
        for edge_val_data_batch in self.edge_val_loader:
            edge_val_data_batch = edge_val_data_batch.to(self.device)

            edge_val_model_output = self.model(
                data_batch=edge_val_data_batch,
                decoder="graph",
                use_only_active_gps=True)
    
            # Calculate evaluation metrics
            edge_recon_probs_val = torch.sigmoid(
                edge_val_model_output["edge_recon_logits"])
            edge_recon_labels_val = edge_val_model_output["edge_recon_labels"]
            edge_same_cat_covariates_cat_val = edge_val_model_output["edge_same_cat_covariates_cat"]
            edge_incl_val = edge_val_model_output["edge_incl"]
            edge_recon_probs_val_accumulated = np.append(
                edge_recon_probs_val_accumulated,
                edge_recon_probs_val.detach().cpu().numpy())
            edge_recon_labels_val_accumulated = np.append(
                edge_recon_labels_val_accumulated,
                edge_recon_labels_val.detach().cpu().numpy())
            if edge_same_cat_covariates_cat_val is not None:
                for i, edge_same_cat_covariate_cat_val in enumerate(edge_same_cat_covariates_cat_val):
                    edge_same_cat_covariates_cat_val_accumulated[i] = np.append(
                        edge_same_cat_covariates_cat_val_accumulated[i],
                        edge_same_cat_covariate_cat_val.detach().cpu().numpy())
            if edge_incl_val is not None:
                edge_incl_val_accumulated = np.append(
                    edge_incl_val_accumulated,
                    edge_incl_val.detach().cpu().numpy())
            else:
                edge_same_cat_covariates_cat_val_accumulated = None
                edge_incl_val_accumulated = None

        # Get node-level ground truth and predictions
        omics_pred_dict_val_accumulated = {}
        for modality in self.model.modalities_:
            for entity in ["target", "source"]:
                omics_pred_dict_val_accumulated[f"{entity}_{modality}_preds"] = np.array([])
                omics_pred_dict_val_accumulated[f"{entity}_{modality}"] = np.array([])
        for node_val_data_batch in self.node_val_loader:
            node_val_data_batch = node_val_data_batch.to(self.device)

            node_val_model_output = self.model(
                data_batch=node_val_data_batch,
                decoder="omics",
                use_only_active_gps=True)

            for modality in self.model.modalities_:
                for entity in ["target", "source"]:
                    omics_pred_dict_val_accumulated[f"{entity}_{modality}_preds"] = np.append(
                        omics_pred_dict_val_accumulated[f"{entity}_{modality}_preds"],
                        node_val_model_output[f"{entity}_{modality}_nb_means"].detach().cpu().numpy())
                    omics_pred_dict_val_accumulated[f"{entity}_{modality}"] = np.append(
                        omics_pred_dict_val_accumulated[f"{entity}_{modality}"],
                        node_val_model_output["node_labels"][f"{entity}_{modality}"].detach().cpu().numpy())

        val_eval_dict = eval_metrics(
            edge_recon_probs=edge_recon_probs_val_accumulated,
            edge_labels=edge_recon_labels_val_accumulated,
            edge_same_cat_covariates_cat=edge_same_cat_covariates_cat_val_accumulated,
            edge_incl=edge_incl_val_accumulated,
            omics_pred_dict=omics_pred_dict_val_accumulated)
        print("\n--- MODEL EVALUATION ---")
        print(f"val AUROC score: {val_eval_dict['auroc_score']:.4f}")
        print(f"val AUPRC score: {val_eval_dict['auprc_score']:.4f}")
        print(f"val best accuracy score: {val_eval_dict['best_acc_score']:.4f}")
        print(f"val best F1 score: {val_eval_dict['best_f1_score']:.4f}")
        for modality in self.model.modalities_:
            for entity in ["target", "source"]:
                print(f"val {entity} {modality} MSE score: "
                      f"{val_eval_dict[f'{entity}_{modality}_mse_score']:.4f}")
        for i in range(self.n_cat_covariates):
            if f"cat_covariate{i}_mean_sim_diff" in val_eval_dict.keys():
                print(f"Val cat covariate{i} mean sim diff: "
                      f"{val_eval_dict[f'cat_covariate{i}_mean_sim_diff']:.4f}")
            
        # Log evaluation metrics
        if self.mlflow_experiment_id is not None:
            for key, value in val_eval_dict.items():
                mlflow.log_metric(f"val_{key}", value)

    def is_early_stopping(self) -> bool:
        """
        Check whether to apply early stopping, update learning rate and save 
        best model state.

        Returns
        ----------
        stop_training:
            If `True`, stop NicheCompass model training.
        """
        early_stopping_metric = self.early_stopping.early_stopping_metric
        current_metric = self.epoch_logs[early_stopping_metric][-1]
        if self.early_stopping.update_state(current_metric):
            self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
            self.best_epoch = self.epoch

        continue_training, reduce_lr = self.early_stopping.step(current_metric)
        if reduce_lr:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.early_stopping.lr_factor
            print(f"New learning rate is {param_group['lr']}.\n")
        stop_training = not continue_training
        return stop_training