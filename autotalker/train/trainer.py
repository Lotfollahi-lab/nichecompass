"""
This module contains the Trainer to train an Autotalker model.
"""

import copy
import itertools
import time
import warnings
from collections import defaultdict
from typing import Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from anndata import AnnData

from autotalker.data import initialize_dataloaders, prepare_data
from .basetrainermixin import BaseTrainerMixin
from .metrics import eval_metrics, plot_eval_metrics
from .utils import (_cycle_iterable,
                    plot_loss_curves,
                    print_progress,
                    EarlyStopping)


class Trainer(BaseTrainerMixin):
    """
    Trainer class. Encapsulates all logic for Autotalker model training 
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
        An Autotalker module model instance.
    counts_key:
        Key under which the counts are stored in ´adata.layer´. If ´None´, uses
        ´adata.X´ as counts.
    adj_key:
        Key under which the sparse adjacency matrix is stored in ´adata.obsp´.
    condition_key:
        Key under which the conditions are stored in ´adata.obs´.    
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
    use_early_stopping:
        If `True`, the EarlyStopping class is used to prevent overfitting.
    reload_best_model:
        If `True`, the best state of the model with respect to the early
        stopping criterion is reloaded at the end of training.
    early_stopping_kwargs:
        Kwargs for the EarlyStopping class.
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
                 condition_key: Optional[str]=None,
                 edge_val_ratio: float=0.1,
                 node_val_ratio: float=0.1,
                 edge_batch_size: int=128,
                 node_batch_size: int=16,
                 use_early_stopping: bool=True,
                 reload_best_model: bool=True,
                 early_stopping_kwargs: Optional[dict]=None,
                 seed: int=0,
                 monitor: bool=True,
                 verbose: bool=False,
                 **kwargs):
        self.adata = adata
        self.adata_atac = adata_atac
        self.model = model
        self.counts_key = counts_key
        self.adj_key = adj_key
        self.condition_key = condition_key
        self.edge_train_ratio_ = 1 - edge_val_ratio
        self.edge_val_ratio_ = edge_val_ratio
        self.node_train_ratio_ = 1 - node_val_ratio
        self.node_val_ratio_ = node_val_ratio
        self.edge_batch_size_ = edge_batch_size
        self.node_batch_size_ = node_batch_size
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
        self.loaders_n_direct_neighbors_ = kwargs.pop(
            "loaders_n_direct_neighbors", -1)
        self.loaders_n_hops_ = kwargs.pop("loaders_n_hops", 2)
        self.grad_clip_value_ = kwargs.pop("grad_clip_value", 0.)
        self.epoch = -1
        self.training_time = 0
        self.optimizer = None
        self.best_epoch = None
        self.best_model_state_dict = None

        print("\n--- INITIALIZING TRAINER ---")
        
        # Set seed and use GPU if available
        torch.manual_seed(self.seed_)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed_)
            self.model.cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "cpu")

        # Prepare data and get node-level and edge-level training and validation
        # splits
        data_dict = prepare_data(
            adata=adata,
            condition_label_encoder=self.model.condition_label_encoder_,
            adata_atac=adata_atac,
            counts_key=self.counts_key,
            adj_key=self.adj_key,
            condition_key=self.condition_key,
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
        
        # Initialize node-level and edge-level dataloaders
        loader_dict = initialize_dataloaders(
            node_masked_data=self.node_masked_data,
            edge_train_data=self.edge_train_data,
            edge_val_data=self.edge_val_data,
            edge_batch_size=self.edge_batch_size_,
            node_batch_size=self.node_batch_size_,
            n_direct_neighbors=self.loaders_n_direct_neighbors_,
            n_hops=self.loaders_n_hops_,
            edges_directed=False,
            neg_edge_sampling_ratio=1.)
        self.edge_train_loader = loader_dict["edge_train_loader"]
        self.edge_val_loader = loader_dict.pop("edge_val_loader", None)
        self.node_train_loader = loader_dict["node_train_loader"]
        self.node_val_loader = loader_dict.pop("node_val_loader", None)

    def train(self,
              n_epochs: int=40,
              n_epochs_all_gps: int=20,
              n_epochs_no_edge_recon: int=0,
              n_epochs_no_cond_contrastive: int=1,
              lr: float=0.001,
              weight_decay: float=0.,
              lambda_edge_recon: Optional[float]=10.,
              lambda_cond_contrastive: Optional[float]=10.,
              contrastive_logits_ratio: Optional[float]=0.1,
              lambda_gene_expr_recon: float=0.01,
              lambda_group_lasso: float=0.,
              lambda_l1_masked: float=0.,
              lambda_l1_addon: float=0.,
              mlflow_experiment_id: Optional[str]=None):
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
        lambda_cond_contrastive:
            Lambda (weighting factor) for the conditional contrastive loss. If
            ´>0´, this will enforce observations from different conditions with
            very similar latent representations to become more similar and 
            observations with different latent representations to become more
            different.
        contrastive_logits_ratio:
            Ratio for determining the contrastive logits for the conditional
            contrastive loss. The top (´contrastive_logits_ratio´ * 100)% logits
            of sampled negative edges with nodes from different conditions serve
            as positive labels for the contrastive loss and the bottom
            (´contrastive_logits_ratio´ * 100)% logits of sampled negative edges
            with nodes from different conditions serve as negative labels.
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
        mlflow_experiment_id:
            ID of the mlflow experiment that will be used for tracking.
        """
        self.n_epochs_ = n_epochs
        self.n_epochs_all_gps_ = n_epochs_all_gps
        self.n_epochs_no_edge_recon_ = n_epochs_no_edge_recon
        self.n_epochs_no_cond_contrastive_ = n_epochs_no_cond_contrastive
        self.lr_ = lr
        self.weight_decay_ = weight_decay
        self.lambda_edge_recon_ = lambda_edge_recon
        self.lambda_gene_expr_recon_ = lambda_gene_expr_recon
        self.lambda_cond_contrastive_ = lambda_cond_contrastive
        self.contrastive_logits_ratio_ = contrastive_logits_ratio
        self.lambda_group_lasso_ = lambda_group_lasso
        self.lambda_l1_masked_ = lambda_l1_masked
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
            if self.epoch < self.n_epochs_no_cond_contrastive_:
                self.cond_contrastive_active = False
            else:
                self.cond_contrastive_active = True

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
                    lambda_cond_contrastive=self.lambda_cond_contrastive_,
                    contrastive_logits_ratio=self.contrastive_logits_ratio_,
                    lambda_group_lasso=self.lambda_group_lasso_,
                    lambda_l1_masked=self.lambda_l1_masked_,
                    lambda_l1_addon=self.lambda_l1_addon_,
                    edge_recon_active=self.edge_recon_active,
                    cond_contrastive_active=self.cond_contrastive_active)
                
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

        """
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
        Epoch evaluation logic of Autotalker model used during training.
        """
        self.model.eval()

        edge_recon_probs_val_accumulated = np.array([])
        edge_recon_labels_val_accumulated = np.array([])

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
                    lambda_cond_contrastive=self.lambda_cond_contrastive_,
                    contrastive_logits_ratio=self.contrastive_logits_ratio_,
                    lambda_group_lasso=self.lambda_group_lasso_,
                    lambda_l1_masked=self.lambda_l1_masked_,
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
            if edge_val_model_output["edge_same_condition_labels"] is not None:
                # only keep edge labels from same condition for eval
                edge_recon_probs_val = edge_recon_probs_val[
                    edge_val_model_output["edge_same_condition_labels"]]
                edge_recon_labels_val = edge_recon_labels_val[
                    edge_val_model_output["edge_same_condition_labels"]]
            edge_recon_probs_val_accumulated = np.append(
                edge_recon_probs_val_accumulated,
                edge_recon_probs_val.detach().cpu().numpy())
            edge_recon_labels_val_accumulated = np.append(
                edge_recon_labels_val_accumulated,
                edge_recon_labels_val.detach().cpu().numpy())
        val_eval_dict = eval_metrics(
            edge_recon_probs=edge_recon_probs_val_accumulated,
            edge_labels=edge_recon_labels_val_accumulated)
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
        End evaluation logic of Autotalker model used after model training.
        """
        self.model.eval()

        # Get edge-level ground truth and predictions
        edge_recon_probs_val_accumulated = np.array([])
        edge_recon_labels_val_accumulated = np.array([])
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
            if edge_val_model_output["edge_same_condition_labels"] is not None:
                # only keep edge labels from same condition for eval
                edge_recon_probs_val = edge_recon_probs_val[
                    edge_val_model_output["edge_same_condition_labels"]]
                edge_recon_labels_val = edge_recon_labels_val[
                    edge_val_model_output["edge_same_condition_labels"]]
            edge_recon_probs_val_accumulated = np.append(
                edge_recon_probs_val_accumulated,
                edge_recon_probs_val.detach().cpu().numpy())
            edge_recon_labels_val_accumulated = np.append(
                edge_recon_labels_val_accumulated,
                edge_recon_labels_val.detach().cpu().numpy())

        # Get node-level ground truth and predictions
        gene_expr_preds_val_accumulated = np.array([])
        gene_expr_val_accumulated = np.array([])
        for node_val_data_batch in self.node_val_loader:
            node_val_data_batch = node_val_data_batch.to(self.device)

            node_val_model_output = self.model(
                data_batch=node_val_data_batch,
                decoder="omics",
                use_only_active_gps=True)

            gene_expr_val = node_val_model_output["node_labels"]

            if self.model.gene_expr_recon_dist_ == "nb":
                nb_means_val = node_val_model_output["gene_expr_dist_params"]
                gene_expr_preds_val = nb_means_val
            elif self.model.gene_expr_recon_dist_ == "zinb":
                nb_means_val, zi_prob_logits_val = (
                    node_val_model_output["gene_expr_dist_params"])
                zi_probs_val = torch.sigmoid(zi_prob_logits_val)
                zi_mask_val = torch.bernoulli(zi_probs_val) > 0
                gene_expr_preds_val = nb_means_val
                gene_expr_preds_val[zi_mask_val] = 0

            gene_expr_preds_val_accumulated = np.append(
                gene_expr_preds_val_accumulated,
                gene_expr_preds_val.detach().cpu().numpy())
            gene_expr_val_accumulated = np.append(
                gene_expr_val_accumulated,
                gene_expr_val.detach().cpu().numpy())

        val_eval_dict = eval_metrics(
            edge_recon_probs=edge_recon_probs_val_accumulated,
            edge_labels=edge_recon_labels_val_accumulated,
            gene_expr_preds=gene_expr_preds_val_accumulated,
            gene_expr=gene_expr_val_accumulated)
        print("\n--- MODEL EVALUATION ---")
        print(f"Val AUROC score: {val_eval_dict['auroc_score']:.4f}")
        print(f"Val AUPRC score: {val_eval_dict['auprc_score']:.4f}")
        print(f"Val best accuracy score: {val_eval_dict['best_acc_score']:.4f}")
        print(f"Val best F1 score: {val_eval_dict['best_f1_score']:.4f}")
        print(f"Val MSE score: {val_eval_dict['mse_score']:.4f}")
        
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
            If `True`, stop Autotalker model training.
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