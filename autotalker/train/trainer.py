import copy
import time
import warnings
from collections import defaultdict
from typing import Literal, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from anndata import AnnData

from .metrics import eval_metrics, plot_eval_metrics
from .utils import plot_loss_curves, print_progress, EarlyStopping
from autotalker.data import initialize_dataloaders, prepare_data
from autotalker.modules.utils import _edge_values_and_sorted_labels


class Trainer:
    """
    Trainer class. Adapted from 
    https://github.com/theislab/scarches/blob/master/scarches/trainers/trvae/trainer.py#L13.

    Encapsulates all logic for Autotalker model training preparation and model 
    training.
    
    Parameters
    ----------
    adata:
        AnnData object with raw counts stored in 
        ´adata.layers[counts_layer_key]´, and sparse adjacency matrix stored in 
        ´adata.obsp[adj_key]´.
    model:
        An Autotalker module model instance.
    counts_layer_key:
        Key under which the raw counts are stored in ´adata.layer´.
    adj_key:
        Key under which the sparse adjacency matrix is stored in adata.obsp.
    node_label_method:
        Node label method that will be used for gene expression reconstruction. 
        If ´self´, use only the input features of the node itself as node labels
        for gene expression reconstruction. If ´one-hop´, use a concatenation of
        the node's input features with an average of the input features of all 
        nodes in the node's one-hop neighborhood.
    edge_val_ratio:
        Fraction of the data that is used as validation set on edge-level.
    edge_test_ratio:
        Fraction of the data that is used as test set on edge-level.
    node_val_ratio:
        Fraction of the data that is used as validation set on node-level.
    node_test_ratio:
        Fraction of the data that is used as test set on node-level.
    edge_batch_size:
        Batch size for the edge-level dataloaders (the batch size for the node-
        level dataloaders will be calculated automatically to match the number
        of iterations between edge-level and node-level dataloaders).
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
    """
    def __init__(self,
                 adata: AnnData,
                 model: nn.Module,
                 counts_layer_key: str="counts",
                 adj_key: str="spatial_connectivities",
                 node_label_method: Literal["self",
                                            "one-hop-sum",
                                            "one-hop-norm"]="one-hop-norm",
                 edge_val_ratio: float=0.1,
                 edge_test_ratio: float=0.05,
                 node_val_ratio: float=0.1,
                 node_test_ratio: float=0.0,
                 edge_batch_size: int=64,
                 use_early_stopping: bool=True,
                 reload_best_model: bool=True,
                 early_stopping_kwargs: Optional[dict]=None,
                 seed: int=0,
                 monitor: bool=True,
                 **kwargs):
        self.adata = adata
        self.model = model
        self.counts_layer_key = counts_layer_key
        self.adj_key = adj_key
        self.node_label_method = node_label_method
        self.edge_train_ratio = 1 - edge_val_ratio - edge_test_ratio
        self.edge_val_ratio = edge_val_ratio
        self.edge_test_ratio = edge_test_ratio
        self.node_train_ratio = 1 - node_val_ratio - node_test_ratio
        self.node_val_ratio = node_val_ratio
        self.node_test_ratio = node_test_ratio
        self.edge_batch_size = edge_batch_size
        self.use_early_stopping = use_early_stopping
        self.reload_best_model = reload_best_model
        self.early_stopping_kwargs = (early_stopping_kwargs if 
            early_stopping_kwargs else {})
        if not "early_stopping_metric" in self.early_stopping_kwargs:
            if edge_val_ratio > 0 and node_val_ratio > 0:
                self.early_stopping_kwargs["early_stopping_metric"] = "val_loss"
            else:
                self.early_stopping_kwargs["early_stopping_metric"] = "train_loss"
        self.early_stopping = EarlyStopping(**self.early_stopping_kwargs)
        self.seed = seed
        self.monitor = monitor
        self.loaders_n_direct_neighbors = kwargs.pop(
            "loaders_n_direct_neighbors", -1)
        self.loaders_n_hops = kwargs.pop("loaders_n_hops", 3)
        self.grad_clip_value = kwargs.pop("grad_clip_value", 0.0)
        self.epoch = -1
        self.training_time = 0
        self.optimizer = None
        self.best_epoch = None
        self.best_model_state_dict = None

        print("--- INITIALIZING TRAINER ---")
        
        # Use GPU if available
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            self.model.cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                   "cpu")

        # Prepare data and get node-level and edge-level training, validation
        # and test splits
        data_dict = prepare_data(adata=adata,
                                 counts_layer_key=self.counts_layer_key,
                                 adj_key=self.adj_key,
                                 node_label_method=node_label_method,
                                 edge_val_ratio=self.edge_val_ratio,
                                 edge_test_ratio=self.edge_test_ratio,
                                 node_val_ratio=self.node_val_ratio,
                                 node_test_ratio=self.node_test_ratio)
        self.node_masked_data = data_dict["node_masked_data"]
        self.edge_train_data = data_dict["edge_train_data"]
        self.edge_val_data = data_dict["edge_val_data"]
        self.edge_test_data = data_dict["edge_test_data"]

        self.n_nodes_train = self.node_masked_data.train_mask.sum().item()
        self.n_nodes_val = self.node_masked_data.val_mask.sum().item()
        self.n_nodes_test = self.node_masked_data.test_mask.sum().item()
        self.n_edges_train = self.edge_train_data.edge_label_index.size(1)
        self.n_edges_val = self.edge_val_data.edge_label_index.size(1)
        self.n_edges_test = self.edge_test_data.edge_label_index.size(1)
        print(f"Number of training nodes: {self.n_nodes_train}")
        print(f"Number of validation nodes: {self.n_nodes_val}")
        print(f"Number of training edges: {self.n_edges_train}")
        print(f"Number of validation edges: {self.n_edges_val}")
        print(f"Number of test edges: {self.n_edges_test}")

        # Calculate node-level batch sizes so that number of iterations of 
        # node-level dataloaders match number of iterations of edge-level 
        # dataloaders (or are as close as possible but with node-level 
        # dataloaders having slightly more iterations, in which case some of 
        # their iterations will be cut off by the zipping in the joint iteration
        # loop; this is fine, however, since we use random shuffling so we will
        # use different nodes in different epochs)
        edge_train_loader_n_iter = int(np.ceil(
            self.n_edges_train / edge_batch_size))
        edge_val_loader_n_iter = int(np.ceil(
            self.n_edges_val / edge_batch_size))
        edge_test_loader_n_iter = int(np.ceil(
            self.n_edges_test / edge_batch_size))
        self.node_batch_size_train = int(np.floor(
            self.n_nodes_train / edge_train_loader_n_iter))
        if edge_val_loader_n_iter != 0:
            self.node_batch_size_val = int(np.floor(
                self.n_nodes_val / edge_val_loader_n_iter))
        else:
            # Avoid division by 0 error
            self.node_batch_size_val = 0
        if edge_test_loader_n_iter != 0:
            self.node_batch_size_test = int(np.floor(
                self.n_nodes_test / edge_test_loader_n_iter))
        else:
            # Avoid division by 0 error
            self.node_batch_size_test = 0
        
        # Initialize node-level and edge-level dataloaders
        loader_dict = initialize_dataloaders(
            node_masked_data=self.node_masked_data,
            edge_train_data=self.edge_train_data,
            edge_val_data=self.edge_val_data,
            edge_test_data=self.edge_test_data,
            node_batch_size_train=self.node_batch_size_train,
            node_batch_size_val=self.node_batch_size_val,
            node_batch_size_test=self.node_batch_size_test,
            edge_batch_size=self.edge_batch_size,
            n_direct_neighbors=self.loaders_n_direct_neighbors,
            n_hops=self.loaders_n_hops,
            edges_directed=False,
            neg_edge_sampling_ratio=1.0)
        self.edge_train_loader = loader_dict["edge_train_loader"]
        self.edge_val_loader = loader_dict.pop("edge_val_loader", None)
        self.edge_test_loader = loader_dict.pop("edge_test_loader", None)
        self.node_train_loader = loader_dict["node_train_loader"]
        self.node_val_loader = loader_dict.pop("node_val_loader", None)
        self.node_test_loader = loader_dict.pop("node_test_loader", None)

    def train(self,
              n_epochs: int=30,
              lr: float=0.01,
              weight_decay: float=0,
              mlflow_experiment_id: Optional[str]=None):
        """
        Train the model.

        Parameters
        ----------
        n_epochs:
            Number of epochs.
        lr:
            Learning rate.
        weight_decay:
            Weight decay (L2 penalty).
        mlflow_experiment_id:
            ID of the Mlflow experiment used for tracking training parameters
            and metrics.
        """
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.mlflow_experiment_id = mlflow_experiment_id

        print("\n--- MODEL TRAINING ---")
        
        # Log hyperparameters
        if self.mlflow_experiment_id is not None:
            mlflow.log_param("node_label_method", self.node_label_method)
            mlflow.log_param("edge_train_ratio", self.edge_train_ratio)
            mlflow.log_param("edge_val_ratio", self.edge_val_ratio)
            mlflow.log_param("edge_test_ratio", self.edge_test_ratio)
            mlflow.log_param("node_train_ratio", self.node_train_ratio)
            mlflow.log_param("node_val_ratio", self.node_val_ratio)
            mlflow.log_param("edge_batch_size", self.edge_batch_size)
            mlflow.log_param("use_early_stopping", self.use_early_stopping)
            mlflow.log_param("reload_best_model", self.reload_best_model)
            mlflow.log_param("early_stopping_kwargs", self.early_stopping_kwargs)
            mlflow.log_param("seed", self.seed)
            mlflow.log_param("loaders_n_hops", self.loaders_n_hops)
            mlflow.log_param("loaders_n_direct_neighbors", self.loaders_n_direct_neighbors)
            mlflow.log_param("grad_clip_value", self.grad_clip_value)
            mlflow.log_param("n_epochs", self.n_epochs)
            mlflow.log_param("lr", self.lr)
            mlflow.log_param("weight_decay", self.weight_decay)
            self.model.log_module_hyperparams_to_mlflow()

        start_time = time.time()
        self.epoch_logs = defaultdict(list)
        self.model.train()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params,
                                          lr=lr,
                                          weight_decay=weight_decay)

        for self.epoch in range(n_epochs):
            self.iter_logs = defaultdict(list)
            self.iter_logs["n_train_iter"] = 0
            self.iter_logs["n_val_iter"] = 0
            
            # Jointly loop through edge- and node-level batches
            for edge_train_data_batch, node_train_data_batch in zip(
                    self.edge_train_loader, self.node_train_loader):
                edge_train_data_batch = edge_train_data_batch.to(self.device)
                node_train_data_batch = node_train_data_batch.to(self.device)

                # Forward pass edge-level batch
                edge_train_model_output = self.model(
                    x=edge_train_data_batch.x,
                    edge_index=edge_train_data_batch.edge_index,
                    decoder="graph")

                # Forward pass node-level batch
                node_train_model_output = self.model(
                    x=node_train_data_batch.x,
                    edge_index=node_train_data_batch.edge_index,
                    decoder="gene_expr")
                    
                # Calculate training loss (edge reconstruction loss + gene 
                # expression reconstruction loss)
                train_loss_dict = self.model.loss(
                    edge_data_batch=edge_train_data_batch,
                    edge_model_output=edge_train_model_output,
                    node_data_batch=node_train_data_batch,
                    node_model_output=node_train_model_output,
                    device=self.device)
                train_loss = train_loss_dict["loss"]
                train_edge_recon_loss = train_loss_dict["edge_recon_loss"]
                train_kl_loss = train_loss_dict["kl_loss"]
                train_gene_expr_recon_loss = train_loss_dict[
                    "gene_expr_recon_loss"]
                self.iter_logs["train_loss"].append(train_loss.item())
                self.iter_logs["train_edge_recon_loss"].append(
                    train_edge_recon_loss.item())
                self.iter_logs["train_kl_loss"].append(
                    train_kl_loss.item())    
                self.iter_logs["train_gene_expr_recon_loss"].append(
                    train_gene_expr_recon_loss.item())
                self.iter_logs["n_train_iter"] += 1

                # Optimize for training loss
                self.optimizer.zero_grad()
                train_loss.backward()

                # Clip gradients
                if self.grad_clip_value > 0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(),
                                                    self.grad_clip_value)

                self.optimizer.step()

            # Validate model
            if (self.edge_val_loader is not None and 
                    self.node_val_loader is not None):
                self.validate()
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
            if self.monitor:
                print_progress(self.epoch, self.epoch_logs, self.n_epochs)

            # Check early stopping
            if self.use_early_stopping:
                if self.is_early_stopping():
                    break

        self.training_time += (time.time() - start_time)
        minutes, seconds = divmod(self.training_time, 60)
        print(f"Model training finished after {int(minutes)} min {int(seconds)}"
               " sec.")

        if self.best_model_state_dict is not None and self.reload_best_model:
            print("Using best model state, which was in epoch "
                  f"{self.best_epoch + 1}.")
            self.model.load_state_dict(self.best_model_state_dict)

        self.model.eval()

        losses = {"train_loss": self.epoch_logs["train_loss"],
                  "val_loss": self.epoch_logs["val_loss"]}

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

        # Test model
        if self.edge_test_loader is not None:
            self.test()

    @torch.no_grad()
    def validate(self):
        self.model.eval()

        edge_recon_probs_val_accumulated = np.array([])
        edge_labels_val_accumulated = np.array([])
        
        # Jointly loop through edge and node level batches
        for edge_val_data_batch, node_val_data_batch in zip(
                self.edge_val_loader, self.node_val_loader):
            edge_val_data_batch = edge_val_data_batch.to(self.device)
            node_val_data_batch = node_val_data_batch.to(self.device)

            # Forward pass edge level batch
            edge_val_model_output = self.model(
                edge_val_data_batch.x,
                edge_val_data_batch.edge_index,
                decoder="graph")

            # Forward pass node level batch
            node_val_model_output = self.model(
                node_val_data_batch.x,
                node_val_data_batch.edge_index,
                decoder="gene_expr")
            
            # Calculate validation loss (edge reconstruction loss + gene 
            # expression reconstruction loss)
            val_loss_dict = self.model.loss(
                    edge_data_batch=edge_val_data_batch,
                    edge_model_output=edge_val_model_output,
                    node_data_batch=node_val_data_batch,
                    node_model_output=node_val_model_output,
                    device=self.device)
            val_loss = val_loss_dict["loss"]
            val_edge_recon_loss = val_loss_dict["edge_recon_loss"]
            val_kl_loss = val_loss_dict["kl_loss"]
            val_gene_expr_recon_loss = val_loss_dict["gene_expr_recon_loss"]
            self.iter_logs["val_loss"].append(val_loss.item())
            self.iter_logs["val_edge_recon_loss"].append(
                val_edge_recon_loss.item())
            self.iter_logs["val_kl_loss"].append(
                    val_kl_loss.item())    
            self.iter_logs["val_gene_expr_recon_loss"].append(
                val_gene_expr_recon_loss.item())
            self.iter_logs["n_val_iter"] += 1
            
            # Calculate evaluation metrics
            adj_recon_probs_val = torch.sigmoid(
                edge_val_model_output["adj_recon_logits"])

            edge_recon_probs_val, edge_labels_val = _edge_values_and_sorted_labels(
                adj=adj_recon_probs_val,
                edge_label_index=edge_val_data_batch.edge_label_index,
                edge_labels=edge_val_data_batch.edge_label)

            edge_recon_probs_val_accumulated = np.append(
                edge_recon_probs_val_accumulated,
                edge_recon_probs_val.detach().cpu().numpy())
            edge_labels_val_accumulated = np.append(
                edge_labels_val_accumulated,
                edge_labels_val.detach().cpu().numpy())

        val_eval_dict = eval_metrics(
            edge_recon_probs=edge_recon_probs_val_accumulated,
            edge_labels=edge_labels_val_accumulated)

        self.epoch_logs["val_auroc_score"].append(val_eval_dict["auroc_score"])
        self.epoch_logs["val_auprc_score"].append(val_eval_dict["auprc_score"])
        self.epoch_logs["val_best_acc_score"].append(val_eval_dict["best_acc_score"])
        self.epoch_logs["val_best_f1_score"].append(val_eval_dict["best_f1_score"])
        
        self.model.train()

    @torch.no_grad()
    def test(self):
        self.model.eval()

        edge_recon_probs_test_accumulated = np.array([])
        edge_labels_test_accumulated = np.array([])

        for edge_test_data_batch in self.edge_test_loader:
            edge_test_data_batch = edge_test_data_batch.to(self.device)

            edge_test_model_output = self.model(edge_test_data_batch.x,
                                                edge_test_data_batch.edge_index,
                                                decoder="graph")
    
            # Calculate evaluation metrics
            adj_recon_probs_test = torch.sigmoid(
                edge_test_model_output["adj_recon_logits"])

            edge_recon_probs_test, edge_labels_test = _edge_values_and_sorted_labels(
                adj=adj_recon_probs_test,
                edge_label_index=edge_test_data_batch.edge_label_index,
                edge_labels=edge_test_data_batch.edge_label)

            edge_recon_probs_test_accumulated = np.append(
                edge_recon_probs_test_accumulated,
                edge_recon_probs_test.detach().cpu().numpy())
            edge_labels_test_accumulated = np.append(
                edge_labels_test_accumulated,
                edge_labels_test.detach().cpu().numpy())

        test_eval_dict = eval_metrics(
            edge_recon_probs=edge_recon_probs_test_accumulated,
            edge_labels=edge_labels_test_accumulated)

        print("\n--- MODEL EVALUATION ---")
        print(f"Test AUROC score: {test_eval_dict['auroc_score']:.4f}")
        print(f"Test AUPRC score: {test_eval_dict['auprc_score']:.4f}")
        print(f"Test best accuracy score: {test_eval_dict['best_acc_score']:.4f}")
        print(f"Test best F1 score: {test_eval_dict['best_f1_score']:.4f}")
        
        # Log evaluation metrics
        if self.mlflow_experiment_id is not None:
            mlflow.log_metric("test_auroc_score", 
                              test_eval_dict['auroc_score'])
            mlflow.log_metric("test_auprc_score",
                              test_eval_dict['auprc_score'])
            mlflow.log_metric("test_best_acc_score",
                              test_eval_dict['best_acc_score'])
            mlflow.log_metric("test_best_f1_score",
                              test_eval_dict['best_f1_score'])
            mlflow.end_run()

    def is_early_stopping(self):
        # Check whether to apply early stopping and save best model state
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

        return not continue_training