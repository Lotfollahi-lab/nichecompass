import time
from collections import defaultdict
from typing import Optional, Union

import anndata as ad
import mlflow
import numpy as np
import torch

from autotalker.data import prepare_data
from autotalker.data import train_valid_test_node_level_mask
from autotalker.data import train_valid_test_link_level_split
from autotalker.data import initialize_link_level_dataloader
from autotalker.modules import VGAE
from ._metrics import get_eval_metrics
from ._metrics import plot_eval_metrics
from ._utils import EarlyStopping
from ._utils import plot_loss_curves
from ._utils import print_progress
from ._utils import transform_test_edge_labels


class Trainer:
    """
    Autotalker trainer module.
    
    Parameters
    ----------
    adata:
        AnnData object with sparse adjacency matrix stored in 
        adata.obsp[adj_key].
    model:
        An Autotalker model.
    adj_key:
        Key under which the sparse adjacency matrix is stored in adata.obsp.
    valid_frac:
        Fraction of the data that is used for validation.
    test_frac:
        Fraction of the data that is used for testing.
    batch_size:
        Batch size per iteration.
    use_early_stopping:
        If "True", the EarlyStopping class is used to prevent overfitting.
    reload_best_model:
        If "True", the best state of the model with respect to the early
        stopping criterion is reloaded at the end of training.
    early_stopping_kwargs:
        Custom parameters for the EarlyStopping class.
    seed:
        Random seed to get reproducible results.
    n_workers:
        Parameter n_workers of the torch dataloaders.
    monitor:
        If "True", the progress of the training will be printed after each 
        epoch.
    """
    def __init__(self,
                 adata: ad.AnnData,
                 model: VGAE,
                 condition_key: str,
                 adj_key: str="spatial_connectivities",
                 valid_frac: float=0.1,
                 test_frac: float=0.05,
                 batch_size: int=128,
                 use_early_stopping: bool=True,
                 reload_best_model: bool=True,
                 early_stopping_kwargs: Optional[dict]=None,
                 **kwargs):
        self.adata = adata
        self.model = model
        self.adj_key = adj_key
        self.condition_key = condition_key
        self.train_frac = 1 - valid_frac - test_frac
        self.valid_frac = valid_frac
        self.test_frac = test_frac
        self.batch_size = batch_size
        self.use_early_stopping = use_early_stopping
        self.reload_best_model = reload_best_model
        early_stopping_kwargs = (early_stopping_kwargs if early_stopping_kwargs 
                                 else {})
        self.early_stopping = EarlyStopping(**early_stopping_kwargs)

        self.seed = kwargs.pop("seed", 0)
        self.n_workers = kwargs.pop("n_workers", 0)
        self.monitor = kwargs.pop("monitor", True)

        self.epoch = -1
        self.training_time = 0
        self.optimizer = None
        self.best_epoch = None
        self.best_model_state_dict = None

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            self.model.cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                   "cpu")
        
        # remove later
        condition_label_dict = {}

        data = prepare_data(
            adata=self.adata,
            condition_key=self.condition_key,
            condition_label_dict=condition_label_dict,
            adj_key=self.adj_key,
            valid_frac=self.valid_frac,
            test_frac=self.test_frac)

        data_link_splits = train_valid_test_link_level_split(data)
        self.train_data = data_link_splits[0]
        self.valid_data = data_link_splits[1]
        self.test_data = data_link_splits[2]

        self.train_dataloader = initialize_link_level_dataloader(
            data=self.train_data,
            batch_size=self.batch_size)
        self.valid_dataloader = initialize_link_level_dataloader(
            data=self.valid_data,
            batch_size=self.batch_size)

    def train(self,
              n_epochs: int=200,
              lr: float=0.01,
              weight_decay: float=0,
              mlflow_experiment_id: Optional[str]=None):
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.mlflow_experiment_id = mlflow_experiment_id

        start_time = time.time()
        self.model.train()

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params,
                                          lr=lr,
                                          weight_decay=weight_decay)

        self.on_training_start()

        self.epoch_logs = defaultdict(list)
        for self.epoch in range(n_epochs):
            self.iter_logs = defaultdict(list)
            self.iter_logs["n_train_iter"] = 0
            self.iter_logs["n_valid_iter"] = 0

            self.on_epoch_start()
            
            for train_data_batch in self.train_dataloader:
                train_data_batch = train_valid_test_node_level_mask(train_data_batch)
                train_data_batch = train_data_batch.to(self.device)
                self.on_iteration(train_data_batch)

            self.on_epoch_end()

            if self.use_early_stopping:
                if self.is_early_stopping():
                    break

        if self.best_model_state_dict is not None and self.reload_best_model:
            print("Saving best state of the network...")
            print("Best state was in epoch", self.best_epoch)
            self.model.load_state_dict(self.best_model_state_dict)

        self.model.eval()
        self.on_training_end()
        self.training_time += (time.time() - start_time)


    def on_training_start(self):
        mlflow.log_param("n_epochs", self.n_epochs)
        mlflow.log_param("lr", self.lr)
        mlflow.log_param("weight_decay", self.weight_decay)
        mlflow.log_param("n_hidden", self.model.n_hidden)
        mlflow.log_param("n_latent", self.model.n_latent)
        mlflow.log_param("dropout_rate", self.model.dropout_rate)


    def on_epoch_start(self):
        pass


    def on_iteration(self, train_data_batch):
        adj_recon_logits, x_recon, mu, logstd = self.model(
            train_data_batch.x,
            train_data_batch.edge_index)
        train_loss, _, _ = self.model.loss(adj_recon_logits,
                                           x_recon,
                                           train_data_batch,
                                           mu,
                                           logstd,
                                           self.device)
        self.iter_logs["train_loss"].append(train_loss.item())
        self.iter_logs["n_train_iter"] += 1
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()


    def on_epoch_end(self):
        # Validate model
        if self.valid_data is not None:
            self.validate()

        # Convert iteration level logs into epoch level logs
        for key in self.iter_logs:
            if key.startswith("train"):
                self.epoch_logs[key].append(np.array(self.iter_logs[key]).sum()
                                            / self.iter_logs["n_train_iter"])
            if key.startswith("valid"):
                self.epoch_logs[key].append(np.array(self.iter_logs[key]).sum()
                                            / self.iter_logs["n_valid_iter"])

        # Monitor epoch level logs
        if self.monitor:
            monitor_logs = {key: self.epoch_logs[key] for key in self.epoch_logs 
                            if key != "train_loss"}
            print_progress(self.epoch, monitor_logs, self.n_epochs)


    @torch.no_grad()
    def validate(self):
        self.model.eval()

        for valid_data_batch in self.valid_dataloader:
            valid_data_batch = valid_data_batch.to(self.device)
            
            adj_recon_logits, x_recon, mu, logstd = self.model(
                valid_data_batch.x,
                valid_data_batch.edge_index)
            valid_loss, valid_vgae_loss, valid_expr_recon_loss = self.model.loss(
                adj_recon_logits,
                x_recon,
                valid_data_batch,
                mu,
                logstd,
                self.device)
            self.iter_logs["valid_loss"].append(valid_loss.item())
            self.iter_logs["valid_vgae_loss"].append(valid_vgae_loss.item())
            self.iter_logs["valid_expr_recon_loss"].append(
                valid_expr_recon_loss.item())
            self.iter_logs["n_valid_iter"] += 1

            adj_recon_probs = torch.sigmoid(adj_recon_logits)
    
            valid_eval_metrics = get_eval_metrics(
                adj_recon_probs,
                valid_data_batch.edge_label_index,
                valid_data_batch.edge_label)

            valid_auroc_score = valid_eval_metrics[0]
            valid_auprc_score = valid_eval_metrics[1]
            valid_best_acc_score = valid_eval_metrics[2]
            valid_best_f1_score = valid_eval_metrics[3]
            
            self.iter_logs["valid_auroc_score"].append(valid_auroc_score)
            self.iter_logs["valid_auprc_score"].append(valid_auprc_score)
            self.iter_logs["valid_best_acc_score"].append(valid_best_acc_score)
            self.iter_logs["valid_best_f1_score"].append(valid_best_f1_score)
        
        self.model.train()
    

    @torch.no_grad()
    def on_training_end(self):
        print("Model training finished...")
        self.model = self.model.to("cpu")

        losses = {"train_loss": self.epoch_logs["train_loss"],
                  "valid_loss": self.epoch_logs["valid_loss"]}

        valid_eval_metrics = {"auroc": self.epoch_logs["valid_auroc_score"],
                              "auprc": self.epoch_logs["valid_auprc_score"],
                              "best_acc": self.epoch_logs["valid_best_acc_score"],
                              "best_f1": self.epoch_logs["valid_best_f1_score"]}
    
        fig = plot_loss_curves(losses)
        mlflow.log_figure(fig, "loss_curves.png")
        fig = plot_eval_metrics(valid_eval_metrics)  
        mlflow.log_figure(fig, "valid_eval_metrics.png") 

        self.adj_recon_logits, self.x_recon, self.mu, self.logstd = self.model(
            self.test_data.x,
            self.test_data.edge_index)
        self.adj_recon_probs = torch.sigmoid(self.adj_recon_logits)

        test_edge_label_index, test_edge_labels = transform_test_edge_labels(
            self.test_data.pos_edge_label_index,
            self.test_data.neg_edge_label_index)
    
        test_eval_metrics = get_eval_metrics(
            self.adj_recon_probs,
            test_edge_label_index,
            test_edge_labels)

        test_auroc_score = test_eval_metrics[0]
        test_auprc_score = test_eval_metrics[1]
        test_best_acc_score = test_eval_metrics[2]
        test_best_f1_score = test_eval_metrics[3]

        mlflow.log_metric("test_auroc_score", test_auroc_score)
        mlflow.log_metric("test_auprc_score", test_auprc_score)
        mlflow.log_metric("test_best_acc_score", test_best_acc_score)
        mlflow.log_metric("test_best_f1_score", test_best_f1_score)

        mlflow.end_run()


    def is_early_stopping(self):
        # Check whether to apply early stopping and save best model state
        early_stopping_metric = self.early_stopping.early_stopping_metric
        current_metric = self.epoch_logs[early_stopping_metric][-1]
        if self.early_stopping.update_state(current_metric):
            self.best_model_state_dict = self.model.state_dict()
            self.best_epoch = self.epoch

        continue_training, reduce_lr = self.early_stopping.step(current_metric)
        if reduce_lr:
            print(f"\nReducing learning rate...")
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.early_stopping.lr_factor

        return not continue_training