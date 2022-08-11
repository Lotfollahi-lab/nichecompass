import time
from collections import defaultdict

import mlflow
import numpy as np
import torch
import torch_geometric

from ._losses import compute_vgae_loss
from ._losses import compute_vgae_loss_parameters
from ._losses import plot_loss_curves
from ._metrics import get_eval_metrics
from ._metrics import plot_eval_metrics
from ._utils import EarlyStopping
from ._utils import prepare_data
from ._utils import print_progress


class Trainer:
    """
    Autotalker trainer module.
    
    Parameters
    ----------
    Returns
    ----------
    """
    def __init__(self,
                 adata,
                 model,
                 valid_frac: float = 0.1,
                 test_frac: float = 0.05,
                 batch_size: int = 3000,
                 use_early_stopping: bool = True,
                 reload_best: bool = True,
                 early_stopping_kwargs: dict = None,
                 **kwargs):
        self.adata = adata
        self.model = model
        self.train_frac = 1 - valid_frac - test_frac
        self.valid_frac = valid_frac
        self.test_frac = test_frac
        self.batch_size = batch_size
        self.use_early_stopping = use_early_stopping
        self.reload_best = reload_best
        self.seed = kwargs.pop("seed", 0)
        self.n_workers = kwargs.pop("n_workers", 0)
        self.monitor = kwargs.pop("monitor", True)
        self.epoch = -1
        self.training_time = 0
        self.optimizer = None
        self.best_epoch = None
        self.best_state_dict = None

        early_stopping_kwargs = (early_stopping_kwargs if early_stopping_kwargs else {})
        self.early_stopping = EarlyStopping(**early_stopping_kwargs)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            self.model.cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.train_data, self.valid_data, self.test_data = prepare_data(
            self.adata,
            valid_frac = self.valid_frac,
            test_frac = self.test_frac)

        #cluster_train_data = torch_geometric.loader.ClusterData(self.train_data, num_parts=100, recursive=False)
        #cluster_valid_data = torch_geometric.loader.ClusterData(self.valid_data, num_parts=100, recursive=False)
#
        #self.train_dataloader = torch_geometric.loader.ClusterLoader(
        #    cluster_data=cluster_train_data,
        #    batch_size=self.batch_size,
        #    shuffle=True,
        #    num_workers=self.n_workers)
#
        #self.valid_dataloader = torch_geometric.loader.ClusterLoader(
        #    cluster_data=cluster_valid_data,
        #    shuffle=True,
        #    batch_size=self.batch_size,
        #    num_workers=self.n_workers)
        
        ### UPDATE TO LinkNeighborLoader once released!

        self.train_dataloader = torch_geometric.loader.NeighborLoader(
            self.train_data,
            num_neighbors=[5]*2,
            batch_size=self.batch_size)

        self.valid_dataloader = torch_geometric.loader.NeighborLoader(
            self.valid_data,
            num_neighbors=[5]*2,
            batch_size=self.batch_size)     


    def loss(self, adj_recon_logits, train_data, mu, logstd):
        vgae_loss_norm_factor, vgae_loss_pos_weight = compute_vgae_loss_parameters(train_data.edge_index)

        vgae_loss_pos_weight = vgae_loss_pos_weight.to(self.device)

        loss = compute_vgae_loss(
            adj_recon_logits = adj_recon_logits,
            edge_label_index = train_data.edge_index,
            pos_weight = vgae_loss_pos_weight,
            mu = mu,
            logstd = logstd,
            n_nodes = train_data.x.size(0),
            norm_factor = vgae_loss_norm_factor)

        return loss


    def train(self,
              n_epochs: int = 200,
              lr: float = 0.01,
              weight_decay: float = 0):
        self.n_epochs = n_epochs
        self.lr = lr
        self.weight_decay = weight_decay

        start_time = time.time()
        self.model.train()

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params, lr = lr, weight_decay = weight_decay)

        self.on_training_start()

        self.epoch_logs = defaultdict(list)
        for self.epoch in range(n_epochs):
            self.iter_logs = defaultdict(list)
            self.iter_logs["n_train_iter"] = 0
            self.iter_logs["n_valid_iter"] = 0

            self.on_epoch_start()
            
            for train_data_batch in self.train_dataloader:
                train_data_batch = train_data_batch.to(self.device)

                self.on_iteration(train_data_batch)

            self.on_epoch_end()

            if self.use_early_stopping:
                if self.is_early_stopping():
                    break

        if self.best_state_dict is not None and self.reload_best:
            print("Saving best state of the network...")
            print("Best state was in epoch", self.best_epoch)
            self.model.load_state_dict(self.best_state_dict)

        self.model.eval()
        self.on_training_end()
        self.training_time += (time.time() - start_time)


    def on_training_start(self):
        mlflow.set_experiment("autotalker")
        mlflow.log_param("n_epochs", self.n_epochs)
        mlflow.log_param("lr", self.lr)
        mlflow.log_param("weight_decay", self.weight_decay)
        mlflow.log_param("n_hidden", self.model.n_hidden)
        mlflow.log_param("n_latent", self.model.n_latent)
        mlflow.log_param("dropout_rate", self.model.dropout_rate)


    def on_epoch_start(self):
        pass


    def on_iteration(self, train_data_batch):
        adj_recon_logits, mu, logstd = self.model(train_data_batch.x, train_data_batch.edge_index)
        train_loss = self.loss(adj_recon_logits, train_data_batch, mu, logstd)
        self.iter_logs["train_losses"].append(train_loss.item())
        self.iter_logs["n_train_iter"] += 1
        self.optimizer.zero_grad()
        train_loss.backward()
        self.optimizer.step()

    @torch.no_grad()
    def on_epoch_end(self):
        # Validate model
        if self.valid_data is not None:
            self.validate()

        # Convert iteration level logs into epoch level logs
        for key in self.iter_logs:
            if key.startswith("train"):
                self.epoch_logs[key].append(np.array(self.iter_logs[key]).sum() / self.iter_logs["n_train_iter"])
            if key.startswith("valid"):
                self.epoch_logs[key].append(np.array(self.iter_logs[key]).sum() / self.iter_logs["n_valid_iter"])

        if self.valid_data is not None:
            self.valid_data = self.valid_data.to(self.device)
            adj_recon_logits, _, _ = self.model(self.valid_data.x, self.valid_data.edge_index)
            adj_recon_probs = torch.sigmoid(adj_recon_logits)
    
            valid_auroc_score, valid_auprc_score, valid_best_acc_score, valid_best_f1_score = get_eval_metrics(
                adj_recon_probs,
                self.valid_data.pos_edge_label_index,
                self.valid_data.neg_edge_label_index)
            
            self.epoch_logs["valid_auroc_scores"].append(valid_auroc_score)
            self.epoch_logs["valid_auprc_scores"].append(valid_auprc_score)
            self.epoch_logs["valid_best_acc_scores"].append(valid_best_acc_score)
            self.epoch_logs["valid_best_f1_scores"].append(valid_best_f1_score)

        # Monitor epoch level logs
        if self.monitor:
            print_progress(self.epoch, self.epoch_logs, self.n_epochs)

    @torch.no_grad()
    def on_training_end(self):
        print("Model training finished...")
        self.model = self.model.to("cpu")

        losses = {"train_loss": self.epoch_logs["train_losses"],
                  "valid_loss": self.epoch_logs["valid_losses"]}

        eval_metrics_valid = {"auroc": self.epoch_logs["valid_auroc_scores"],
                              "auprc": self.epoch_logs["valid_auprc_scores"],
                              "best_acc": self.epoch_logs["valid_best_acc_scores"],
                              "best_f1": self.epoch_logs["valid_best_f1_scores"]}
    
        fig = plot_loss_curves(losses)
        mlflow.log_figure(fig, "loss_curves.png")
        fig = plot_eval_metrics(eval_metrics_valid)  
        mlflow.log_figure(fig, "valid_eval_metrics.png") 

        self.adj_recon_logits, self.mu, self.logstd = self.model(self.test_data.x, self.test_data.edge_index)
        self.adj_recon_probs = torch.sigmoid(self.adj_recon_logits)
    
        test_auroc_score, test_auprc_score, test_best_acc_score, test_best_f1_score = get_eval_metrics(
            self.adj_recon_probs,
            self.test_data.pos_edge_label_index,
            self.test_data.neg_edge_label_index)

        mlflow.log_metric("test_auroc_score", test_auroc_score)
        mlflow.log_metric("test_auprc_score", test_auprc_score)
        mlflow.log_metric("test_best_acc_score", test_best_acc_score)
        mlflow.log_metric("test_best_f1_score", test_best_f1_score)


    def validate(self):
        self.model.eval()

        for valid_data_batch in self.valid_dataloader:
            valid_data_batch = valid_data_batch.to(self.device)
            adj_recon_logits, mu, logstd = self.model(valid_data_batch.x, valid_data_batch.edge_index)
            valid_loss = self.loss(adj_recon_logits, valid_data_batch, mu, logstd)
            self.iter_logs["valid_losses"].append(valid_loss.item())
            self.iter_logs["n_valid_iter"] += 1
        
        self.model.train()


    def is_early_stopping(self):
        # Check wheter to stop early and save best state
        early_stopping_metric = self.early_stopping.early_stopping_metric
        if self.early_stopping.update_state(self.epoch_logs[early_stopping_metric][-1]):
            self.best_state_dict = self.model.state_dict()
            self.best_epoch = self.epoch

        continue_training, reduce_lr = self.early_stopping.step(self.epoch_logs[early_stopping_metric][-1])
        if reduce_lr:
            print(f"\nReducing learning rate...")
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.early_stopping.lr_factor

        return not continue_training