import time
from collections import defaultdict

import mlflow
import numpy as np
import torch

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
                 val_frac: float = 0.1,
                 test_frac: float = 0.05,
                 use_early_stopping: bool = True,
                 early_stopping_kwargs: dict = None,
                 **kwargs):
        self.adata = adata
        self.model = model
        self.train_frac = 1 - val_frac - test_frac
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.use_early_stopping = use_early_stopping
        self.seed = kwargs.pop("seed", 0)
        self.monitor = kwargs.pop("monitor", True)
        self.epoch = -1
        self.training_time = 0
        self.optimizer = None
        self.best_epoch = None
        self.best_state_dict = None
        self.logs = defaultdict(list)

        early_stopping_kwargs = (early_stopping_kwargs if early_stopping_kwargs else {})
        self.early_stopping = EarlyStopping(**early_stopping_kwargs)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            self.model.cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.train_data, self.valid_data, self.test_data = prepare_data(
            self.adata,
            val_frac = self.val_frac,
            test_frac = self.test_frac)

        self.train_data = self.train_data.to(self.device)
        self.valid_data = self.valid_data.to(self.device)
        self.test_data = self.test_data.to(self.device)


    def loss(adj_recon_logits, train_data, mu, logstd):
        vgae_loss_norm_factor, vgae_loss_pos_weight = compute_vgae_loss_parameters(train_data.edge_index)
        
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

        for epoch in range(n_epochs):
            self.on_epoch_start()

            self.adj_recon_logits, self.mu, self.logstd = self.model(
                self.train_data.x, self.train_data.edge_index)

            train_loss = self.loss(self.adj_recon_logits, self.train_data, self.mu, self.logstd)
            self.train_losses.append(train_loss)

            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()

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


    def on_epoch_end(self):
        # Validate model
        if self.valid_data is not None:
            self.validate()

        # Monitor logs
        if self.monitor:
            print_progress(self.epoch, self.logs, self.n_epochs, False)


    def on_training_end(self):
        print("Model training finished...")

        losses = {"train_loss": self.log["train_losses"],
                  "val_loss": self.log["val_losses"]}

        eval_metrics_val = {"auroc": self.logs["val_auroc_scores"],
                            "auprc": self.logs["val_auprc_scores"],
                            "best_acc": self.logs["val_best_acc_scores"],
                            "best_f1": self.logs["val_best_f1_scores"]}
    
        fig = plot_loss_curves(losses)
        mlflow.log_figure(fig, "loss_curve.png")
        fig = plot_eval_metrics(eval_metrics_val)  
        mlflow.log_figure(fig, "val_eval_metrics.png") 
    
        test_auroc_score, test_auprc_score, test_best_acc_score, test_best_f1_score = get_eval_metrics(
            self.adj_recon_probs,
            self.test_data.pos_edge_label_index,
            self.test_data.neg_edge_label_index)

        mlflow.log_metric("test_auroc_score", test_auroc_score)
        mlflow.log_metric("test_auprc_score", test_auprc_score)
        mlflow.log_metric("test_best_acc_score", btest_est_acc_score)
        mlflow.log_metric("test_best_f1_score", test_best_f1_score)


    @torch.no_grad()
    def validate(self):
        self.model.eval()

        val_loss = self.loss(self.adj_recon_logits, self.val_data, self.mu, self.logstd)
        auroc_score_val, auprc_score_val, best_acc_score_val, best_f1_score_val = get_eval_metrics(
            self.adj_recon_probs,
            self.val_data.pos_edge_label_index,
            self.val_data.neg_edge_label_index)
        
        self.logs["val_losses"].append(val_loss)
        self.logs["auroc_scores_val"].append(auroc_score_val)
        self.logs["auprc_scores_val"].append(auprc_score_val)
        self.logs["best_acc_scores_val"].append(best_acc_score_val)
        self.logs["best_f1_scores_val"].append(best_f1_score_val)
        
        self.model.train()


    def is_early_stopping(self):
        # Check wheter to stop early and save best state
        early_stopping_metric = self.early_stopping.early_stopping_metric
        if self.early_stopping.update_state(self.logs[early_stopping_metric][-1]):
            self.best_state_dict = self.model.state_dict()
            self.best_epoch = self.epoch

        continue_training, reduce_lr = self.early_stopping.step(self.logs[early_stopping_metric][-1])
        if reduce_lr:
            print(f"\nReducing learning rate...")
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.early_stopping.lr_factor

        return not continue_training