import time
from collections import defaultdict

import mlflow
import torch
from torch_geometric.utils import add_self_loops
from toch_geometric.utils import to_dense_adj

from ._utils import EarlyStopping
from ._utils import prepare_data
from ._losses import compute_vgae_loss
from ._losses import compute_vgae_loss_parameters
from ._losses import plot_loss
from ._metrics import get_eval_metrics
from ._metrics import plot_eval_metrics


class Trainer:
    """
    Autotalker trainer module.
    
    Parameters
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
        early_stopping_kwargs = (early_stopping_kwargs if early_stopping_kwargs else {})
        self.early_stopping = EarlyStopping(**early_stopping_kwargs)

        self.n_samples = kwargs.pop("n_samples", None)
        self.train_frac = kwargs.pop("train_frac", 0.9)
        self.use_stratified_sampling = kwargs.pop("use_stratified_sampling", True)
        self.weight_decay = kwargs.pop("weight_decay", 0.04)
        self.clip_value = kwargs.pop("clip_value", 0.0)
        self.n_workers = kwargs.pop("n_workers", 0)
        self.seed = kwargs.pop("seed", 2020)
        self.monitor = kwargs.pop("monitor", True)
        self.monitor_only_val = kwargs.pop("monitor_only_val", True)

        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            self.model.cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epoch = -1
        self.n_epochs = None
        self.iter = 0
        self.best_epoch = None
        self.best_state_dict = None
        self.current_loss = None
        self.previous_loss_was_nan = False
        self.nan_counter = 0
        self.optimizer = None
        self.training_time = 0

        self.train_data = None
        self.valid_data = None
        self.test_data = None

        self.logs = defaultdict(list)

        self.train_data, self.valid_data, self.test_data = prepare_data(
            self.adata,
            val_frac = self.val_frac,
            test_frac = self.test_frac)


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
              eps: float = 0.01):

        start_time = time.time()
        self.model.train()
        self.n_epochs = n_epochs

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params, lr = lr, eps = eps, weight_decay = self.weight_decay)

        self.on_training_start()

        for self.epoch in range(n_epochs):
            self.on_epoch_start(lr, eps)
            self.iter_logs = defaultdict(list)

            # Validation of Model, Monitoring, Early Stopping

        # Calculate Loss depending on Trainer/Model
        self.current_loss = loss = self.loss()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

            self.on_epoch_end()
            if self.use_early_stopping:
                if not self.check_early_stop():
                    break

        if self.best_state_dict is not None and self.reload_best:
            print("Saving best state of the network...")
            print("Best state was in epoch", self.best_epoch)
            self.model.load_state_dict(self.best_state_dict)

        self.model.eval()
        self.on_training_end()

        self.training_time += (time.time() - start_time)

    def on_training_start(self):
        pass

    def on_epoch_start(self, lr, eps):
        pass

    def on_training_end(self):
        print("Model training finished...")

        eval_metrics_val = {"auroc": self.auroc_scores_val,
                            "auprc": self.auprc_scores_val,
                            "best_acc": self.best_acc_scores_val,
                            "best_f1": self.best_f1_scores_val}
    
        plot_loss(self.losses)
        mlflow.log_artifact("images/training_loss.png")
        plot_eval_metrics(eval_metrics_val)               
        mlflow.log_artifact("images/eval_metrics.png")
    
        auroc_score_test, auprc_score_test, best_acc_score_test, best_f1_score_test = get_eval_metrics(
            self.adj_recon_probs,
            self.test_data.pos_edge_label_index,
            self.test_data.neg_edge_label_index)

        print(f"Test (balanced) AUROC score: {auroc_score_test}")
        print(f"Test (balanced) AUPRC score: {auprc_score_test}")
        print(f"Test (balanced) best ACC score: {best_acc_score_test}")
        print(f"Test (balanced) best F1 score: {best_f1_score_test}")
    
        mlflow.log_metric("auroc_score_test", auroc_score_test)
        mlflow.log_metric("auprc_score_test", auprc_score_test)
        mlflow.log_metric("best_acc_score_test", best_acc_score_test)
        mlflow.log_metric("best_f1_score_test", best_f1_score_test)

    def on_epoch_end(self):
        # Get Train Epoch Logs
        for key in self.iter_logs:
            self.logs["epoch_" + key].append(np.array(self.iter_logs[key]).mean())

        # Validate Model
        if self.valid_data is not None:
            self.validate()

        # Monitor Logs
        if self.monitor:
            print_progress(self.epoch, self.logs, self.n_epochs, self.monitor_only_val)

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.iter_logs = defaultdict(list)
        # Calculate Validation Losses
        for val_iter, batch_data in enumerate(self.dataloader_valid):
            for key, batch in batch_data.items():
                batch_data[key] = batch.to(self.device)

            val_loss = self.loss(batch_data)

        # Get Validation Logs
        for key in self.iter_logs:
            self.logs["val_" + key].append(np.array(self.iter_logs[key]).mean())

        self.model.train()

    def check_early_stop(self):
        # Calculate Early Stopping and best state
        early_stopping_metric = self.early_stopping.early_stopping_metric
        if self.early_stopping.update_state(self.logs[early_stopping_metric][-1]):
            self.best_state_dict = self.model.state_dict()
            self.best_epoch = self.epoch

        continue_training, update_lr = self.early_stopping.update(self.logs[early_stopping_metric][-1])
        if update_lr:
            print(f"\nADJUSTED LR")
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.early_stopping.lr_factor

        return continue_training