import time
from collections import defaultdict


import torch
from torch_geometric.loader import DataLoader

from .utils import EarlyStopping

class Trainer:
    """
    Autotalker base trainer class
    
    Parameters
    ----------

    
    
    """

    def __init__(self,
                 adata,
                 model,
                 batch_size: int = 128,
                 alpha_epoch_anneal: int = None,
                 alpha_kl: float = 1.,
                 use_early_stopping: bool = True,
                 reload_best: bool = True,
                 early_stopping_kwargs: dict = None,
                 **kwargs):

        self.adata = adata
        self.model = model
        self.batch_size = batch_size
        self.alpha_epoch_anneal = alpha_epoch_anneal
        self.alpha_kl = alpha_kl
        self.use_early_stopping = use_early_stopping
        self.reload_best = reload_best

        early_stopping_kwargs = (early_stopping_kwargs if early_stopping_kwargs else {})

        self.alpha_iter_anneal = kwargs.pop("alpha_iter_anneal", None)
        self.n_samples = kwargs.pop("n_samples", None)
        self.train_frac = kwargs.pop("train_frac", 0.9)
        self.use_stratified_sampling = kwargs.pop("use_stratified_sampling", True)

        self.weight_decay = kwargs.pop("weight_decay", 0.04)
        self.clip_value = kwargs.pop("clip_value", 0.0)

        self.n_workers = kwargs.pop("n_workers", 0)
        self.seed = kwargs.pop("seed", 2020)
        self.monitor = kwargs.pop("monitor", True)
        self.monitor_only_val = kwargs.pop("monitor_only_val", True)

        self.early_stopping = EarlyStopping(**early_stopping_kwargs)

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
        self.sampler = None
        self.dataloader_train = None
        self.dataloader_valid = None

        self.iters_per_epoch = None
        self.val_iters_per_epoch = None

        self.logs = defaultdict(list)

        # Create Train/Valid AnnotatetDataset objects
        self.train_data, self.valid_data = make_dataset(
            self.adata,
            train_frac=self.train_frac,
            condition_key=self.condition_key,
            cell_type_keys=self.cell_type_keys,
            condition_encoder=self.model.condition_encoder,
            cell_type_encoder=self.model.cell_type_encoder,
        )

    def train(self,
              n_epochs: int = 400,
              lr: float = 1e-3,
              eps: float = 0.01):

        self.initialize_loaders()
        start_time = time.time()
        self.model.train()
        self.n_epochs = n_epochs

        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params, lr = lr, eps = eps, weight_decay = self.weight_decay)

        self.before_loop()

        for self.epoch in range(n_epochs):
            self.on_epoch_begin(lr, eps)
            self.iter_logs = defaultdict(list)
            for self.iter, batch_data in enumerate(self.dataloader_train):
                for key, batch in batch_data.items():
                    batch_data[key] = batch.to(self.device)

                # Loss Calculation
                self.on_iteration(batch_data)

            # Validation of Model, Monitoring, Early Stopping
            self.on_epoch_end()
            if self.use_early_stopping:
                if not self.check_early_stop():
                    break

        if self.best_state_dict is not None and self.reload_best:
            print("Saving best state of network...")
            print("Best State was in Epoch", self.best_epoch)
            self.model.load_state_dict(self.best_state_dict)

        self.model.eval()
        self.after_loop()

        self.training_time += (time.time() - start_time)

    def before_loop(self):
        pass

    def on_epoch_begin(self, lr, eps):
        pass

    def after_loop(self):
        pass

    def on_iteration(self, batch_data):
        # Dont update any weight on first layers except condition weights
        if self.model.freeze:
            for name, module in self.model.named_modules():
                if isinstance(module, nn.BatchNorm1d):
                    if not module.weight.requires_grad:
                        module.affine = False
                        module.track_running_stats = False

        # Calculate Loss depending on Trainer/Model
        self.current_loss = loss = self.loss(batch_data)
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        if self.clip_value > 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)

        self.optimizer.step()

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