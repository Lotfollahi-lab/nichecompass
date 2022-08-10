import numpy as np
import sys
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

from autotalker.data import SpatialAnnDataset


class EarlyStopping:
    """
    EarlyStopping class for early stopping of Autotalker training.

    This early stopping class was inspired by:
    Title: scvi-tools
    Authors: Romain Lopez <romain_lopez@gmail.com>,
             Adam Gayoso <adamgayoso@berkeley.edu>,
             Galen Xing <gx2113@columbia.edu>
    Date: 24th December 2020
    Code version: 0.8.1
    Availability: 
    https://github.com/YosefLab/scvi-tools/blob/8f5a9cc362325abbb7be1e07f9523cfcf7e55ec0/scvi/core/trainers/trainer.py
    
    Parameters
    ----------
    early_stopping_metric:
        The metric on which the early stopping criterion is calculated.
    metric_improvement_threshold:
        The minimum value which counts as metric_improvement.
    patience:
        Number of epochs which are allowed to have no metric_improvement until the training is stopped.
    reduce_lr_on_plateau:
        If "True", the learning rate gets adjusted by "lr_factor" after a given number of epochs with no
        metric_improvement.
    lr_patience:
        Number of epochs which are allowed to have no metric_improvement until the learning rate is adjusted.
    lr_factor:
        Scaling factor for adjusting the learning rate.
     """
    def __init__(self,
                 early_stopping_metric: str = None,
                 metric_improvement_threshold: float = 0,
                 patience: int = 15,
                 reduce_lr_on_plateau: bool = True,
                 lr_patience: int = 13,
                 lr_factor: float = 0.1):

        self.early_stopping_metric = early_stopping_metric
        self.metric_improvement_threshold = metric_improvement_threshold
        self.patience = patience
        self.reduce_lr_on_plateau = reduce_lr_on_plateau
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor

        self.epochs = 0
        self.epochs_not_improved = 0
        self.epochs_not_improved_lr = 0

        self.current_performance = np.inf
        self.best_performance = np.inf
        self.best_performance_state = np.inf

    def step(self, current_metric):
        self.epochs += 1
        # Determine whether to continue training
        if self.epochs < self.patience:
            continue_training = True
            reduce_lr = False
        elif self.epochs_not_improved >= self.patience:
            continue_training = False
            reduce_lr = False
        # Determine whether to reduce the learning rate
        else:
            if self.reduce_lr_on_plateau == False:
                reduce_lr = False
            elif self.epochs_not_improved_lr >= self.lr_patience:
                reduce_lr = True
                self.epochs_not_improved_lr = 0
            else:
                reduce_lr = False
            
            # Shift
            self.current_performance = current_metric
                
            metric_improvement = self.best_performance - self.current_performance
            
            # Updating best performance
            if metric_improvement > 0:
                self.best_performance = self.current_performance

            # Updating epochs not improved
            if metric_improvement < self.metric_improvement_threshold:
                self.epochs_not_improved += 1
                self.epochs_not_improved_lr += 1
            else:
                self.epochs_not_improved = 0
                self.epochs_not_improved_lr = 0

            continue_training = True

        if not continue_training:
            print("\nStopping early: metric has not improved more than " 
                  + str(self.metric_improvement_threshold) +
                  " in the last " + str(self.patience) + " epochs")
            print("If the early stopping criterion is too strong, "
                  "please instantiate it with different parameters in the train method.")
        return continue_training, reduce_lr

    def update_state(self, current_metric):
        improved = (self.best_performance_state - current_metric) > 0
        if improved:
            self.best_performance_state = current_metric
        return improved


def prepare_data(
    adata,
    val_frac: float = 0.1,
    test_frac: float = 0.05):

    dataset = SpatialAnnDataset(adata)
    data = Data(x = dataset.x, edge_index = dataset.edge_index, adj = dataset.adj)

    transform = RandomLinkSplit(
        num_val = val_frac,
        num_test = test_frac,
        is_undirected = True,
        split_labels = True)

    train_data, val_data, test_data = transform(data)

    return train_data, val_data, test_data


def print_progress(epoch, logs, n_epochs=10000, only_val_losses=True):
    """Creates Message for '_print_progress_bar'.
       Parameters
       ----------
       epoch: Integer
            Current epoch iteration.
       logs: Dict
            Dictionary of all current losses.
       n_epochs: Integer
            Maximum value of epochs.
       only_val_losses: Boolean
            If 'True' only the validation dataset losses are displayed, if 'False' additionally the training dataset
            losses are displayed.
       Returns
       -------
    """
    message = ""
    for key in logs:
        if only_val_losses:
            if "val_" in key and "unweighted" not in key:
                message += f" - {key:s}: {logs[key][-1]:7.10f}"
        else:
            if "unweighted" not in key:
                message += f" - {key:s}: {logs[key][-1]:7.10f}"

    _print_progress_bar(epoch + 1, n_epochs, prefix='', suffix=message, decimals=1, length=20)


def _print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """Prints out message with a progress bar.
       Parameters
       ----------
       iteration: Integer
            Current epoch.
       total: Integer
            Maximum value of epochs.
       prefix: String
            String before the progress bar.
       suffix: String
            String after the progress bar.
       decimals: Integer
            Digits after comma for all the losses.
       length: Integer
            Length of the progress bar.
       fill: String
            Symbol for filling the bar.
       Returns
       -------
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_len = int(length * iteration // total)
    bar = fill * filled_len + '-' * (length - filled_len)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()