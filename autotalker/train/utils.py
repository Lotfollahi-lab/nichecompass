import numpy as np


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
        Number of n_epochs which are allowed to have no metric_improvement until the training is stopped.
    reduce_lr_on_plateau:
        If "True", the learning rate gets adjusted by "lr_factor" after a given number of n_epochs with no
        metric_improvement.
    lr_patience:
        Number of n_epochs which are allowed to have no metric_improvement until the learning rate is adjusted.
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

        self.n_epochs = 0
        self.n_epochs_not_improved = 0
        self.n_epochs_not_improved_lr = 0

        self.current_performance = np.inf
        self.best_performance = np.inf
        self.best_performance_state = np.inf

    def update(self, current_metric):
        self.n_epochs += 1
        # Determine whether to continue training
        if self.n_epochs < self.patience:
            continue_training = True
            reduce_lr = False
        elif self.n_epochs_not_improved >= self.patience:
            continue_training = False
            reduce_lr = False
        # Determine whether to reduce the learning rate
        else:
            if self.reduce_lr_on_plateau == False:
                reduce_lr = False
            elif self.n_epochs_not_improved_lr >= self.lr_patience:
                reduce_lr = True
                self.n_epochs_not_improved_lr = 0
            else:
                reduce_lr = False
            
            # Shift
            self.current_performance = current_metric
                
            metric_improvement = self.best_performance - self.current_performance
            
            # Updating best performance
            if metric_improvement > 0:
                self.best_performance = self.current_performance

            # Updating n_epochs not improved
            if metric_improvement < self.metric_improvement_threshold:
                self.n_epochs_not_improved += 1
                self.n_epochs_not_improved_lr += 1
            else:
                self.n_epochs_not_improved = 0
                self.n_epochs_not_improved_lr = 0

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