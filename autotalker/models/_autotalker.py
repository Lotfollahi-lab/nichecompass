import logging

import anndata as ad
import numpy as np
import pandas as pd
import torch


from autotalker.modules import VGAE
from autotalker.train import Trainer

logger = logging.getLogger(__name__)


class Autotalker():
    """
    Autotalker model.

    Parameters
    ----------
    Returns
    ----------
    """
    def __init__(self,
                 adata: ad.AnnData,
                 n_hidden: int = 32,
                 n_latent: int = 16,
                 dropout_rate: float = 0,
                 **model_kwargs):
        self.adata = adata
        self.n_input = adata.n_vars
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.dropout_rate = dropout_rate

        self.model = VGAE(n_input = self.n_input,
                          n_hidden = self.n_hidden,
                          n_latent = self.n_latent,
                          dropout_rate = self.dropout_rate)

        self.is_trained_ = False


    def train(self,
              n_epochs: int = 200,
              lr: float = 0.01,
              weight_decay: float = 0,
              val_frac: float = 0.1,
              test_frac: float = 0.05,
              **kwargs):
        """
        Train the model.
        
        Parameters
        ----------
        n_epochs:
            Number of epochs.
        lr:
            Learning rate.
        kwargs:
            kwargs for the trainer.
        """
        self.trainer = Trainer(self.adata,
                               self.model,
                               val_frac = val_frac,
                               test_frac = test_frac,
                               **kwargs)
        self.trainer.train(n_epochs, lr)
        self.is_trained_ = True