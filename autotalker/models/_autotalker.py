import logging

import anndata as ad

from autotalker.modules import VGAE
from autotalker.train import Trainer


logger = logging.getLogger(__name__)


class Autotalker():
    """
    Autotalker model class.

    Parameters
    ----------
    adata:
        AnnData object with sparse adjacency matrix stored in 
        data.obsp[adj_key].
    adj_key:
        Key under which the sparse adjacency matrix is stored in adata.obsp.
    n_hidden:
        Number of nodes in the VGAE hidden layer.
    n_latent:
        Number of nodes in the latent space.
    dropout_rate:
        Probability that nodes will be dropped during training.
    """
    def __init__(
            self,
            adata: ad.AnnData,
            adj_key: str="spatial_connectivities",
            n_hidden: int=32,
            n_latent: int=16,
            dropout_rate: float=0,
            **model_kwargs):
        self.adata = adata
        self.adj_key = adj_key
        self.n_input = adata.n_vars
        self.n_hidden = n_hidden
        self.n_latent = n_latent
        self.dropout_rate = dropout_rate

        self.model = VGAE(n_input = self.n_input,
                          n_hidden = self.n_hidden,
                          n_latent = self.n_latent,
                          dropout_rate = self.dropout_rate)

        self.is_trained_ = False


    def train(
            self,
            n_epochs: int=200,
            lr: float=0.01,
            weight_decay: float=0,
            val_frac: float=0.1,
            test_frac: float=0.05,
            **trainer_kwargs):
        """
        Train the model.
        
        Parameters
        ----------
        n_epochs:
            Number of epochs.
        lr:
            Learning rate.
        weight_decay:
            Weight decay (L2 penalty) used with optimizer.
        kwargs:
            kwargs for the trainer.
        """
        self.trainer = Trainer(
                self.adata,
                self.adj_key,
                self.model,
                val_frac,
                test_frac,
                **trainer_kwargs)
        self.trainer.train(n_epochs, lr, weight_decay)
        self.is_trained_ = True