import logging
from typing import Optional, Union

import anndata as ad
import numpy as np
import torch

from ._basemodel import BaseModel
from ._vgaemodelmixin import VGAEModelMixin
from autotalker.modules import VGAE, VGPGAE
from autotalker.train import VGAETrainer, VGPGAETrainer


logger = logging.getLogger(__name__)


class Autotalker(BaseModel, VGAEModelMixin):
    """
    Autotalker model class.

    Parameters
    ----------
    adata:
        AnnData object with sparse adjacency matrix stored in 
        data.obsp[adj_key].
    adj_key:
        Key under which the sparse adjacency matrix is stored in adata.obsp.
    cell_type_key:
        Key under which the cell types are stored in data.obs.
    n_hidden:
        Number of nodes in the VGAE hidden layer.
    n_latent:
        Number of nodes in the latent space.
    dropout_rate:
        Probability that nodes will be dropped during training.
    """
    def __init__(self,
                 adata: ad.AnnData,
                 mask: Optional[Union[np.ndarray, list]]=None,
                 mask_key: str="I",
                 adj_key: str="spatial_connectivities",
                 cell_type_key: str="cell_type",
                 n_hidden: int=32,
                 dropout_rate: float=0.0,
                 expr_decoder_recon_loss: str="mse",
                 mlflow_experiment_id: Optional[str]=None,
                 **model_kwargs):

        if mask is None:
            if mask_key in adata.varm:
                mask = adata.varm[mask_key].T
            else:
                raise ValueError("Please provide a mask or specify an adquate "
                                 "mask key.")

        self.adata = adata
        self.mask_ = torch.tensor(mask).float()
        self.mask_key_ = mask_key
        self.adj_key_ = adj_key
        self.n_latent_ = len(self.mask_)
        self.cell_type_key_ = cell_type_key
        self.n_input_ = adata.n_vars
        self.n_output_ = adata.n_vars
        self.n_hidden_ = n_hidden
        self.dropout_rate_ = dropout_rate
        self.expr_decoder_recon_loss_ = expr_decoder_recon_loss

        self.encoder_layer_sizes_ = [self.n_input_,
                                     self.n_hidden_,
                                     self.n_latent_]
        self.expr_decoder_layer_sizes_ = [self.n_latent_,
                                          self.n_output_]

        self.model = VGAE(n_input=self.n_input_,
                          n_hidden=self.n_hidden_,
                          n_latent=self.n_latent_,
                          dropout_rate=self.dropout_rate_)

        self.is_trained_ = False
        self.init_params_ = self._get_init_params(locals())


    def train(self,
              n_epochs: int=200,
              lr: float=0.01,
              weight_decay: float=0,
              val_frac: float=0.1,
              test_frac: float=0.05,
              mlflow_experiment_id: Optional[str]=None,
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
            Weight decay (L2 penalty).
        kwargs:
            Kwargs for the trainer.
        """
        self.trainer = VGAETrainer(adata=self.adata,
                                   model=self.model,
                                   adj_key=self.adj_key_,
                                   val_frac=val_frac,
                                   test_frac=test_frac,
                                   mlflow_experiment_id=mlflow_experiment_id,
                                   **trainer_kwargs)
        self.trainer.train(n_epochs, lr, weight_decay)
        self.is_trained_ = True