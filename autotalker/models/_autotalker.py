import logging
from typing import Literal, Optional, Union

import anndata as ad
import numpy as np
import torch

from ._basemodelmixin import BaseModelMixin
from ._vgaemodelmixin import VGAEModelMixin
from autotalker.modules import VGAE, VGPGAE
from autotalker.train import Trainer


logger = logging.getLogger(__name__)


class Autotalker(BaseModelMixin, VGAEModelMixin):
    """
    Autotalker model class.

    Parameters
    ----------
    adata:
        AnnData object with sparse adjacency matrix stored in 
        adata.obsp[adj_key] and binary gene program mask stored in 
        adata.varm[gp_mask_key] (unless gene program mask is passed directly to
        the model).
    autotalker_module:
        Autotalker module that is used for model training.
    n_hidden_encoder:
        Number of nodes in the encoder hidden layer.
    dropout_rate_encoder:
        Probability that nodes will be dropped in the encoder during training.
    gp_mask:
        Gene program mask that is directly passed to the model (if not None, 
        this mask will have prevalence over a gene program mask stored in 
        adata.varm[gp_mask_key]).
    gp_mask_key:
        Key under which the gene program mask is stored in adata.varm. This mask
        will only be used if no mask is passed directly to the model.
    adj_key:
        Key under which the sparse adjacency matrix is stored in adata.obsp.
    mlflow_experiment_id:
        ID of the mlflow experiment if tracking of model training is desired.
    """
    def __init__(self,
                 adata: ad.AnnData,
                 autotalker_module: Literal["VGAE", "VGPGAE"]="VGPGAE",
                 n_hidden_encoder: int=32,
                 dropout_rate_encoder: float=0.0,
                 dropout_rate_graph_decoder: float=0.0,
                 gp_mask: Optional[Union[np.ndarray, list]]=None,
                 gp_mask_key: str="autotalker_gp_mask",
                 adj_key: str="spatial_connectivities",
                 mlflow_experiment_id: Optional[str]=None,
                 **model_kwargs):
        self.adata = adata
        self.autotalker_module_ = autotalker_module
        self.n_input_ = adata.n_vars
        self.n_output_ = adata.n_vars
        self.n_hidden_encoder_ = n_hidden_encoder
        self.dropout_rate_encoder_ = dropout_rate_encoder
        self.dropout_rate_graph_decoder_ = dropout_rate_graph_decoder
        self.gp_mask_key_ = gp_mask_key
        self.adj_key_ = adj_key

        # Retrieve gene program mask
        if gp_mask is None:
            if gp_mask_key in adata.varm:
                gp_mask = adata.varm[gp_mask_key].T
            else:
                raise ValueError("Please directly provide a gene program mask "
                                 "to the model or specify an adquate mask key "
                                 "for your adata object. If you do not want to "
                                 "mask gene expression reconstruction, you can "
                                 "create a mask of 1s that allows all gene "
                                 "programs (latent nodes) to reconstruct all "
                                 "genes by passing a mask created with ´mask "
                                 "= np.ones((n_latent, len(adata.var)))´).")

        self.gp_mask_ = torch.tensor(gp_mask, dtype=torch.float32)
        self.n_latent_ = len(self.gp_mask_)

        if self.autotalker_module_ == "VGAE":
            self.model = VGAE(
                n_input=self.n_input_,
                n_hidden_encoder=self.n_hidden_encoder_,
                n_latent=self.n_latent_,
                dropout_rate_encoder=self.dropout_rate_encoder_,
                dropout_rate_graph_decoder=self.dropout_rate_graph_decoder_)
        elif self.autotalker_module_ == "VGPGAE":
            self.model = VGPGAE(
                n_input=self.n_input_,
                n_hidden_encoder=self.n_hidden_encoder_,
                n_latent=self.n_latent_,
                gene_expr_decoder_mask=self.gp_mask_,
                dropout_rate_encoder=self.dropout_rate_encoder_,
                dropout_rate_graph_decoder=self.dropout_rate_graph_decoder_)

        self.is_trained_ = False
        self.init_params_ = self._get_init_params(locals())


    def train(self,
              n_epochs: int=200,
              lr: float=0.01,
              weight_decay: float=0,
              edge_val_ratio: float=0.1,
              edge_test_ratio: float=0.05,
              edge_batch_size: int=128,
              node_val_ratio: float=0.1,
              node_test_ratio: float=0.0,
              node_batch_size: int=32,
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
        self.trainer = Trainer(adata=self.adata,
                               model=self.model,
                               adj_key=self.adj_key_,
                               edge_val_ratio=edge_val_ratio,
                               edge_test_ratio=edge_test_ratio,
                               edge_batch_size=edge_batch_size,
                               node_val_ratio=node_val_ratio,
                               node_test_ratio=node_test_ratio,
                               node_batch_size=node_batch_size,
                               **trainer_kwargs)
                                    
        self.trainer.train(n_epochs=n_epochs,
                           lr=lr,
                           weight_decay=weight_decay,
                           mlflow_experiment_id=mlflow_experiment_id,)
        self.is_trained_ = True