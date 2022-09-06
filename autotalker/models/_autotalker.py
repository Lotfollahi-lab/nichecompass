import logging
from typing import Literal, Optional, Union

import torch
from anndata import AnnData
from numpy import ndarray

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
    adj_key:
        Key under which the sparse adjacency matrix is stored in adata.obsp.
    gp_mask_key:
        Key under which the gene program mask is stored in adata.varm. This mask
        will only be used if no mask is passed directly to the model.
    include_edge_recon_loss:
        If `True`, include the edge reconstruction loss in the loss 
        optimization of the model.
    include_gene_expr_recon_loss:
        If `True`, include the gene expression reconstruction loss in the 
        loss optimization.
    node_label_method:
        Node label method that will be used for gene expression reconstruction 
        if ´include_gene_exp_recon_loss'is ´True´. If  ´self´, use only the 
        input features of the node itself as node labels for gene expression 
        reconstruction. If ´one-hop´, use a concatenation of the node's input 
        features with a sum of the input features of all nodes in the node's 
        one-hop neighborhood.
    n_hidden_encoder:
        Number of nodes in the encoder hidden layer.
    dropout_rate_encoder:
        Probability that nodes will be dropped in the encoder during training.
    dropout_rate_graph_decoder:
        Probability that nodes will be dropped in the graph decoder during 
        training.
    gp_mask:
        Gene program mask that is directly passed to the model (if not None, 
        this mask will have prevalence over a gene program mask stored in 
        adata.varm[gp_mask_key]).
    """
    def __init__(self,
                 adata: AnnData,
                 adj_key: str="spatial_connectivities",
                 gp_targets_mask_key: str="autotalker_gp_targets",
                 gp_sources_mask_key: str="autotalker_gp_sources",
                 include_edge_recon_loss: bool=True,
                 include_gene_expr_recon_loss: bool=True,
                 node_label_method: Literal["self", "one-hop"]="one-hop",
                 n_hidden_encoder: int=32,
                 dropout_rate_encoder: float=0.0,
                 dropout_rate_graph_decoder: float=0.0,
                 gp_targets_mask: Optional[Union[ndarray, list]]=None,
                 gp_sources_mask: Optional[Union[ndarray, list]]=None):
        self.adata = adata
        self.adj_key_ = adj_key
        self.gp_targets_mask_key_ = gp_targets_mask_key
        self.gp_sources_mask_key_ = gp_sources_mask_key
        self.include_edge_recon_loss = include_edge_recon_loss
        self.include_gene_expr_recon_loss = include_gene_expr_recon_loss
        self.node_label_method = node_label_method
        self.n_input_ = adata.n_vars
        self.n_output_ = adata.n_vars
        if node_label_method == "one-hop":
            self.n_output_ *= 2
        self.n_hidden_encoder_ = n_hidden_encoder
        self.dropout_rate_encoder_ = dropout_rate_encoder
        self.dropout_rate_graph_decoder_ = dropout_rate_graph_decoder

        # Retrieve gene program mask
        if gp_targets_mask is None:
            if gp_targets_mask_key in adata.varm:
                gp_targets_mask = adata.varm[gp_targets_mask_key].T
            else:
                raise ValueError("Please directly provide a gene program "
                                 "targets mask to the model or specify an"
                                 " adequate gp targets mask key for your adata "
                                 "object. If you do not want to mask gene "
                                 "expression reconstruction, you can create a "
                                 "mask of 1s that allows all gene programs "
                                 "(latent nodes) to reconstruct all genes by "
                                 "passing a mask created with ´mask = "
                                 "np.ones((n_latent, n_output))´).")
        self.gp_mask_ = torch.tensor(gp_targets_mask, dtype=torch.float32)
        
        if node_label_method != "self":
            if gp_sources_mask is None:
                if gp_sources_mask_key in adata.varm:
                    gp_sources_mask = adata.varm[gp_sources_mask_key].T
                else:
                    raise ValueError("Please directly provida a gene program "
                                     "sources mask to the model or specify an"
                                     " adequate gp sources mask key for your "
                                     "adata object.")
            self.gp_mask_ = torch.cat(
                (self.gp_mask_, torch.tensor(gp_sources_mask)), dim=1)
        
        self.n_latent_ = len(self.gp_mask_)

        # Validate adjacency key
        if adj_key not in adata.obsp:
            raise ValueError("Please specify an adequate adjacency key.")
        
        # Initialize model with module
        self.model = VGPGAE(
            n_input=self.n_input_,
            n_hidden_encoder=self.n_hidden_encoder_,
            n_latent=self.n_latent_,
            n_output=self.n_output_,
            gene_expr_decoder_mask=self.gp_mask_,
            dropout_rate_encoder=self.dropout_rate_encoder_,
            dropout_rate_graph_decoder=self.dropout_rate_graph_decoder_)

        self.is_trained_ = False
        # Store init params for saving and loading
        self.init_params_ = self._get_init_params(locals())

    def train(self,
              n_epochs: int=200,
              lr: float=0.01,
              weight_decay: float=0,
              edge_val_ratio: float=0.1,
              edge_test_ratio: float=0.05,
              node_val_ratio: float=0.1,
              node_test_ratio: float=0.0,
              edge_batch_size: int=64,
              mlflow_experiment_id: Optional[str]=None,
              **trainer_kwargs):
        """
        Train the model.
        
        Parameters
        ----------
        n_epochs:
            Number of epochs for model training.
        lr:
            Learning rate for model training.
        weight_decay:
            Weight decay (L2 penalty) for model training.
        edge_val_ratio:
            Fraction of the data that is used as validation set on edge-level.
        edge_test_ratio:
            Fraction of the data that is used as test set on edge-level.
        node_val_ratio:
            Fraction of the data that is used as validation set on node-level.
        node_test_ratio:
            Fraction of the data that is used as test set on node-level.
        edge_batch_size:
            Batch size for the edge-level dataloaders.
        mlflow_experiment_id:
            ID of the Mlflow experiment used for tracking training parameters
            and metrics.
        trainer_kawrgs:
            Kwargs for the model Trainer.
        """
        self.trainer = Trainer(
            adata=self.adata,
            model=self.model,
            adj_key=self.adj_key_,
            node_label_method=self.node_label_method,
            edge_val_ratio=edge_val_ratio,
            edge_test_ratio=edge_test_ratio,
            node_val_ratio=node_val_ratio,
            node_test_ratio=node_test_ratio,
            edge_batch_size=edge_batch_size,                 
            include_edge_recon_loss=self.include_edge_recon_loss,
            include_gene_expr_recon_loss=self.include_gene_expr_recon_loss,
            **trainer_kwargs)

        self.trainer.train(n_epochs=n_epochs,
                           lr=lr,
                           weight_decay=weight_decay,
                           mlflow_experiment_id=mlflow_experiment_id,)
        
        self.is_trained_ = True