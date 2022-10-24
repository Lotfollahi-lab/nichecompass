from typing import Literal, Optional, Union

import torch
from anndata import AnnData
from numpy import ndarray

from .basemodelmixin import BaseModelMixin
from .vgaemodelmixin import VGAEModelMixin
from autotalker.modules import VGPGAE
from autotalker.train import Trainer


class Autotalker(BaseModelMixin, VGAEModelMixin):
    """
    Autotalker model class.

    Parameters
    ----------
    adata:
        AnnData object with raw counts stored in 
        ´adata.layers[counts_layer_key]´, sparse adjacency matrix stored in 
        ´adata.obsp[adj_key]´ and binary gene program targets and (optionally) 
        sources masks stored in ´adata.varm[gp_targets_mask_key]´ and 
        ´adata.varm[gp_sources_mask_key]´ respectively (unless gene program 
        masks are passed explicitly to the model via parameters 
        ´gp_targets_mask_key´ and ´gp_sources_mask_key´).
    counts_layer_key:
        Key under which the raw counts are stored in ´adata.layer´.
    adj_key:
        Key under which the sparse adjacency matrix is stored in ´adata.obsp´.
    gp_targets_mask_key:
        Key under which the gene program targets mask is stored in ´adata.varm´. 
        This mask will only be used if no ´gp_targets_mask_key´ is passed 
        explicitly to the model.
    gp_sources_mask_key:
        Key under which the gene program sources mask is stored in ´adata.varm´. 
        This mask will only be used if no ´gp_sources_mask_key´ is passed 
        explicitly to the model.    
    include_edge_recon_loss:
        If `True`, include the edge reconstruction loss in the loss 
        optimization of the model.
    include_gene_expr_recon_loss:
        If `True`, include the gene expression reconstruction loss in the 
        loss optimization.
    log_variational:
        If ´True´, transform x by log(x+1) prior to encoding for numerical 
        stability. Not normalization.
    node_label_method:
        Node label method that will be used for gene expression reconstruction. 
        If ´self´, use only the input features of the node itself as node labels
        for gene expression reconstruction. If ´one-hop-sum´, use a 
        concatenation of the node's input features with the sum of the input 
        features of all nodes in the node's one-hop neighborhood. If 
        ´one-hop-norm´, use a concatenation of the node`s input features with
        the node's one-hop neighbors input features normalized as per Kipf, T. 
        N. & Welling, M. Semi-Supervised Classification with Graph Convolutional
        Networks. arXiv [cs.LG] (2016))
    n_hidden_encoder:
        Number of nodes in the encoder hidden layer.
    dropout_rate_encoder:
        Probability that nodes will be dropped in the encoder during training.
    dropout_rate_graph_decoder:
        Probability that nodes will be dropped in the graph decoder during 
        training.
    gp_targets_mask:
        Gene program targets mask that is directly passed to the model (if not 
        ´None´, this mask will have prevalence over a gene program targets mask
        stored in ´adata.varm[gp_targets_mask_key]´).
    gp_sources_mask:
        Gene program sources mask that is directly passed to the model (if not 
        ´None´, this mask will have prevalence over a gene program sources mask
        stored in ´adata.varm[gp_sources_mask_key]´).    
    """
    def __init__(self,
                 adata: AnnData,
                 counts_layer_key="counts",
                 adj_key: str="spatial_connectivities",
                 gp_targets_mask_key: str="autotalker_gp_targets",
                 gp_sources_mask_key: str="autotalker_gp_sources",
                 include_edge_recon_loss: bool=True,
                 include_gene_expr_recon_loss: bool=True,
                 log_variational: bool=True,
                 node_label_method: Literal["self",
                                            "one-hop-sum",
                                            "one-hop-norm",
                                            "one-hop-attention"]="one-hop-attention",
                 n_hidden_encoder: int=256,
                 dropout_rate_encoder: float=0.0,
                 dropout_rate_graph_decoder: float=0.0,
                 gp_targets_mask: Optional[Union[ndarray, list]]=None,
                 gp_sources_mask: Optional[Union[ndarray, list]]=None,
                 n_addon_gps: int=0):
        self.adata = adata
        self.counts_layer_key_ = counts_layer_key
        self.adj_key_ = adj_key
        self.gp_targets_mask_key_ = gp_targets_mask_key
        self.gp_sources_mask_key_ = gp_sources_mask_key
        self.include_edge_recon_loss_ = include_edge_recon_loss
        self.include_gene_expr_recon_loss_ = include_gene_expr_recon_loss
        self.log_variational_ = log_variational
        self.node_label_method_ = node_label_method
        self.n_input_ = adata.n_vars
        self.n_output_ = adata.n_vars
        if node_label_method != "self":
            self.n_output_ *= 2
        self.n_hidden_encoder_ = n_hidden_encoder
        self.dropout_rate_encoder_ = dropout_rate_encoder
        self.dropout_rate_graph_decoder_ = dropout_rate_graph_decoder

        # Retrieve gene program masks
        if gp_targets_mask is None:
            if gp_targets_mask_key in adata.varm:
                gp_targets_mask = adata.varm[gp_targets_mask_key].T
            else:
                raise ValueError("Please explicitly provide a ´gp_targets_mask´"
                                 " to the model or specify an adequate "
                                 "´gp_targets_mask_key´ for your adata object. "
                                 "If you do not want to mask gene expression "
                                 "reconstruction, you can create a mask of 1s "
                                 "that allows all gene program latent nodes "
                                 "to reconstruct all genes by passing a mask "
                                 "created with ´mask = "
                                 "np.ones((n_latent, n_output))´).")
        self.gp_mask_ = torch.tensor(gp_targets_mask, dtype=torch.float32)
        
        if node_label_method != "self":
            if gp_sources_mask is None:
                if gp_sources_mask_key in adata.varm:
                    gp_sources_mask = adata.varm[gp_sources_mask_key].T
                else:
                    raise ValueError("Please explicitly provide a "
                                     "´gp_sources_mask´ to the model or specify"
                                     " an adequate ´gp_sources_mask_key´ for "
                                     "your adata object.")
            # Horizontally concatenate targets and sources masks
            self.gp_mask_ = torch.cat(
                (self.gp_mask_, torch.tensor(gp_sources_mask, 
                dtype=torch.float32)), dim=1)
        
        self.n_gps_ = len(self.gp_mask_)
        self.n_addon_gps_ = n_addon_gps
        
        # Validate counts layer key and counts values
        if counts_layer_key not in adata.layers:
            raise ValueError("Please specify an adequate ´counts_layer_key´. "
                             "By default the raw counts are assumed to be "
                             f"stored in adata.layers['counts'].")
        if include_gene_expr_recon_loss or log_variational:
            if (adata.layers[counts_layer_key] < 0).sum() > 0:
                raise ValueError("Please make sure that "
                                 "´adata.layers[counts_layer_key]´ contains the"
                                 " raw counts (not log library size "
                                 "normalized) if ´include_gene_expr_recon_loss´"
                                 " is ´True´ or ´log_variational´ is ´True´.")

        # Validate adjacency key
        if adj_key not in adata.obsp:
            raise ValueError("Please specify an adequate ´adj_key´.")
        
        # Initialize model with module
        self.model = VGPGAE(
            n_input=self.n_input_,
            n_hidden_encoder=self.n_hidden_encoder_,
            n_latent=self.n_gps_,
            n_addon_latent=self.n_addon_gps_,
            n_output=self.n_output_,
            gene_expr_decoder_mask=self.gp_mask_,
            dropout_rate_encoder=self.dropout_rate_encoder_,
            dropout_rate_graph_decoder=self.dropout_rate_graph_decoder_,
            include_edge_recon_loss=self.include_edge_recon_loss_,
            include_gene_expr_recon_loss=self.include_gene_expr_recon_loss_,
            node_label_method=self.node_label_method_,
            log_variational=self.log_variational_)

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
              edge_batch_size: int=64,
              mlflow_experiment_id: Optional[str]=None,
              **trainer_kwargs):
        """
        Train the Autotalker model.
        
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
            The rest of the data will be used as training or test set (as 
            defined in edge_test_ratio) on edge-level.
        edge_test_ratio:
            Fraction of the data that is used as test set on edge-level.
        node_val_ratio:
            Fraction of the data that is used as validation set on node-level.
            The rest of the data will be used as training set on node-level.
        edge_batch_size:
            Batch size for the edge-level dataloaders.
        mlflow_experiment_id:
            ID of the Mlflow experiment used for tracking training parameters
            and metrics.
        trainer_kwargs:
            Kwargs for the model Trainer.
        """
        self.trainer = Trainer(
            adata=self.adata,
            model=self.model,
            counts_layer_key=self.counts_layer_key_,
            adj_key=self.adj_key_,
            node_label_method=self.node_label_method_,
            edge_val_ratio=edge_val_ratio,
            edge_test_ratio=edge_test_ratio,
            node_val_ratio=node_val_ratio,
            node_test_ratio=0.0,
            edge_batch_size=edge_batch_size,
            **trainer_kwargs)

        self.trainer.train(n_epochs=n_epochs,
                           lr=lr,
                           weight_decay=weight_decay,
                           mlflow_experiment_id=mlflow_experiment_id,)
        
        self.is_trained_ = True