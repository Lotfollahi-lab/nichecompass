import anndata as ad
import mlflow
import torch.nn as nn

from ._basetrainer import BaseTrainer


class VGAETrainer(BaseTrainer):
    """
    VGAE trainer class.
    
    Parameters
    ----------
    Same as for BaseTrainer class.
    """
    def __init__(self,
                 adata: ad.AnnData,
                 model: nn.Module,
                 **kwargs):
        super().__init__(adata, model, **kwargs)

    def on_training_start(self):
        if self.mlflow_experiment_id is not None:
            mlflow.log_param("n_hidden", self.model.n_hidden)
            mlflow.log_param("n_latent", self.model.n_latent)
            mlflow.log_param("dropout_rate", self.model.dropout_rate)