from dataclasses import make_dataclass
import torch
import torch.nn as nn

class Trainer:
    """
    Trainer class.
    """
    def __init__(self,
                 model,
                 adata,
                 batch_size: int = 128,
                 use_early_stopping: bool = True,
                 **kwargs):
        self.adata = adata
        self.model = model
        self.batch_size = batch_size
        self.use_early_stopping = use_early_stopping

    self.train_data, self.valid_data = make_dataset(
        self.adata,
        fraction_train=self.fraction_train
    )

    def initialize_loaders(self):
        
