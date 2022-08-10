from lib2to3.pytree import Base
from ..autotalker.train._trainer import BaseTrainer

class AutotalkerTrainer(BaseTrainer):
    def __init__(
        self,
        adata,
        model,
        **kwargs):
    super().__init__(model, adata, **kwargs)

    def loss():
        
        

