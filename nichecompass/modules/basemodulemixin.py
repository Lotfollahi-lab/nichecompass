"""
This module contains generic module functionalities, added as a Mixin to the
Variational Gene Program Graph Autoencoder module.
"""

import inspect
from collections import OrderedDict

import torch


class BaseModuleMixin:
    """
    Base module mix in class containing universal module functionalities.

    Parts of the implementation are adapted from
    https://github.com/scverse/scvi-tools/blob/master/scvi/model/base/_base_model.py#L63
    (01.10.2022).
    """
    def _get_user_attributes(self) -> list:
        """
        Get all the attributes defined in a module instance.

        Returns
        ----------
        attributes:
            Attributes defined in a module instance.
        """
        attributes = inspect.getmembers(
            self, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if not (
            a[0].startswith("__") and a[0].endswith("__"))]
        return attributes

    def _get_public_attributes(self) -> dict:
        """
        Get only public attributes defined in a module instance. By convention
        public attributes have a trailing underscore.

        Returns
        ----------
        public_attributes:
            Public attributes defined in a module instance.
        """
        public_attributes = self._get_user_attributes()
        public_attributes = {a[0]: a[1] for a in public_attributes if 
                             a[0][-1] == "_"}
        return public_attributes

    def load_and_expand_state_dict(self,
                                   model_state_dict: OrderedDict):
        """
        Load model state dictionary into model and expand it to account for
        architectural changes through e.g. add-on nodes. 
        
        Parts of the implementation are adapted from 
        https://github.com/theislab/scarches/blob/master/scarches/models/base/_base.py#L92
        (01.10.2022).
        """
        load_state_dict = model_state_dict.copy() # old model architecture state 
        # dict
        new_state_dict = self.state_dict() # new model architecture state dict
        device = next(self.parameters()).device

        # Update parameter tensors from old model architecture with changes from
        # new model architecture
        for key, load_param_tensor in load_state_dict.items():
            new_param_tensor = new_state_dict[key]
            if new_param_tensor.size() == load_param_tensor.size():
                continue # nothing needs to be updated
            else:
                # new model architecture parameter tensors are different from
                # old model architecture parameter tensors; updates are 
                # necessary
                load_param_tensor = load_param_tensor.to(device)
                n_dims = len(new_param_tensor.shape)
                idx_slicers = [slice(None)] * n_dims
                for i in range(n_dims):
                    dim_diff = (new_param_tensor.shape[i] - 
                                load_param_tensor.shape[i])
                    idx_slicers[i] = slice(-dim_diff, None)
                    if dim_diff > 0:
                        break
                expanded_param_tensor = torch.cat(
                    [load_param_tensor, new_param_tensor[tuple(idx_slicers)]],
                    dim=i)
                load_state_dict[key] = expanded_param_tensor

        # Add parameter tensors from new model architecture to old model 
        # architecture state dict
        for key, new_param_tensor in new_state_dict.items():
            if key not in load_state_dict:
                load_state_dict[key] = new_param_tensor

        self.load_state_dict(load_state_dict)
        