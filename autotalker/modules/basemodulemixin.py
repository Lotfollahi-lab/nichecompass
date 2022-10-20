import torch


class BaseModuleMixin:
    """
    Base module mix in class containing universal module functionalities.
    """
    def load_and_expand_state_dict(self, model_state_dict):
        """
        Load model state dictionary into model and expand it to account for
        architectural changes through e.g. add-on nodes. Adapted from 
        https://github.com/theislab/scarches/blob/master/scarches/models/base/_base.py#L92.
        """
        load_state_dict = model_state_dict.copy() # old model architecture state dict
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
                # old model architecture parameter tensors; updates are necessary
                load_param_tensor = load_param_tensor.to(device)
                n_dims = len(new_param_tensor.shape)
                idx_slicers = [slice(None)] * n_dims
                for i in range(n_dims):
                    dim_diff = new_param_tensor.shape[i] - load_param_tensor.shape[i]
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
                print("nope")
                load_state_dict[key] = new_param_tensor

        self.load_state_dict(load_state_dict)