import inspect
import os
from typing import Optional

import anndata as ad
import numpy as np
import pickle
import torch

from ._utils import _validate_var_names
from ._utils import _initialize_model
from ._utils import _load_saved_files


class BaseModelMixin:
    """
    BaseModelMixin class for basic model functionalities. Adapted from 
    https://github.com/theislab/scarches and 
    https://github.com/scverse/scvi-tools.
    """
    def _get_user_attributes(self):
        """
        Returns all the self attributes defined in a model class, eg, 
        self.is_trained_
        """
        attributes = inspect.getmembers(
            self, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if not (
            a[0].startswith("__") and a[0].endswith("__"))]
        return attributes

    def _get_public_attributes(self):
        public_attributes = self._get_user_attributes()
        public_attributes = {a[0]: a[1] for a in public_attributes if a[0][-1] == "_"}
        return public_attributes

    def _get_init_params(self, locals):
        """
        Returns the model init signature with associated passed in values
        except the anndata objects passed in.
        """
        init = self.__init__
        sig = inspect.signature(init)
        init_params = [p for p in sig.parameters]
        user_params = {p: locals[p] for p in locals if p in init_params}
        user_params = {
            k: v for (k, v) in user_params.items() if not isinstance(v, AnnData)
        }
        return user_params

    def save(
            self,
            dir_path: str,
            overwrite: bool=False,
            save_anndata: bool=False,
            **anndata_write_kwargs):
        """
        Save the state of the model.
        Neither the trainer optimizer state nor the trainer history are saved.
        Parameters
        ----------
        dir_path
             Path to a directory.
        overwrite
             Overwrite existing data or not. If `False` and directory
             already exists at `dir_path`, error will be raised.
        save_anndata:
             If "True", also saves the AnnData object.
        anndata_write_kwargs:
             Kwargs for anndata write function
        """
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(
                f"{dir_path} already exists."
                "Please provide another directory for saving.")

        model_save_path = os.path.join(dir_path, "model_params.pt")
        attr_save_path = os.path.join(dir_path, "attr.pkl")
        var_names_save_path = os.path.join(dir_path, "var_names.csv")

        if save_anndata:
            self.adata.write(
                os.path.join(dir_path, "adata.h5ad"), **anndata_write_kwargs)

        var_names = self.adata.var_names.astype(str).to_numpy()
        public_attributes = self._get_public_attributes()
        
        np.savetxt(var_names_save_path, var_names, fmt="%s")
        torch.save(self.model.state_dict(), model_save_path)
        with open(attr_save_path, "wb") as f:
            pickle.dump(public_attributes, f)


    @classmethod
    def load(
            cls,
            dir_path: str,
            adata: Optional[ad.AnnData]=None,
            adata_file_name: Optional[str]="adata.h5ad",
            use_cuda: bool=False):
        """
        Instantiate a model from saved output.
        
        Parameters
        ----------
        dir_path:
            Path to saved outputs.
        adata:
            AnnData organized in the same way as data used to train model.
            It is not necessary to run :func:`~scvi.data.setup_anndata`,
            as AnnData is validated against the saved `scvi` setup dictionary.
            If None, will check for and load anndata saved with the model.
        use_cuda:
            Whether to load model on GPU.
        Returns
        -------
        Model with loaded state dictionaries.
        """
        load_adata = adata is None
        use_cuda = use_cuda and torch.cuda.is_available()
        map_location = torch.device("cpu") if use_cuda is False else None

        attr_dict, var_names, model_state_dict, new_adata = _load_saved_files(
            dir_path, load_adata, map_location=map_location)
        adata = new_adata if new_adata is not None else adata

        _validate_var_names(adata, var_names)
        model = _initialize_model(cls, adata, attr_dict, use_cuda)

        # set saved attrs for loaded model
        for attr, val in attr_dict.items():
            setattr(model, attr, val)

        model.model.load_state_dict(model_state_dict)
        if use_cuda:
            model.model.cuda()

        model.model.eval()

        return model


class VAEModelMixin:
    def sampling(self, mu, log_var):
        """Samples from standard Normal distribution and applies re-parametrization trick.
           It is actually sampling from latent space distributions with N(mu, var), computed by encoder.
           Parameters
           ----------
           mu: torch.Tensor
                Torch Tensor of Means.
           log_var: torch.Tensor
                Torch Tensor of log. variances.
           Returns
           -------
           Torch Tensor of sampled data.
        """
        var = torch.exp(log_var) + 1e-4
        return Normal(mu, var.sqrt()).rsample()

    def get_latent(self, x, c=None, mean=False, mean_var=False):
        """Map `x` in to the latent space. This function will feed data in encoder  and return  z for each sample in
           data.
           Parameters
           ----------
           x:  torch.Tensor
                Torch Tensor to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
           c: torch.Tensor
                Torch Tensor of condition labels for each sample.
           mean: boolean
           Returns
           -------
           Returns Torch Tensor containing latent space encoding of 'x'.
        """
        x_ = torch.log(1 + x)
        if self.recon_loss == 'mse':
            x_ = x
        z_mean, z_log_var = self.encoder(x_, c)
        latent = self.sampling(z_mean, z_log_var)
        if mean:
            return z_mean
        elif mean_var:
            return (z_mean, torch.exp(z_log_var) + 1e-4)
        return latent

    def get_y(self, x, c=None):
        """Map `x` in to the y dimension (First Layer of Decoder). This function will feed data in encoder  and return
           y for each sample in data.
           Parameters
           ----------
           x:  torch.Tensor
                Torch Tensor to be mapped to latent space. `x` has to be in shape [n_obs, input_dim].
           c: torch.Tensor
                Torch Tensor of condition labels for each sample.
           Returns
           -------
           Returns Torch Tensor containing output of first decoder layer.
        """
        x_ = torch.log(1 + x)
        if self.recon_loss == 'mse':
            x_ = x
        z_mean, z_log_var = self.encoder(x_, c)
        latent = self.sampling(z_mean, z_log_var)
        output = self.decoder(latent, c)
        return output[-1]
