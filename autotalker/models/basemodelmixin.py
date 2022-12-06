"""
This module contains generic base model functionalities, added as a Mixin to the 
Autotalker model.
"""

import inspect
import os
import warnings
from typing import Optional

import numpy as np
import pickle
import scipy.sparse as sp
import torch
from anndata import AnnData

from .utils import initialize_model, load_saved_files, validate_var_names


class BaseModelMixin():
    """
    Base model mix in class for universal model functionalities. 
    
    Parts of the implementation are adapted from
    https://github.com/theislab/scarches/blob/master/scarches/models/base/_base.py#L15
    (01.10.2022) and 
    https://github.com/scverse/scvi-tools/blob/master/scvi/model/base/_base_model.py#L63
    (01.10.2022).
    """
    def _get_user_attributes(self) -> list:
        """
        Get all the attributes defined in a model instance, for example 
        self.is_trained_.

        Returns
        ----------
        attributes:
            Attributes defined in a model instance.
        """
        attributes = inspect.getmembers(
            self, lambda a: not (inspect.isroutine(a)))
        attributes = [a for a in attributes if not (
            a[0].startswith("__") and a[0].endswith("__"))]
        return attributes

    def _get_public_attributes(self) -> dict:
        """
        Get only public attributes defined in a model instance. By convention
        public attributes have a trailing underscore.

        Returns
        ----------
        public_attributes:
            Public attributes defined in a model instance.
        """
        public_attributes = self._get_user_attributes()
        public_attributes = {a[0]: a[1] for a in public_attributes if 
                             a[0][-1] == "_"}
        return public_attributes

    def _get_init_params(self, locals: dict) -> dict:
        """
        Get the model init signature with associated passed in values from 
        locals (except the AnnData object passed in).

        Parameters
        ----------
        locals:
            Dictionary returned by calling the ´locals()´ method.

        Returns
        ----------
        user_params:
            Model initialization attributes defined in a model instance.
        """
        init = self.__init__
        sig = inspect.signature(init)
        init_params = [p for p in sig.parameters]
        user_params = {p: locals[p] for p in locals if p in init_params}
        user_params = {k: v for (k, v) in user_params.items() if not 
                       isinstance(v, AnnData)}
        return user_params

    def save(self,
             dir_path: str,
             overwrite: bool=False,
             save_adata: bool=False,
             adata_file_name: str="adata.h5ad",
             **anndata_write_kwargs):
        """
        Save model to disk (the Trainer optimizer state is not saved).

        Parameters
        ----------
        dir_path:
            Path of the directory where the model will be saved.
        overwrite:
            If `True`, overwrite existing data. If `False` and directory
            already exists at `dir_path`, error will be raised.
        save_adata:
            If `True`, also saves the AnnData object.
        adata_file_name:
            File name under which the AnnData object will be saved.
        adata_write_kwargs:
            Kwargs for adata write function.
        """
        if not os.path.exists(dir_path) or overwrite:
            os.makedirs(dir_path, exist_ok=overwrite)
        else:
            raise ValueError(f"Directory '{dir_path}' already exists."
                             "Please provide another directory for saving.")

        model_save_path = os.path.join(dir_path, "model_params.pt")
        attr_save_path = os.path.join(dir_path, "attr.pkl")
        var_names_save_path = os.path.join(dir_path, "var_names.csv")

        if save_adata:
            # Convert storage format of adjacency matrix to be writable by 
            # adata.write()
            if self.adata.obsp["spatial_connectivities"] is not None:
                self.adata.obsp["spatial_connectivities"] = sp.csr_matrix(
                    self.adata.obsp["spatial_connectivities"])
            self.adata.write(
                os.path.join(dir_path, adata_file_name), **anndata_write_kwargs)

        var_names = self.adata.var_names.astype(str).to_numpy()
        public_attributes = self._get_public_attributes()
        
        torch.save(self.model.state_dict(), model_save_path)
        np.savetxt(var_names_save_path, var_names, fmt="%s")
        with open(attr_save_path, "wb") as f:
            pickle.dump(public_attributes, f)

    @classmethod
    def load(cls,
             dir_path: str,
             adata: Optional[AnnData]=None,
             adata_file_name: str="adata.h5ad",
             use_cuda: bool=False,
             n_addon_gps: int=0,
             gp_names_key: Optional[str]=None,
             freeze_non_addon_weights: bool=False) -> torch.nn.Module:
        """
        Instantiate a model from saved output. Can be used for transfer learning
        scenarios and to learn de-novo gene programs by adding add-on gene 
        programs and freezing non add-on weights.
        
        Parameters
        ----------
        dir_path:
            Path to saved outputs.
        adata:
            AnnData organized in the same way as data used to train the model.
            If ´None´, will check for and load adata saved with the model.
        adata_file_name:
            File name of the AnnData object to be loaded.
        use_cuda:
            If `True`, load model on GPU.
        n_addon_gps:
            Number of (new) add-on gene programs to be added to the model's
            architecture.
        gp_names_key:
            Key under which the gene program names are stored in ´adata.uns´.            
        freeze_non_addon_weights:
            If `True`, freeze non-addon weights to constrain training to add-on
            gene programs.
        
        Returns
        -------
        model:
            Model with loaded state dictionaries and, if specified, frozen non 
            add-on weights.
        """
        load_adata = adata is None
        use_cuda = use_cuda and torch.cuda.is_available()
        map_location = torch.device("cpu") if use_cuda is False else None

        model_state_dict, var_names, attr_dict, new_adata = load_saved_files(
            dir_path, load_adata, adata_file_name, map_location=map_location)
        adata = new_adata if new_adata is not None else adata

        validate_var_names(adata, var_names)

        if n_addon_gps != 0:
            attr_dict["n_addon_gps_"] += n_addon_gps
            attr_dict["init_params_"]["n_addon_gps"] += n_addon_gps

            if gp_names_key is None:
                raise ValueError("Please specify 'gp_names_key' so that addon "
                                 "gps can be added to the gene program list.")

            gps = list(adata.uns[gp_names_key])

            if any("addon_GP_" in gp for gp in gps):
                addon_gp_idx = int(gps[-1][-1]) + 1
                adata.uns[gp_names_key] = np.array(
                    gps + ["addon_GP_" + str(addon_gp_idx + i) for i in 
                    range(n_addon_gps)])
            else:
                adata.uns[gp_names_key] = np.array(
                    gps + ["addon_GP_" + str(i) for i in range(n_addon_gps)])

        model = initialize_model(cls, adata, attr_dict)

        # set saved attrs for loaded model
        for attr, val in attr_dict.items():
            setattr(model, attr, val)

        if n_addon_gps != 0:
            model.model.load_and_expand_state_dict(model_state_dict)
        else:
            model.model.load_state_dict(model_state_dict)

        if use_cuda:
            model.model.cuda()
        model.model.eval()

        # Freeze pre-trained weights to only allow updates of addon gene program
        # weights
        if freeze_non_addon_weights:
            if not model.is_trained_:
                raise ValueError("The model has not been pre-trained and "
                                 "therefore weights should not be frozen.")
            for param_name, param in model.model.named_parameters():
                param.requires_grad = False
                if "addon" in param_name or "theta" in param_name:
                    param.requires_grad = True    
        return model

    def _check_if_trained(self,
                          warn: bool=True):
        """
        Check if the model is trained.

        Parameters
        -------
        warn:
             If not trained and `warn` is True, raise a warning, else raise a 
             RuntimeError.
        """
        message = ("Trying to query inferred values from an untrained model. "
                   "Please train the model first.")
        if not self.is_trained_:
            if warn:
                warnings.warn(message)
            else:
                raise RuntimeError(message)