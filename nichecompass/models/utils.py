"""
This module contains helper functions for the ´models´ subpackage.
"""

import logging
import os
import pickle
from collections import OrderedDict
from typing import Optional, Tuple, Literal

import anndata as ad
import numpy as np
import torch


logger = logging.getLogger(__name__)


def load_saved_files(dir_path: str,
                     load_adata: bool,
                     adata_file_name: str="adata.h5ad",
                     load_adata_atac: bool=False,
                     adata_atac_file_name: str="adata_atac.h5ad",
                     map_location: Optional[Literal["cpu", "cuda"]]=None
                     ) -> Tuple[OrderedDict, dict, np.ndarray, ad.AnnData]:
    """
    Helper to load saved model files. 
    
    Parts of the implementation are adapted from 
    https://github.com/scverse/scvi-tools/blob/master/scvi/model/base/_utils.py#L55
    (01.10.2022)

    Parameters
    ----------
    dir_path:
        Path where the saved model files are stored.
    load_adata:
        If `True`, also load the stored AnnData object.
    adata_file_name:
        File name under which the AnnData object is saved.
    load_adata_atac:
        If `True`, also load the stored ATAC AnnData object.
    adata_atac_file_name:
        File name under which the ATAC AnnData object is saved.
    map_location:
        Memory location where to map the model files to.

    Returns
    ----------
    model_state_dict:
        The stored model state dict.
    var_names:
        The stored variable names.
    attr_dict:
        The stored attributes.
    adata:
        The stored AnnData object.
    adata_atac:
        The stored ATAC AnnData object.
    """
    attr_path = os.path.join(dir_path, "attr.pkl")
    adata_path = os.path.join(dir_path, adata_file_name)
    var_names_path = os.path.join(dir_path, "var_names.csv")
    model_path = os.path.join(dir_path, "model_params.pt")

    if os.path.exists(adata_path) and load_adata:
        adata = ad.read(adata_path)
    elif not os.path.exists(adata_path) and load_adata:
        raise ValueError("Dir path contains no saved anndata and no adata was "
                         "passed.")
    else:
        adata = None

    if load_adata_atac:
        adata_atac_path = os.path.join(dir_path, adata_atac_file_name)
        if os.path.exists(adata_atac_path):
            adata_atac = ad.read(adata_atac_path)
        else:
            raise ValueError("Dir path contains no saved 'adata_atac' and no "
                             "'adata_atac' was passed.")
    else:
        adata_atac = None

    model_state_dict = torch.load(model_path, map_location=map_location)
    var_names = np.genfromtxt(var_names_path, delimiter=",", dtype=str)
    with open(attr_path, "rb") as handle:
        attr_dict = pickle.load(handle)
    return model_state_dict, var_names, attr_dict, adata, adata_atac


def validate_var_names(adata: ad.AnnData, source_var_names: str):
    """
    Helper to validate variable names. 
    
    Parts of the implementation are adapted from 
    https://github.com/scverse/scvi-tools/blob/master/scvi/model/base/_utils.py#L141
    (01.10.2022)

    Parameters
    ----------
    source_var_names:
        Variables names against which to validate.
    """
    user_var_names = adata.var_names.astype(str)
    if not np.array_equal(source_var_names, user_var_names):
        logger.warning(
            "The ´var_names´ of the passed in adata do not match the "
            "´var_names´ of the adata used to train the model. For valid "
            "results, the var_names need to be the same and in the same order "
            "as the adata used to train the model.")


def initialize_model(cls,
                     adata: ad.AnnData,
                     attr_dict: dict,
                     adata_atac: Optional[ad.AnnData]=None) -> torch.nn.Module:
    """
    Helper to initialize a model. Adapted from 
    https://github.com/scverse/scvi-tools/blob/master/scvi/model/base/_utils.py#L103.

    Parameters
    ----------
    adata:
        AnnData object to be used for initialization.
    attr_dict:
        Dictionary with attributes for model initialization.
    adata_atac:
        ATAC AnnData object to be used for initialization.
    """
    if "init_params_" not in attr_dict.keys():
        raise ValueError("No init_params_ were saved by the model.")
    # Get the parameters for the class init signature
    init_params = attr_dict.pop("init_params_")

    # Grab all the parameters except for kwargs (is a dict)
    non_kwargs = {k: v for k, v in init_params.items() if not isinstance(v, dict)}
    # Expand out kwargs
    kwargs = {k: v for k, v in init_params.items() if isinstance(v, dict)}
    kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
    if adata_atac is None:
        # Support for legacy models
        model = cls(adata=adata,
                    **non_kwargs,
                    **kwargs)
    else:
        model = cls(adata=adata,
                    adata_atac=adata_atac,
                    **non_kwargs,
                    **kwargs)        
    return model