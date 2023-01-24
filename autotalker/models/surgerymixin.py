"""
This module contains query-reference mapping surgery model functionalities,
added as a Mixin to the Autotalker model.
"""

from typing import Union

import anndata as ad
import torch

from .utils import initialize_model, load_saved_files, validate_var_names

class SurgeryMixin:
    @classmethod
    def load_query_data(
        cls,
        adata: ad.AnnData,
        reference_model: Union[str],
        use_cuda: bool=True,
        freeze: bool = True,
        freeze_expression: bool = True,
        remove_dropout: bool = True,
        **kwargs
    ):
        """
        Transfer Learning function for new data. Uses old trained model and expands it for new conditions.

        Parts of the implementation are adapted from
        https://github.com/theislab/scarches/blob/c21492d409150cec73d26409f9277b3ac971f4a7/scarches/models/base/_base.py#L183.


        Parameters
        ----------
        adata
            Query AnnData object.
        reference_model
            A model to expand or a path to a model folder.
        freeze: Boolean
            If 'True' freezes every part of the network except the first layers of encoder/decoder.
        freeze_expression: Boolean
            If 'True' freeze every weight in first layers except the condition weights.
        kwargs:
            Kwargs for the initialization of the query model.
        
        Returns
        -------
        new_model
            New model to train on query data.
        """
        use_cuda = use_cuda and torch.cuda.is_available()
        map_location = torch.device("cpu") if use_cuda is False else None

        if isinstance(reference_model, str):
            model_state_dict, var_names, attr_dict, _ = load_saved_files(
                dir_path=reference_model,
                load_adata=False,
                map_location=map_location)
            # attr_dict, model_state_dict, var_names = cls._load_params(reference_model)
            validate_var_names(adata, var_names)
        else:
            attr_dict = reference_model._get_public_attributes()
            model_state_dict = reference_model.model.state_dict()
            validate_var_names(adata, reference_model.adata.var_names)

        print(attr_dict)
        print(attr_dict["init_params_"])

        # Add new conditions from query data
        conditions = attr_dict["conditions_"]
        condition_key = attr_dict["init_params_"]["condition_key"]
        new_conditions = []
        adata_conditions = adata.obs[condition_key].unique().tolist()
        for condition in adata_conditions:
            if condition not in conditions:
                new_conditions.append(condition)
        for condition in new_conditions:
            conditions.append(condition)
        attr_dict["init_params_"]["conditions"] = conditions

        print(attr_dict["init_params_"]["conditions"])

        attr_dict["init_params_"].update(kwargs)

        new_model = cls(adata, **attr_dict["init_params_"])
        """
        new_model._load_expand_params_from_dict(model_state_dict)

        if freeze:
            new_model.model.freeze = True
            for name, p in new_model.model.named_parameters():
                p.requires_grad = False
                if 'theta' in name:
                    p.requires_grad = True
                if freeze_expression:
                    if 'cond_L.weight' in name:
                        p.requires_grad = True
                else:
                    if "L0" in name or "N0" in name:
                        p.requires_grad = True
        """

        return new_model