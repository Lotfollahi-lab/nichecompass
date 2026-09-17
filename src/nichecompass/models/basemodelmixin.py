"""
This module contains generic base model functionalities, added as a Mixin to the 
NicheCompass model.
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

from nichecompass.train.distributed import (get_local_rank,
                                            is_distributed_launch,
                                            unwrap_model)
from .utils import initialize_model, load_saved_files, validate_var_names



# Parameter group names accepted by the unfreeze arguments of ´load´. A group
# is defined by a predicate over the parameter's qualified name rather than by
# a substring test, so that a name never lands in a group the caller did not
# ask for.
PARAMETER_GROUPS = ("encoder",
                    "addon_gp_encoder",
                    "addon_gp_decoder",
                    "prior_gp_decoder",
                    "graph_decoder",
                    "dispersion",
                    "node_label_aggregator",
                    "cat_covariates_embedder",
                    "cat_covariates_projection",
                    "graph_adapter")


def _parameter_group_of(param_name: str) -> str:
    """
    Classify one parameter of a ´VGPGAE´ into exactly one group.

    Order matters: the add-on gene program tensors live inside the encoder
    (´encoder.addon_conv_mu´) and inside the omics decoders
    (´*.addon_l´), so they are claimed first and the broader groups only see
    what is left.
    """
    # The add-on heads in the encoder are separated from the add-on loadings
    # in the decoders: unfreezing the encoder must carry its own add-on heads
    # with it, or they would keep frozen weights while consuming a hidden
    # representation that has moved.
    # Claimed before the encoder group: the adapter lives inside the encoder
    # but is unfrozen independently of it, since its whole purpose is to be
    # the only trainable thing in an otherwise frozen encoder.
    if "graph_adapter" in param_name:
        return "graph_adapter"
    if "addon_conv" in param_name:
        return "addon_gp_encoder"
    if ".addon_l." in f".{param_name}.":
        return "addon_gp_decoder"
    if param_name.endswith("_theta"):
        return "dispersion"
    if "node_label_aggregator" in param_name:
        return "node_label_aggregator"
    # The per category embedding table and the projection into the decoder are
    # separate groups. The table is indexed per category, so a query row is
    # the query's own; the projection is shared with the reference, so
    # training it moves the reference's offset too. They must be opt-in
    # separately, and the historical flag covers only the table.
    if "_embedder" in param_name:
        return "cat_covariates_embedder"
    if "cat_covariates_embed_l" in param_name:
        return "cat_covariates_projection"
    if param_name.startswith("encoder."):
        return "encoder"
    if "graph_decoder" in param_name:
        return "graph_decoder"
    return "prior_gp_decoder"


def _parameter_groups(module: torch.nn.Module) -> dict:
    """
    Map each group name to the qualified names of the parameters it holds.

    Returns
    -------
    groups:
        Dictionary with one entry per name in ´PARAMETER_GROUPS´, possibly
        empty (a model without categorical covariates has no embedder).
    """
    groups = {name: [] for name in PARAMETER_GROUPS}
    for param_name, _ in module.named_parameters():
        groups[_parameter_group_of(param_name)].append(param_name)
    return groups


def _freeze_running_statistics(module: torch.nn.Module,
                               model_is_frozen: bool) -> list:
    """
    Stop frozen normalisation layers from drifting on new data.

    ´requires_grad´ governs gradients, not buffers, and ´Trainer.train()´ puts
    the module back into train mode, so a batch norm whose weights are frozen
    still rewrites ´running_mean´ and ´running_var´ on every forward. The
    latent is later read in eval mode using those drifted values, which moves
    every gene program score of a model the caller asked to keep fixed.

    ´track_running_stats=False´ alone is NOT enough, and on its own is worse:
    in training mode it makes the layer normalise with the current minibatch
    instead of the stored statistics, so the model would train under query
    statistics and be read out under reference ones. It is set here only to
    stop the buffers being written; keeping the layer evaluating is done by
    ´VGPGAE.train´, which re-asserts eval mode on the layers returned here.

    Returns
    -------
    frozen:
        The normalisation layers whose statistics were pinned. The caller
        stores them on the module as ´_pinned_eval_modules´.
    """
    frozen = []
    for submodule in module.modules():
        if not isinstance(submodule, torch.nn.modules.batchnorm._BatchNorm):
            continue
        own_params = list(submodule.parameters(recurse=False))
        # With ´affine=False´ there are no parameters to inspect, so fall back
        # to whether the model as a whole is frozen.
        is_frozen = (not any(param.requires_grad for param in own_params)
                     if own_params else model_is_frozen)
        if is_frozen:
            submodule.track_running_stats = False
            submodule.eval()
            frozen.append(submodule)
    return frozen


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
             save_adata_atac: bool=False,
             adata_atac_file_name: str="adata_atac.h5ad",
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
        save_adata_atac:
            If `True`, also saves the ATAC AnnData object.
        adata_atac_file_name:
            File name under which the ATAC AnnData object will be saved.
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

        # Dynamic masks are nonpersistent torch buffers. Preserve their fitted
        # support in new checkpoints without changing legacy state_dict keys.
        self.gp_analysis_dynamic_masks_ = {
            name: buffer.detach().cpu().numpy().copy()
            for name, buffer in self.model.named_buffers()
            if "dynamic_decoder_mask" in name}

        if save_adata:
            # Convert storage format of adjacency matrix to be writable by
            # adata.write()
            if self.adj_key_ in self.adata.obsp:
                self.adata.obsp[self.adj_key_] = sp.csr_matrix(
                    self.adata.obsp[self.adj_key_])
            self.adata.write(
                os.path.join(dir_path, adata_file_name), **anndata_write_kwargs)
            
        if save_adata_atac:
            self.adata_atac.write(
                os.path.join(dir_path, adata_atac_file_name))
            
        var_names = self.adata.var_names.astype(str).to_numpy()
        public_attributes = self._get_public_attributes()
        
        # Distributed training keeps the wrapper inside the trainer and leaves
        # ´self.model´ as the bare model, so this is already the unprefixed
        # state dict. The unwrap only guards against a future change that
        # stores the wrapper here, which would otherwise write keys prefixed
        # with ´module.´ that ´load´ cannot read back.
        torch.save(unwrap_model(self.model).state_dict(), model_save_path)
        np.savetxt(var_names_save_path, var_names, fmt="%s")
        with open(attr_save_path, "wb") as f:
            pickle.dump(public_attributes, f)

    @classmethod
    def load(cls,
             dir_path: str,
             adata: Optional[AnnData]=None,
             adata_atac: Optional[AnnData]=None,
             adata_file_name: str="adata.h5ad",
             adata_atac_file_name: Optional[str]=None,
             use_cuda: bool=False,
             n_addon_gps: int=0,
             gp_names_key: Optional[str]=None,
             genes_idx_key: Optional[str]=None,
             unfreeze_all_weights: bool=False,
             unfreeze_addon_gp_weights: bool=False,
             unfreeze_cat_covariates_embedder_weights: bool=False,
             unfreeze_encoder_weights: bool=False,
             unfreeze_dispersion: bool=False,
             unfreeze_node_label_aggregator: bool=False,
             unfreeze_cat_covariates_projection: bool=False,
             unfreeze_graph_adapters: bool=False,
             n_graph_adapter_hidden: Optional[int]=None
             ) -> torch.nn.Module:
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
        adata_atac:
            ATAC AnnData organized in the same way as data used to train the
            model. If ´None´ and ´adata_atac_file_name´ is not ´None´, will
            check for and load adata_atac saved with the model.
        adata_file_name:
            File name of the AnnData object to be loaded.
        adata_atac_file_name:
            File name of the ATAC AnnData object to be loaded.
        use_cuda:
            If `True`, load model on GPU.
        n_addon_gps:
            Number of (new) add-on gene programs to be added to the model's
            architecture.
        gp_names_key:
            Key under which the gene program names are stored in ´adata.uns´.         
        genes_idx_key:
            Key under which the gene indices are stored in ´adata.uns´.
            REQUIRED when ´n_addon_gps´ > 0, since the add-on programs are
            unmasked and every gene has to be in the index.
        unfreeze_all_weights:
            If `True`, unfreeze everything and treat the run as a full refit.
            This is the only setting that clears ´freeze_´, so it is also the
            only one after which gene program orientations are recomputed
            rather than inherited from the reference. Gene program pruning is
            gated per program on whether that program's LOADINGS are
            trainable, so it also runs for the add-on programs under
            ´unfreeze_addon_gp_weights´, and for programs added by this call.
        unfreeze_addon_gp_weights:
            If `True`, unfreeze the add-on gene program weights, in both the
            encoder (´encoder.addon_conv_*´) and the omics decoders
            (´*.addon_l´). For backwards compatibility this also unfreezes
            the dispersion and the node label aggregator, which used to be
            caught by the same substring test; prefer the dedicated arguments
            below.
        unfreeze_cat_covariates_embedder_weights:
            If `True`, unfreeze the per category covariate embedding tables.
            The layers projecting them into the decoders are a separate
            group, ´unfreeze_cat_covariates_projection´, because that
            projection is shared with the reference.
        unfreeze_encoder_weights:
            If `True`, unfreeze the encoder while leaving the gene program
            loadings fixed. This is the setting for query mapping where the
            query's neighbourhood structure differs from the reference's: the
            encoder learns to place query cells on the reference gene program
            axes, and because the loadings do not move, the axes keep their
            meaning and their inherited orientation.
        unfreeze_dispersion:
            If `True`, refit the per feature negative binomial dispersion.
        unfreeze_node_label_aggregator:
            If `True`, unfreeze the node label aggregator. This has an effect
            only for ´node_label_method="one-hop-attention"´; the other
            aggregators have no parameters.
        unfreeze_cat_covariates_projection:
            If `True`, unfreeze the layers projecting the covariate embeddings
            into the decoders. Separate from the embedding tables because that
            projection is shared with the reference, so training it moves the
            reference's offset too.
        unfreeze_graph_adapters:
            If `True`, unfreeze the encoder's graph adapters. Implied by
            ´unfreeze_encoder_weights´.
        n_graph_adapter_hidden:
            Attach graph adapters of this bottleneck width to the encoder,
            even if the reference was trained without them. An adapter is a
            per cell residual bottleneck with NO message passing of its own,
            so it adds no hop and ´loaders_n_hops´ is unaffected: the frozen
            convolution that already follows it is what turns its per cell
            corrections into a neighbourhood dependent one, which is how it
            responds to the query's neighbourhood COMPOSITION without
            changing the spatial scale the reference summarized. It is the
            identity at initialization, so the model is unchanged until
            trained. Unlike injecting covariate embeddings into the encoder,
            this can be done to an already trained reference, because it adds
            parameters rather than changing the shape of existing ones. Pair
            with ´unfreeze_graph_adapters=True´; on its own it only changes
            the architecture. Must match the width the checkpoint was trained
            with, if it has adapters. ´None´ keeps whatever the checkpoint
            has.

        Returns
        -------
        model:
            Model with loaded state dictionaries and the requested parameter
            groups unfrozen. The names of the unfrozen parameters are printed
            and recorded in ´model.unfrozen_parameter_names_´.

        Notes
        -----
        What "frozen" does and does not mean. ´requires_grad=False´ stops
        gradient updates. It does not stop anything else, so ´load´
        additionally pins the running statistics of frozen normalisation
        layers, and the module is told it is frozen so that gene program
        pruning - which is destructive and irreversible - does not delete
        reference programs on the basis of query data.

        Partial unfreezing leaves ´freeze_´ True, and ´freeze_´ is what
        ´prepare_gp_analysis´ reads to decide that a gene program was
        inherited from the reference. A program whose loadings actually moved
        is detected by comparing the inherited sign against the sign its
        current loadings imply, and loses its inherited status.
        """
        load_adata = adata is None
        load_adata_atac = ((adata_atac is None) &
                           (adata_atac_file_name is not None))
        use_cuda = use_cuda and torch.cuda.is_available()
        map_location = torch.device("cpu") if use_cuda is False else None

        model_state_dict, var_names, attr_dict, new_adata, new_adata_atac = (
            load_saved_files(dir_path,
                             load_adata,
                             adata_file_name,
                             load_adata_atac,
                             adata_atac_file_name,
                             map_location=map_location))
        adata = new_adata if new_adata is not None else adata
        adata_atac = (new_adata_atac if new_adata_atac is not None else
                      adata_atac)

        validate_var_names(adata, var_names)
        # Historical checkpoints already store RNA feature order separately.
        # Preserve it for strict high-level analysis even if legacy raw loading
        # proceeds with its existing warning-only feature validation.
        attr_dict.setdefault("gp_analysis_feature_names_", {})["rna"] = list(map(str, np.atleast_1d(var_names)))

        # Include all genes in gene expression reconstruction if addon nodes
        # are present
        if n_addon_gps != 0:
            if genes_idx_key not in adata.uns:
                raise ValueError("Please specifiy a valid 'genes_idx_key' if "
                                 "'n_addon_gps' > 0, so that all genes can be "
                                 "included in the genes idx.")
            adata.uns[genes_idx_key] = np.arange(adata.n_vars * 2)
        
        # Add new categorical covariates categories from query data
        cat_covariates_cats = attr_dict["cat_covariates_cats_"]
        cat_covariates_keys = attr_dict["init_params_"]["cat_covariates_keys"]
        new_cat_covariates_cats = []
        if cat_covariates_keys is not None:
            for i, cat_covariate_key in enumerate(cat_covariates_keys):
                new_cat_covariate_cats = []
                adata_cat_covariate_cats = adata.obs[cat_covariate_key].unique().tolist()
                for cat_covariate_cat in adata_cat_covariate_cats:
                    if cat_covariate_cat not in cat_covariates_cats[i]:
                        new_cat_covariate_cats.append(cat_covariate_cat)
                for cat_covariate_cat in new_cat_covariate_cats:
                    new_cat_covariates_cats.append(cat_covariate_cat)
                    cat_covariates_cats[i].append(cat_covariate_cat)
        attr_dict["init_params_"]["cat_covariates_cats"] = cat_covariates_cats

        if n_addon_gps != 0:
            attr_dict["n_addon_gp_"] += n_addon_gps
            attr_dict["init_params_"]["n_addon_gp"] += n_addon_gps

            if gp_names_key is None:
                raise ValueError("Please specify 'gp_names_key' so that addon "
                                 "gps can be added to the gene program list.")

            # The constructor appends the additional Add-on_<index>_GP names.

        # Attaching graph adapters to a reference that was trained without
        # them. This works where encoder covariate injection cannot, because
        # an adapter ADDS parameters rather than changing the shape of any
        # existing one: ´load_and_expand_state_dict´ fills keys that the
        # checkpoint does not have from the freshly built model, and the
        # adapter is the identity at initialization, so the model is
        # unchanged until it is trained.
        if n_graph_adapter_hidden is not None:
            if n_graph_adapter_hidden <= 0:
                raise ValueError(
                    "´n_graph_adapter_hidden´ must be a positive integer, or "
                    "´None´ to keep whatever the checkpoint was trained with.")
            saved_width = attr_dict["init_params_"].get(
                "n_graph_adapter_hidden", 0) or 0
            if saved_width and saved_width != n_graph_adapter_hidden:
                # ´load_and_expand_state_dict´ can only GROW a tensor, so a
                # narrower request fails inside a ´torch.cat´ with a message
                # naming neither the adapter nor the width.
                raise ValueError(
                    f"´n_graph_adapter_hidden´ is {n_graph_adapter_hidden}, "
                    f"but this checkpoint was trained with adapters of width "
                    f"{saved_width}. Pass {saved_width}, or ´None´ to keep "
                    "the checkpoint's own width. Changing the width of "
                    "existing adapters is not supported.")
            attr_dict["init_params_"]["n_graph_adapter_hidden"] = (
                n_graph_adapter_hidden)
            attr_dict["n_graph_adapter_hidden_"] = n_graph_adapter_hidden

        # ´encoder_use_bn´ used to be accepted and ignored, so the encoder
        # always carried a batch norm when it had two fully connected layers,
        # whatever the stored value said. Checkpoints written then hold
        # ´encoder.fc_l2_bn´ weights alongside ´encoder_use_bn=False´; now
        # that the argument is honoured, believing the stored value would
        # build a model without the layer and fail on unexpected state dict
        # keys. Trust the weights.
        if any(key.startswith("encoder.fc_l2_bn")
               for key in model_state_dict):
            if not attr_dict["init_params_"].get("encoder_use_bn", True):
                attr_dict["init_params_"]["encoder_use_bn"] = True
                attr_dict["encoder_use_bn_"] = True

        model = initialize_model(cls, adata, attr_dict, adata_atac)

        # Historical checkpoints have raw analysis semantics until the user
        # explicitly calls prepare_gp_analysis(). New checkpoints persist the
        # chosen convention even when AnnData is not saved alongside them.
        model.gp_analysis_default_orientation_ = attr_dict.get(
            "gp_analysis_default_orientation_", "raw")

        # set saved attrs for loaded model
        for attr, val in attr_dict.items():
            setattr(model, attr, val)

        if (n_addon_gps != 0 or len(new_cat_covariates_cats) > 0
                or n_graph_adapter_hidden is not None):
            model.model.load_and_expand_state_dict(model_state_dict)
        else:
            model.model.load_state_dict(model_state_dict)

        for name, saved_mask in getattr(model, "gp_analysis_dynamic_masks_", {}).items():
            buffer = getattr(model.model, name)
            if saved_mask.shape[1:] != tuple(buffer.shape[1:]) or saved_mask.shape[0] > buffer.shape[0]:
                raise ValueError(f"Saved dynamic mask {name} does not match the model.")
            buffer[:len(saved_mask)].copy_(torch.as_tensor(saved_mask, device=buffer.device))

        if use_cuda:
            # Bind to the device this process owns, so that the processes of a
            # distributed job do not all load onto the first device. Outside a
            # distributed launch, pass ´None´ to keep the old behaviour of
            # loading onto whichever device is CURRENT, which a caller may have
            # chosen with ´torch.cuda.set_device´. Naming device 0
            # unconditionally would silently move such a load.
            model.model.cuda(get_local_rank() if is_distributed_launch()
                             else None)
        model.model.eval()

        # Freeze everything, then unfreeze the requested groups. The groups
        # are matched by an explicit predicate per group rather than by a
        # substring of the parameter name: substrings bundled unrelated
        # tensors together (a flag named for add-on gene programs also
        # unfroze the negative binomial dispersion and the node label
        # aggregator) and missed intended ones (´cat_covariates_embed_l´ does
        # not contain "embedder", so the query's covariate offset stayed
        # pinned to a reference fitted projection).
        for param in model.model.parameters():
            param.requires_grad = False
        model.freeze_ = True

        groups = _parameter_groups(model.model)
        parameters_by_name = dict(model.model.named_parameters())
        requested = {
            "addon_gp_decoder": unfreeze_addon_gp_weights,
            # Unfreezing the encoder carries its add-on heads with it.
            "addon_gp_encoder": (unfreeze_addon_gp_weights
                                 or unfreeze_encoder_weights),
            "dispersion": unfreeze_addon_gp_weights or unfreeze_dispersion,
            "node_label_aggregator": (unfreeze_addon_gp_weights
                                      or unfreeze_node_label_aggregator),
            "cat_covariates_embedder": unfreeze_cat_covariates_embedder_weights,
            "cat_covariates_projection": unfreeze_cat_covariates_projection,
            # Unfreezing the encoder subsumes its adapter: leaving the adapter
            # frozen while the layers around it move would waste it.
            "graph_adapter": (unfreeze_graph_adapters
                              or unfreeze_encoder_weights),
            "encoder": unfreeze_encoder_weights}
        if unfreeze_all_weights:
            requested = {name: True for name in groups}

        unfrozen = []
        for group, wanted in requested.items():
            if not wanted:
                continue
            for param_name in groups[group]:
                parameters_by_name[param_name].requires_grad = True
                unfrozen.append(param_name)
        if unfreeze_all_weights:
            # A full refit: the gene program loadings move, so nothing is
            # inherited from the reference any more.
            model.freeze_ = False

        # The module needs to know as well. Gene program pruning is
        # destructive and irreversible, and it must not delete programs whose
        # loadings are frozen and therefore cannot adapt to the query, so the
        # forward pass reads this flag before pruning.
        model.model.freeze_ = model.freeze_

        # Which programs hold their activity statistic. Per program, not per
        # module: a program whose LOADINGS are trainable still needs the
        # statistic, and a program added here starts at zero, so holding it
        # would leave it either vacuously active or permanently inactive
        # depending on ´active_gp_type´.
        #
        # Keyed on the loadings, deliberately, and NOT on the encoder heads.
        # ´unfreeze_encoder_weights´ unfreezes ´encoder.addon_conv_*´, and
        # keying on that released the hold for every add-on program while
        # their ´addon_l´ loadings stayed frozen - so the 100 add-on programs
        # a default model carries became prunable under the recipe the guide
        # recommends, which is the prior-program regression again. An encoder
        # head is not a loading.
        prior_loadings_trainable = any(
            parameters_by_name[name].requires_grad
            for name in groups["prior_gp_decoder"])
        addon_loadings_trainable = any(
            parameters_by_name[name].requires_grad
            for name in groups["addon_gp_decoder"])
        hold = torch.zeros_like(model.model.frozen_gp_statistic_mask)
        if model.freeze_:
            hold[:model.model.n_prior_gp_] = not prior_loadings_trainable
            hold[model.model.n_prior_gp_:] = not addon_loadings_trainable
            if n_addon_gps:
                # Rows created by THIS call start at zero whatever the rest of
                # the model does, so they are never held.
                hold[-n_addon_gps:] = False
        model.model.frozen_gp_statistic_mask = hold

        # ´requires_grad´ does not reach buffers, and ´Trainer.train()´ puts
        # the module back into train mode, so a frozen module holding running
        # statistics would still drift on query data. The layers are recorded
        # on the module so that ´VGPGAE.train´ can keep them evaluating on
        # every epoch.
        model.model._pinned_eval_modules = _freeze_running_statistics(
            model.model, model.freeze_)

        if model.freeze_ and not model.is_trained_:
            raise ValueError("The model has not been pre-trained and therefore "
                             "weights should not be frozen.")

        model.unfrozen_parameter_names_ = sorted(unfrozen)
        if unfrozen:
            print(f"Unfrozen parameters ({len(unfrozen)}): "
                  f"{', '.join(sorted(unfrozen))}")
            # Whether anything unfrozen can actually move the latent. A
            # covariate embedding only reaches it when the encoder was built
            # to take one, which is fixed when the REFERENCE is trained: it
            # changes the encoder's input dimension, so it cannot be switched
            # on for an existing checkpoint. Without that, fine tuning fits a
            # reconstruction offset downstream of the latent and the gene
            # program scores are exactly the reference's - a mapping run that
            # trains for hours and changes none of the quantities the user
            # goes on to interpret.
            injection = model.init_params_.get(
                "cat_covariates_embeds_injection") or []
            # Gated on groups that are actually NON-EMPTY, not merely
            # requested. Asking to unfreeze adapters on a model that has none,
            # or add-on heads on a model with no add-on programs, unfreezes
            # nothing - and testing the request alone would silence this
            # warning in exactly that case. Likewise "encoder" is in the
            # default injection for every model now, including models with no
            # categorical covariates at all, whose encoder was built with no
            # embedding input and has no embedder parameters.
            covariate_reaches_latent = bool(
                "encoder" in injection
                and requested.get("cat_covariates_embedder", False)
                and groups["cat_covariates_embedder"])
            encoder_is_trainable = any(
                requested.get(group, False) and groups[group]
                for group in ("encoder", "addon_gp_encoder", "graph_adapter"))
            if (unfreeze_graph_adapters and not groups["graph_adapter"]):
                warnings.warn(
                    "´unfreeze_graph_adapters=True´ was passed but this model "
                    "has no graph adapters, so nothing was unfrozen by it. "
                    "Pass ´n_graph_adapter_hidden=<width>´ to attach them; it "
                    "works on an already trained reference.")
            if not (encoder_is_trainable or covariate_reaches_latent):
                warnings.warn(
                    "Nothing that was unfrozen can change the latent space, "
                    "so fine tuning will not change any gene program score: "
                    "the query latent stays exactly the reference encoder "
                    "applied to the query data. This model's "
                    f"´cat_covariates_embeds_injection´ is {list(injection)} "
                    f"and it has {len(groups['cat_covariates_embedder'])} "
                    "covariate embedding tensors, so the covariate route "
                    "cannot move the latent: either 'encoder' is absent from "
                    "that list, which cannot be changed after training since "
                    "it defines the encoder's input dimension, or the model "
                    "has no categorical covariates at all. To adapt the "
                    "query latent, pass "
                    "´unfreeze_encoder_weights=True´, which leaves the gene "
                    "program loadings - and therefore the meaning of the "
                    "scores and their inherited orientation - untouched, or "
                    "attach graph adapters with "
                    "´n_graph_adapter_hidden=<width>, "
                    "unfreeze_graph_adapters=True´ for a bounded adaptation "
                    "that starts from the reference. To "
                    "use the covariate route instead, retrain the reference "
                    "with 'encoder' in ´cat_covariates_embeds_injection´.")
        else:
            # Not raised here: loading with everything frozen is the correct
            # and common way to load a model for analysis only. Training such
            # a model is what is wrong, and ´Trainer´ raises then.
            print("All parameters are frozen. This is correct for analysis, "
                  "but training this model would update nothing - pass one of "
                  "the unfreeze arguments to fine tune it.")

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
