"""
This module contains the Trainer to train an NicheCompass model.
"""

import copy
import itertools
import math
import time
import warnings
from collections import defaultdict
from typing import List, Optional

import mlflow
import numpy as np
import torch
import torch.nn as nn
from anndata import AnnData

from nichecompass.data import initialize_dataloaders, prepare_data
from .basetrainermixin import BaseTrainerMixin
from .distributed import (all_gather_numpy,
                          all_reduce_sum_scalar,
                          barrier,
                          broadcast_object,
                          cleanup_distributed,
                          get_local_rank,
                          get_rank,
                          get_world_size,
                          init_distributed,
                          is_initialized,
                          is_main_process,
                          shard_indices,
                          unwrap_model)
from .metrics import eval_metrics, plot_eval_metrics
from .utils import (_cycle_iterable,
                    plot_loss_curves,
                    print_progress,
                    EarlyStopping)


class _JointForwardModule(nn.Module):
    """
    Run the node-level (omics) and the edge-level (graph) forward pass of a
    NicheCompass model in a single call.

    ´DistributedDataParallel´ prepares its gradient reduction at the end of
    every forward pass and expects exactly one forward per backward. The
    training step calls the model twice, once per decoder, before a single
    backward over the combined loss, which would leave the first pass's
    gradients unreduced. Joining the two passes into one forward restores the
    contract without changing what is computed: the two passes are independent
    given the same parameters, so the outputs are identical to calling the
    model twice.

    The LOSS is computed here as well, and that is not incidental. The loss
    uses parameters directly rather than only the outputs of the forward pass:
    the negative binomial dispersions ´target_rna_theta´ and
    ´source_rna_theta´, and the decoder weights that the L1 and group lasso
    regularizers penalize. A parameter used outside the wrapped forward has its
    gradient produced by an autograd node the reducer did not see, so its hook
    fires a second time and the backward pass fails with "Expected to mark a
    variable ready only once".

    Only used for distributed training. On a single device the model and the
    loss are called directly, exactly as before.
    """
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self,
                node_data_batch,
                edge_data_batch,
                use_only_active_gps: bool,
                loss_kwargs: dict) -> dict:
        node_model_output = self.model(
            data_batch=node_data_batch,
            decoder="omics",
            use_only_active_gps=use_only_active_gps)
        edge_model_output = self.model(
            data_batch=edge_data_batch,
            decoder="graph",
            use_only_active_gps=use_only_active_gps)
        loss_dict = self.model.loss(edge_model_output=edge_model_output,
                                    node_model_output=node_model_output,
                                    **loss_kwargs)

        # Only ´optim_loss´ is backpropagated, so every other entry leaves
        # detached. ´find_unused_parameters´ decides which parameters the
        # reducer waits for by walking the autograd graphs of everything this
        # forward returns, and the documented contract is that every output
        # derived from a parameter must take part in the backward pass or the
        # wrapper hangs waiting for gradients that never arrive.
        # ´global_loss´ would breach that: it carries terms ´optim_loss´
        # deliberately omits while they warm up -- the edge reconstruction
        # loss for the first ´n_epochs_no_edge_recon´ epochs, the contrastive
        # loss for the first ´n_epochs_no_cat_covariates_contrastive´. A
        # surrogate of this arrangement did not in fact hang on torch 2.x, so
        # this follows the contract rather than fixing an observed failure.
        # The trainer only calls ´.item()´ on these entries, so detaching them
        # changes nothing that is observed.
        return {key: (value if key == "optim_loss"
                      else value.detach() if torch.is_tensor(value)
                      else value)
                for key, value in loss_dict.items()}


class Trainer(BaseTrainerMixin):
    """
    Trainer class. Encapsulates all logic for NicheCompass model training 
    preparation and model training.
    
    Parts of the implementation are inspired by 
    https://github.com/theislab/scarches/blob/master/scarches/trainers/trvae/trainer.py#L13
    (01.10.2022)
    
    Parameters
    ----------
    adata:
        AnnData object with counts stored in ´adata.layers[counts_key]´ or
        ´adata.X´ depending on ´counts_key´ and sparse adjacency matrix stored
        in ´adata.obsp[adj_key]´.
    adata_atac:
        Additional optional AnnData object with paired spatial ATAC data.
    model:
        An NicheCompass module model instance.
    counts_key:
        Key under which the counts are stored in ´adata.layer´. If ´None´, uses
        ´adata.X´ as counts.
    adj_key:
        Key under which the sparse adjacency matrix is stored in ´adata.obsp´.
    cat_covariates_keys:
        Keys under which the categorical covariates are stored in ´adata.obs´.
    gp_targets_mask_key:
        Key under which the gene program targets mask is stored in ´model.adata.varm´. 
        This mask will only be used if no ´gp_targets_mask´ is passed explicitly
        to the model.
    gp_sources_mask_key:
        Key under which the gene program sources mask is stored in ´model.adata.varm´. 
        This mask will only be used if no ´gp_sources_mask´ is passed explicitly
        to the model.
    edge_val_ratio:
        Fraction of the data that is used as validation set on edge-level. The
        rest of the data will be used as training set on edge-level.
    node_val_ratio:
        Fraction of the data that is used as validation set on node-level. The
        rest of the data will be used as training set on edge-level.
    edge_batch_size:
        Batch size for the edge-level dataloaders.
    node_batch_size:
        Batch size for the node-level dataloaders.
    n_sampled_neighbors:
        Number of neighbors that are sampled during model training from the spatial
        neighborhood graph.
    use_early_stopping:
        If `True`, the EarlyStopping class is used to prevent overfitting.
    reload_best_model:
        If `True`, the best state of the model with respect to the early
        stopping criterion is reloaded at the end of training.
    early_stopping_kwargs:
        Kwargs for the EarlyStopping class.
    use_cuda_if_available:
        If `True`, use cuda if available.
    seed:
        Random seed to get reproducible results.
    monitor:
        If ´True´, the progress of training will be printed after each epoch.
    verbose:
        If ´True´, print out detailed training progress of individual losses.
    """
    def __init__(self,
                 adata: AnnData,
                 model: nn.Module,
                 adata_atac: Optional[AnnData]=None,
                 counts_key: Optional[str]="counts",
                 adj_key: str="spatial_connectivities",
                 cat_covariates_keys: Optional[List[str]]=None,
                 gp_targets_mask_key: str="nichecompass_gp_targets",
                 gp_sources_mask_key: str="nichecompass_gp_sources",                 
                 edge_val_ratio: float=0.1,
                 node_val_ratio: float=0.1,
                 edge_batch_size: int=512,
                 node_batch_size: Optional[int]=None,
                 n_sampled_neighbors: int=-1,
                 use_early_stopping: bool=True,
                 reload_best_model: bool=True,
                 early_stopping_kwargs: Optional[dict]=None,
                 use_cuda_if_available: bool=True,
                 multi_gpu: bool=False,
                 seed: int=0,
                 monitor: bool=True,
                 verbose: bool=False,
                 **kwargs):
        self.adata = adata
        self.adata_atac = adata_atac
        self.model = model
        self.counts_key = counts_key
        self.adj_key = adj_key
        self.cat_covariates_keys = cat_covariates_keys
        if self.cat_covariates_keys is None:
            self.n_cat_covariates = 0
        else:
            self.n_cat_covariates = len(self.cat_covariates_keys)
        self.gp_targets_mask_key = gp_targets_mask_key
        self.gp_sources_mask_key = gp_sources_mask_key
        self.edge_train_ratio_ = 1 - edge_val_ratio
        self.edge_val_ratio_ = edge_val_ratio
        self.node_train_ratio_ = 1 - node_val_ratio
        self.node_val_ratio_ = node_val_ratio
        self.edge_batch_size_ = edge_batch_size
        self.node_batch_size_ = node_batch_size
        self.n_sampled_neighbors_ = n_sampled_neighbors
        self.use_early_stopping_ = use_early_stopping
        self.reload_best_model_ = reload_best_model
        self.early_stopping_kwargs_ = (early_stopping_kwargs if 
            early_stopping_kwargs else {})
        if not "early_stopping_metric" in self.early_stopping_kwargs_:
            if edge_val_ratio > 0 and node_val_ratio > 0:
                self.early_stopping_kwargs_["early_stopping_metric"] = (
                    "val_global_loss")
            else:
                self.early_stopping_kwargs_["early_stopping_metric"] = (
                    "train_global_loss")
        self.early_stopping = EarlyStopping(**self.early_stopping_kwargs_)
        self.seed_ = seed
        self.monitor_ = monitor
        self.verbose_ = verbose
        self.loaders_n_hops_ = kwargs.pop("loaders_n_hops", 1)
        self.grad_clip_value_ = kwargs.pop("grad_clip_value", 0.)
        self.epoch = -1
        self.training_time = 0
        self.optimizer = None
        self.best_epoch = None
        self.best_model_state_dict = None

        # Join the process group before anything else, so that the rank is
        # known when the device is chosen and when output is printed
        self.multi_gpu_ = multi_gpu
        self.distributed_ = init_distributed() if multi_gpu else False
        self.world_size_ = get_world_size()
        self.rank_ = get_rank()
        if multi_gpu and not self.distributed_:
            raise RuntimeError(
                "´multi_gpu´ was requested but this process was not launched "
                "as part of a distributed job, so there is only one process "
                "to train with. Launch the script with, for example, "
                "´torchrun --nproc_per_node=4 your_script.py´. See the "
                "multi-GPU section of the user guide.")

        if is_main_process():
            print("\n--- INITIALIZING TRAINER ---")
            if self.distributed_:
                print(f"Distributed training across {self.world_size_} "
                      "processes.")

        # Set seed and use GPU if available. Every process seeds identically
        # here, so that the model is initialized identically and so that the
        # train/validation split below is the same everywhere. The per process
        # seed is only diverged afterwards, once the split exists.
        np.random.seed(self.seed_)
        if use_cuda_if_available & torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed_)
            torch.manual_seed(self.seed_)
            # Each process owns exactly one device, identified by its rank
            # within the node
            self.device = (torch.device("cuda", get_local_rank())
                           if self.distributed_ else torch.device("cuda"))
        else:
            torch.manual_seed(self.seed_)
            self.device = torch.device("cpu")
        self.model.to(self.device)

        # Prepare data and get node-level and edge-level training and validation
        # splits
        data_dict = prepare_data(
            adata=adata,
            cat_covariates_label_encoders=self.model.cat_covariates_label_encoders_,
            adata_atac=adata_atac,
            counts_key=self.counts_key,
            adj_key=self.adj_key,
            cat_covariates_keys=self.cat_covariates_keys,
            edge_val_ratio=self.edge_val_ratio_,
            edge_test_ratio=0.,
            node_val_ratio=self.node_val_ratio_,
            node_test_ratio=0.)
        self.node_masked_data = data_dict["node_masked_data"]
        self.edge_train_data = data_dict["edge_train_data"]
        self.edge_val_data = data_dict["edge_val_data"]
        self.n_nodes_train = self.node_masked_data.train_mask.sum().item()
        self.n_nodes_val = self.node_masked_data.val_mask.sum().item()
        self.n_edges_train = self.edge_train_data.edge_label_index.size(1)
        self.n_edges_val = self.edge_val_data.edge_label_index.size(1)
        if is_main_process():
            print(f"Number of training nodes: {self.n_nodes_train}")
            print(f"Number of validation nodes: {self.n_nodes_val}")
            print(f"Number of training edges: {self.n_edges_train}")
            print(f"Number of validation edges: {self.n_edges_val}")

        # Determine node batch size automatically if not specified
        if self.node_batch_size_ is None:
            self.node_batch_size_ = int(self.edge_batch_size_ / math.floor(
                self.n_edges_train / self.n_nodes_train))
        
        # The batch sizes are the GLOBAL batch sizes, so that a distributed
        # run performs the same number of optimizer steps over the same
        # effective batches as a single device run. Each process therefore
        # takes a ´world_size´-th of every batch, which is where the speedup
        # comes from, rather than making the batch larger.
        self.global_edge_batch_size_ = self.edge_batch_size_
        self.global_node_batch_size_ = self.node_batch_size_
        if self.distributed_:
            if self.edge_batch_size_ % self.world_size_ != 0:
                warnings.warn(
                    f"The edge batch size {self.edge_batch_size_} is not "
                    f"divisible by the number of processes "
                    f"{self.world_size_}, so the effective global edge batch "
                    "size is rounded down.")
            self.edge_batch_size_ = max(
                1, self.edge_batch_size_ // self.world_size_)
            self.node_batch_size_ = max(
                1, self.node_batch_size_ // self.world_size_)
        if is_main_process():
            print(f"Edge batch size: {self.global_edge_batch_size_}"
                  + (f" ({self.edge_batch_size_} per process)"
                     if self.distributed_ else ""))
            print(f"Node batch size: {self.global_node_batch_size_}"
                  + (f" ({self.node_batch_size_} per process)"
                     if self.distributed_ else ""))

        # Initialize node-level and edge-level dataloaders
        loader_dict = initialize_dataloaders(
            node_masked_data=self.node_masked_data,
            edge_train_data=self.edge_train_data,
            edge_val_data=self.edge_val_data,
            edge_batch_size=self.edge_batch_size_,
            node_batch_size=self.node_batch_size_,
            n_direct_neighbors=self.n_sampled_neighbors_,
            n_hops=self.loaders_n_hops_,
            edges_directed=False,
            neg_edge_sampling_ratio=1.,
            node_input_shard_fn=(shard_indices if self.distributed_ else None),
            edge_input_shard_fn=(
                (lambda edge_label_index: edge_label_index[
                    :, shard_indices(torch.arange(
                        edge_label_index.size(1)))])
                if self.distributed_ else None),
            drop_last=self.distributed_)
        self.edge_train_loader = loader_dict["edge_train_loader"]
        self.edge_val_loader = loader_dict.pop("edge_val_loader", None)
        self.node_train_loader = loader_dict["node_train_loader"]
        self.node_val_loader = loader_dict.pop("node_val_loader", None)

        if self.distributed_:
            # Checked up front, because an empty loader has no symptom later.
            # ´drop_last´ is set for distributed runs so that every process
            # performs the same number of iterations, and a split shorter than
            # one batch then yields nothing: cycling over an empty node loader
            # never returns, and an empty edge loader silently produces an
            # epoch of no iterations and NaN logs. Both are worse than a
            # message here.
            for description, loader in [
                    ("edge training", self.edge_train_loader),
                    ("node training", self.node_train_loader),
                    ("edge validation", self.edge_val_loader),
                    ("node validation", self.node_val_loader)]:
                if loader is not None and len(loader) == 0:
                    raise ValueError(
                        f"The {description} loader yields no batches on each "
                        f"of the {self.world_size_} processes. Its per process "
                        "share of the split is smaller than one per process "
                        "batch, and the last incomplete batch is dropped so "
                        "that every process runs the same number of "
                        "iterations. Train on fewer processes, or lower the "
                        "batch size, or raise the validation ratio.")

        if self.distributed_:
            # Diverge the random state per process, but only now that the
            # split and the loaders exist. Everything above had to be
            # identical across processes; from here on the processes must
            # differ, because the negative edges are drawn from the global
            # torch generator at iteration time. With a shared seed every
            # process would draw the same negative edges, so the extra devices
            # would recompute the same negatives instead of covering more of
            # them.
            torch.manual_seed(self.seed_ + self.rank_)
            np.random.seed(self.seed_ + self.rank_)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed_ + self.rank_)

            # The wrapper is held only by the trainer. ´self.model´ stays the
            # bare model, so that saving, loading and every attribute lookup on
            # the model keep working and checkpoints stay compatible.
            self.ddp_model = nn.parallel.DistributedDataParallel(
                _JointForwardModule(self.model),
                device_ids=([self.device.index]
                            if self.device.type == "cuda" else None),
                output_device=(self.device.index
                               if self.device.type == "cuda" else None),
                # The gene program pruning state is reduced explicitly in the
                # model, so letting the wrapper overwrite every process's
                # buffers with the ones of the first process would replace a
                # global mean by the first process's local mean
                broadcast_buffers=False,
                # Add-on gene program parameters receive no gradient when
                # there are no add-on gene programs, and the categorical
                # covariate embeddings receive none while the contrastive loss
                # is still disabled
                find_unused_parameters=True)
        else:
            self.ddp_model = None

    def train(self,
              n_epochs: int=100,
              n_epochs_all_gps: int=25,
              n_epochs_no_edge_recon: int=0,
              n_epochs_no_cat_covariates_contrastive: int=5,
              lr: float=0.001,
              weight_decay: float=0.,
              lambda_edge_recon: Optional[float]=500000.,
              lambda_cat_covariates_contrastive: Optional[float]=0.,
              contrastive_logits_pos_ratio: Optional[float]=0.125,
              contrastive_logits_neg_ratio: Optional[float]=0.125,
              lambda_gene_expr_recon: float=100.,
              lambda_chrom_access_recon: float=10.,
              lambda_group_lasso: float=0.,
              lambda_l1_masked: float=0.,
              l1_targets_mask: Optional[torch.Tensor]=None,
              l1_sources_mask: Optional[torch.Tensor]=None,
              lambda_l1_addon: float=0.,
              mlflow_experiment_id: Optional[str]=None):
        """
        Train the NicheCompass model.

        Parameters
        ----------
        n_epochs:
            Number of epochs.
        n_epochs_all_gps:
            Number of epochs during which all gene programs are used for model
            training. After that only active gene programs are retained.
        n_epochs_no_edge_recon:
            Number of epochs without edge reconstruction loss for gene
            expression decoder pretraining.
        lr:
            Learning rate.
        weight_decay:
            Weight decay (L2 penalty).
        lambda_edge_recon:
            Lambda (weighting factor) for the edge reconstruction loss. If ´>0´,
            this will enforce gene programs to be meaningful for edge
            reconstruction and, hence, to preserve spatial colocalization
            information.
        lambda_cat_covariates_contrastive:
            Lambda (weighting factor) for the categorical covariates contrastive
            loss. If ´>0´, this will enforce observations with different
            categorical covariates categories with very similar latent
            representations to become more similar, and observations with
            different latent representations to become more different.
        contrastive_logits_pos_ratio:
            Ratio for determining the logits threshold of positive contrastive
            examples of node pairs from different categorical covariates
            categories. The top (´contrastive_logits_pos_ratio´ * 100)% logits
            of node pairs from different categorical covariates categories serve
            as positive labels for the contrastive loss.
        contrastive_logits_neg_ratio:
            Ratio for determining the logits threshold of negative contrastive
            examples of node pairs from different categorical covariates
            categories. The bottom (´contrastive_logits_neg_ratio´ * 100)%
            logits of node pairs from different categorical covariates
            categories serve as negative labels for the contrastive loss.
        lambda_gene_expr_recon:
            Lambda (weighting factor) for the gene expression reconstruction
            loss. If ´>0´, this will enforce interpretable gene programs that
            can be combined in a linear way to reconstruct gene expression.
        lambda_chrom_access_recon:
            Lambda (weighting factor) for the chromatin accessibility
            reconstruction loss. If ´>0´, this will enforce interpretable gene
            programs that can be combined in a linear way to reconstruct
            chromatin accessibility.
        lambda_group_lasso:
            Lambda (weighting factor) for the group lasso regularization loss of
            gene programs. If ´>0´, this will enforce sparsity of gene programs.
        lambda_l1_masked:
            Lambda (weighting factor) for the L1 regularization loss of genes in
            masked gene programs. If ´>0´, this will enforce sparsity of genes
            in masked gene programs.
        l1_targets_mask:
            Boolean gene program gene mask that is True for all gene program target
            genes to which the L1 regularization loss should be applied (dim:
            n_genes, n_gps).
        l1_sources_mask:
            Boolean gene program gene mask that is True for all gene program source
            genes to which the L1 regularization loss should be applied (dim:
            n_genes, n_gps).
        lambda_l1_addon:
            Lambda (weighting factor) for the L1 regularization loss of genes in
            addon gene programs. If ´>0´, this will enforce sparsity of genes in
            addon gene programs.
        mlflow_experiment_id:
            ID of the mlflow experiment that will be used for tracking.
        """
        self.n_epochs_ = n_epochs
        self.n_epochs_all_gps_ = n_epochs_all_gps
        self.n_epochs_no_edge_recon_ = n_epochs_no_edge_recon
        self.n_epochs_no_cat_covariates_contrastive_ = (
            n_epochs_no_cat_covariates_contrastive)
        self.lr_ = lr
        self.weight_decay_ = weight_decay
        self.lambda_edge_recon_ = lambda_edge_recon
        self.lambda_gene_expr_recon_ = lambda_gene_expr_recon
        self.lambda_chrom_access_recon_ = lambda_chrom_access_recon
        self.lambda_cat_covariates_contrastive_ = (
            lambda_cat_covariates_contrastive)
        self.contrastive_logits_pos_ratio_ = contrastive_logits_pos_ratio
        self.contrastive_logits_neg_ratio_ = contrastive_logits_neg_ratio
        self.lambda_group_lasso_ = lambda_group_lasso
        self.lambda_l1_masked_ = lambda_l1_masked
        self.l1_targets_mask = l1_targets_mask
        self.l1_sources_mask = l1_sources_mask
        self.lambda_l1_addon_ = lambda_l1_addon
        self.mlflow_experiment_id = mlflow_experiment_id

        if is_main_process():
            print("\n--- MODEL TRAINING ---")
        
        # Log hyperparameters. Only the main process writes to MLflow, so
        # that a distributed run produces one record rather than one per
        # process.
        if self.mlflow_experiment_id is not None and is_main_process():
            for attr, attr_value in self._get_public_attributes().items():
                mlflow.log_param(attr, attr_value)
            self.model.log_module_hyperparams_to_mlflow()

        start_time = time.time()
        self.epoch_logs = defaultdict(list)
        self.model.train()
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.Adam(params,
                                          lr=lr,
                                          weight_decay=weight_decay)

        for self.epoch in range(n_epochs):
            if self.epoch < self.n_epochs_no_edge_recon_:
                self.edge_recon_active = False
            else:
                self.edge_recon_active = True
            if self.epoch < self.n_epochs_all_gps_:
                self.use_only_active_gps = False
            else:
                self.use_only_active_gps = True
            if self.epoch < self.n_epochs_no_cat_covariates_contrastive_:
                self.cat_covariates_contrastive_active = False
            else:
                self.cat_covariates_contrastive_active = True

            self.iter_logs = defaultdict(list)
            self.iter_logs["n_train_iter"] = 0
            self.iter_logs["n_val_iter"] = 0
            
            # Jointly loop through edge- and node-level batches, repeating node-
            # level batches until edge-level batches are complete
            for edge_train_data_batch, node_train_data_batch in zip(
                    self.edge_train_loader,
                    _cycle_iterable(self.node_train_loader)): # itertools.cycle
                                                              # resulted in
                                                              # memory leak
                node_train_data_batch = node_train_data_batch.to(self.device)
                edge_train_data_batch = edge_train_data_batch.to(self.device)
                loss_kwargs = dict(
                    lambda_edge_recon=self.lambda_edge_recon_,
                    lambda_gene_expr_recon=self.lambda_gene_expr_recon_,
                    lambda_chrom_access_recon=self.lambda_chrom_access_recon_,
                    lambda_cat_covariates_contrastive=self.lambda_cat_covariates_contrastive_,
                    contrastive_logits_pos_ratio=self.contrastive_logits_pos_ratio_,
                    contrastive_logits_neg_ratio=self.contrastive_logits_neg_ratio_,
                    lambda_group_lasso=self.lambda_group_lasso_,
                    lambda_l1_masked=self.lambda_l1_masked_,
                    l1_targets_mask=self.l1_targets_mask,
                    l1_sources_mask=self.l1_sources_mask,
                    lambda_l1_addon=self.lambda_l1_addon_,
                    edge_recon_active=self.edge_recon_active,
                    cat_covariates_contrastive_active=self.cat_covariates_contrastive_active)

                if self.ddp_model is not None:
                    # Both passes and the loss go through one call, so that the
                    # gradient reduction covers every use of every parameter
                    train_loss_dict = self.ddp_model(
                        node_train_data_batch,
                        edge_train_data_batch,
                        self.use_only_active_gps,
                        loss_kwargs)
                else:
                    # Forward pass node-level batch
                    node_train_model_output = self.model(
                        data_batch=node_train_data_batch,
                        decoder="omics",
                        use_only_active_gps=self.use_only_active_gps)

                    # Forward pass edge-level batch
                    edge_train_model_output = self.model(
                        data_batch=edge_train_data_batch,
                        decoder="graph",
                        use_only_active_gps=self.use_only_active_gps)

                    # Calculate training loss
                    train_loss_dict = self.model.loss(
                        edge_model_output=edge_train_model_output,
                        node_model_output=node_train_model_output,
                        **loss_kwargs)

                train_global_loss = train_loss_dict["global_loss"]
                train_optim_loss = train_loss_dict["optim_loss"]

                if self.verbose_:
                    for key, value in train_loss_dict.items():
                        self.iter_logs[f"train_{key}"].append(value.item())
                else:
                    self.iter_logs["train_global_loss"].append(
                        train_global_loss.item())   
                    self.iter_logs["train_optim_loss"].append(
                        train_optim_loss.item())
                self.iter_logs["n_train_iter"] += 1
                # Optimize for training loss
                self.optimizer.zero_grad()
                
                train_optim_loss.backward()
                # Clip gradients
                if self.grad_clip_value_ > 0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(),
                                                    self.grad_clip_value_)
                self.optimizer.step()

            # Validate model
            if (self.edge_val_loader is not None and 
                self.node_val_loader is not None):
                    self.eval_epoch()
            elif (self.edge_val_loader is None and 
            self.node_val_loader is not None):
                warnings.warn("You have specified a node validation set but no "
                              "edge validation set. Skipping validation...")
            elif (self.edge_val_loader is not None and 
            self.node_val_loader is None):
                warnings.warn("You have specified an edge validation set but no"
                              " node validation set. Skipping validation...")
    
            # Convert iteration level logs into epoch level logs. Every
            # process ran the same number of iterations over a disjoint part of
            # the same data, so averaging the per process means across
            # processes gives the mean over the whole epoch.
            for key in sorted(self.iter_logs):
                if key.startswith("train"):
                    epoch_value = (np.array(self.iter_logs[key]).sum() /
                                   self.iter_logs["n_train_iter"])
                elif key.startswith("val"):
                    epoch_value = (np.array(self.iter_logs[key]).sum() /
                                   self.iter_logs["n_val_iter"])
                else:
                    continue
                if self.distributed_:
                    epoch_value = (all_reduce_sum_scalar(float(epoch_value),
                                                         self.device)
                                   / self.world_size_)
                self.epoch_logs[key].append(epoch_value)

            # Monitor epoch level logs
            if self.monitor_ and is_main_process():
                print_progress(self.epoch, self.epoch_logs, self.n_epochs_)

            # Check early stopping. This runs on EVERY process, not just the
            # main one, because ´is_early_stopping´ does two other things
            # besides returning a decision: it reduces the learning rate on
            # the optimizer, and it records the best model state. Running it
            # on the main process alone would leave the other processes with
            # the ORIGINAL learning rate as soon as the scheduler fired, and
            # from that step on they would apply different updates to the same
            # averaged gradients and silently train a different model.
            #
            # It is safe to run everywhere because it reads only
            # ´epoch_logs´, whose entries are all-reduced above and are
            # therefore the same number on every process. The decision is
            # still broadcast, so that agreement is guaranteed rather than
            # inferred: a process that kept training while the others stopped
            # would wait forever on the next gradient reduction.
            if self.use_early_stopping_:
                stop_training = self.is_early_stopping()
                if self.distributed_:
                    stop_training = broadcast_object(stop_training)
                if stop_training:
                    break

        # Track training time and load best model
        self.training_time += (time.time() - start_time)
        minutes, seconds = divmod(self.training_time, 60)
        if is_main_process():
            print(f"Model training finished after {int(minutes)} min "
                  f"{int(seconds)} sec.")
        # Every process recorded the best model state itself, at the epoch
        # every process agreed was the best, so there is nothing to broadcast.
        # It used to be broadcast from the main process, which crashed: the
        # state dictionary holds tensors on the main process's device, pickle
        # records that device, and under ´mode=exclusive_process´ no other
        # process may open a context there. Broadcasting it would also send a
        # full copy of the weights over the interconnect once per run for no
        # gain, since ´DistributedDataParallel´ has kept the parameters
        # identical all along.
        if self.best_model_state_dict is not None and self.reload_best_model_:
            if is_main_process():
                print("Using best model state, which was in epoch "
                      f"{self.best_epoch + 1}.")
            self.model.load_state_dict(self.best_model_state_dict)

        self.model.eval()

        """
        # Track losses and eval metrics
        losses = {"train_global_loss": self.epoch_logs["train_global_loss"],
                  "train_optim_loss": self.epoch_logs["train_optim_loss"],
                  "val_global_loss": self.epoch_logs["val_global_loss"],
                  "val_optim_loss": self.epoch_logs["val_optim_loss"]}
        val_eval_metrics_over_epochs = {
            "auroc": self.epoch_logs["val_auroc_score"],
            "auprc": self.epoch_logs["val_auprc_score"],
            "best_acc": self.epoch_logs["val_best_acc_score"],
            "best_f1": self.epoch_logs["val_best_f1_score"]}

        fig = plot_loss_curves(losses)
        if self.mlflow_experiment_id is not None and is_main_process():
            mlflow.log_figure(fig, "loss_curves.png")
        fig = plot_eval_metrics(val_eval_metrics_over_epochs) 
        if self.mlflow_experiment_id is not None and is_main_process():
            mlflow.log_figure(fig, "val_eval_metrics.png")
        """

        # Calculate after training validation metrics
        if self.edge_val_loader is not None:
            self.eval_end()

    @torch.no_grad()
    def eval_epoch(self):
        """
        Epoch evaluation logic of NicheCompass model used during training.
        """
        self.model.eval()

        edge_recon_probs_val_accumulated = np.array([])
        edge_recon_labels_val_accumulated = np.array([])
        edge_same_cat_covariates_cat_val_accumulated = [
            np.array([]) for _ in range(self.n_cat_covariates)]
        edge_incl_val_accumulated = np.array([])

        # Jointly loop through edge- and node-level batches, repeating node-
        # level batches until edge-level batches are complete
        for edge_val_data_batch, node_val_data_batch in zip(
                self.edge_val_loader, _cycle_iterable(self.node_val_loader)):
            # Forward pass node level batch
            node_val_data_batch = node_val_data_batch.to(self.device)
            node_val_model_output = self.model(
                data_batch=node_val_data_batch,
                decoder="omics",
                use_only_active_gps=self.use_only_active_gps)

            # Forward pass edge level batch
            edge_val_data_batch = edge_val_data_batch.to(self.device)
            edge_val_model_output = self.model(
                data_batch=edge_val_data_batch,
                decoder="graph",
                use_only_active_gps=self.use_only_active_gps)

            # Calculate validation loss
            val_loss_dict = self.model.loss(
                    edge_model_output=edge_val_model_output,
                    node_model_output=node_val_model_output,
                    lambda_edge_recon=self.lambda_edge_recon_,
                    lambda_gene_expr_recon=self.lambda_gene_expr_recon_,
                    lambda_chrom_access_recon=self.lambda_chrom_access_recon_,
                    lambda_cat_covariates_contrastive=self.lambda_cat_covariates_contrastive_,
                    contrastive_logits_pos_ratio=self.contrastive_logits_pos_ratio_,
                    contrastive_logits_neg_ratio=self.contrastive_logits_neg_ratio_,
                    lambda_group_lasso=self.lambda_group_lasso_,
                    lambda_l1_masked=self.lambda_l1_masked_,
                    l1_targets_mask=self.l1_targets_mask,
                    l1_sources_mask=self.l1_sources_mask,
                    lambda_l1_addon=self.lambda_l1_addon_,
                    edge_recon_active=True)

            val_global_loss = val_loss_dict["global_loss"]
            val_optim_loss = val_loss_dict["optim_loss"]
            if self.verbose_:
                for key, value in val_loss_dict.items():
                    self.iter_logs[f"val_{key}"].append(value.item())
            else:
                self.iter_logs["val_global_loss"].append(val_global_loss.item())
                self.iter_logs["val_optim_loss"].append(val_optim_loss.item())  
            self.iter_logs["n_val_iter"] += 1
            
            # Calculate evaluation metrics
            edge_recon_probs_val = torch.sigmoid(
                edge_val_model_output["edge_recon_logits"])
            edge_recon_labels_val = edge_val_model_output["edge_recon_labels"]
            edge_same_cat_covariates_cat_val = edge_val_model_output["edge_same_cat_covariates_cat"]
            edge_incl_val = edge_val_model_output["edge_incl"]
            edge_recon_probs_val_accumulated = np.append(
                edge_recon_probs_val_accumulated,
                edge_recon_probs_val.detach().cpu().numpy())
            edge_recon_labels_val_accumulated = np.append(
                edge_recon_labels_val_accumulated,
                edge_recon_labels_val.detach().cpu().numpy())
            if edge_same_cat_covariates_cat_val is not None:
                for i, edge_same_cat_covariate_cat_val in enumerate(edge_same_cat_covariates_cat_val):
                    edge_same_cat_covariates_cat_val_accumulated[i] = np.append(
                        edge_same_cat_covariates_cat_val_accumulated[i],
                        edge_same_cat_covariate_cat_val.detach().cpu().numpy())
            if edge_incl_val is not None:
                edge_incl_val_accumulated = np.append(
                    edge_incl_val_accumulated,
                    edge_incl_val.detach().cpu().numpy())
            else:
                edge_same_cat_covariates_cat_val_accumulated = None
                edge_incl_val_accumulated = None

        # Every process only validated its own shard, so the predictions and
        # labels are concatenated across processes before the metrics are
        # computed, exactly as ´eval_end´ does. Averaging per shard metrics
        # afterwards would not do: an AUROC over a quarter of the validation
        # edges is a different quantity from the AUROC over all of them, so the
        # epoch would not be comparable to a single device run. It also has to
        # be the same number on every process, because these entries go
        # straight into ´epoch_logs´ without passing through the all-reduce
        # that the iteration level logs get, and ´early_stopping_metric´ may
        # name one of them.
        if self.distributed_:
            edge_recon_probs_val_accumulated = all_gather_numpy(
                edge_recon_probs_val_accumulated, self.device)
            edge_recon_labels_val_accumulated = all_gather_numpy(
                edge_recon_labels_val_accumulated, self.device)
            if edge_incl_val_accumulated is not None:
                edge_incl_val_accumulated = all_gather_numpy(
                    edge_incl_val_accumulated, self.device)
            if edge_same_cat_covariates_cat_val_accumulated is not None:
                edge_same_cat_covariates_cat_val_accumulated = [
                    all_gather_numpy(accumulated, self.device) for accumulated
                    in edge_same_cat_covariates_cat_val_accumulated]

        val_eval_dict = eval_metrics(
            edge_recon_probs=edge_recon_probs_val_accumulated,
            edge_labels=edge_recon_labels_val_accumulated,
            edge_same_cat_covariates_cat=edge_same_cat_covariates_cat_val_accumulated,
            edge_incl=edge_incl_val_accumulated)
        if self.verbose_:
            self.epoch_logs["val_auroc_score"].append(
                val_eval_dict["auroc_score"])
            self.epoch_logs["val_auprc_score"].append(
                val_eval_dict["auprc_score"])
            self.epoch_logs["val_best_acc_score"].append(
                val_eval_dict["best_acc_score"])
            self.epoch_logs["val_best_f1_score"].append(
                val_eval_dict["best_f1_score"])
        
        self.model.train()

    @torch.no_grad()
    def eval_end(self):
        """
        End evaluation logic of NicheCompass model used after model training.
        """
        self.model.eval()

        # Get edge-level ground truth and predictions
        edge_recon_probs_val_accumulated = np.array([])
        edge_recon_labels_val_accumulated = np.array([])
        edge_same_cat_covariates_cat_val_accumulated = [
            np.array([]) for _ in range(self.n_cat_covariates)]
        edge_incl_val_accumulated = np.array([])
        for edge_val_data_batch in self.edge_val_loader:
            edge_val_data_batch = edge_val_data_batch.to(self.device)

            edge_val_model_output = self.model(
                data_batch=edge_val_data_batch,
                decoder="graph",
                use_only_active_gps=True)
    
            # Calculate evaluation metrics
            edge_recon_probs_val = torch.sigmoid(
                edge_val_model_output["edge_recon_logits"])
            edge_recon_labels_val = edge_val_model_output["edge_recon_labels"]
            edge_same_cat_covariates_cat_val = edge_val_model_output["edge_same_cat_covariates_cat"]
            edge_incl_val = edge_val_model_output["edge_incl"]
            edge_recon_probs_val_accumulated = np.append(
                edge_recon_probs_val_accumulated,
                edge_recon_probs_val.detach().cpu().numpy())
            edge_recon_labels_val_accumulated = np.append(
                edge_recon_labels_val_accumulated,
                edge_recon_labels_val.detach().cpu().numpy())
            if edge_same_cat_covariates_cat_val is not None:
                for i, edge_same_cat_covariate_cat_val in enumerate(edge_same_cat_covariates_cat_val):
                    edge_same_cat_covariates_cat_val_accumulated[i] = np.append(
                        edge_same_cat_covariates_cat_val_accumulated[i],
                        edge_same_cat_covariate_cat_val.detach().cpu().numpy())
            if edge_incl_val is not None:
                edge_incl_val_accumulated = np.append(
                    edge_incl_val_accumulated,
                    edge_incl_val.detach().cpu().numpy())
            else:
                edge_same_cat_covariates_cat_val_accumulated = None
                edge_incl_val_accumulated = None

        # Get node-level ground truth and predictions
        omics_pred_dict_val_accumulated = {}
        for modality in self.model.modalities_:
            for entity in ["target", "source"]:
                omics_pred_dict_val_accumulated[f"{entity}_{modality}_preds"] = np.array([])
                omics_pred_dict_val_accumulated[f"{entity}_{modality}"] = np.array([])
        for node_val_data_batch in self.node_val_loader:
            node_val_data_batch = node_val_data_batch.to(self.device)

            node_val_model_output = self.model(
                data_batch=node_val_data_batch,
                decoder="omics",
                use_only_active_gps=True)

            for modality in self.model.modalities_:
                for entity in ["target", "source"]:
                    omics_pred_dict_val_accumulated[f"{entity}_{modality}_preds"] = np.append(
                        omics_pred_dict_val_accumulated[f"{entity}_{modality}_preds"],
                        node_val_model_output[f"{entity}_{modality}_nb_means"].detach().cpu().numpy())
                    omics_pred_dict_val_accumulated[f"{entity}_{modality}"] = np.append(
                        omics_pred_dict_val_accumulated[f"{entity}_{modality}"],
                        node_val_model_output["node_labels"][f"{entity}_{modality}"].detach().cpu().numpy())

        # Every process only saw its own shard of the validation set, so the
        # predictions and labels are concatenated across processes before the
        # metrics are computed. Otherwise each process would report a metric
        # over a ´world_size´-th of the validation data.
        if self.distributed_:
            edge_recon_probs_val_accumulated = all_gather_numpy(
                edge_recon_probs_val_accumulated, self.device)
            edge_recon_labels_val_accumulated = all_gather_numpy(
                edge_recon_labels_val_accumulated, self.device)
            if edge_incl_val_accumulated is not None:
                edge_incl_val_accumulated = all_gather_numpy(
                    edge_incl_val_accumulated, self.device)
            if edge_same_cat_covariates_cat_val_accumulated is not None:
                edge_same_cat_covariates_cat_val_accumulated = [
                    all_gather_numpy(accumulated, self.device) for accumulated
                    in edge_same_cat_covariates_cat_val_accumulated]
            omics_pred_dict_val_accumulated = {
                key: all_gather_numpy(value, self.device) for key, value
                in omics_pred_dict_val_accumulated.items()}

        val_eval_dict = eval_metrics(
            edge_recon_probs=edge_recon_probs_val_accumulated,
            edge_labels=edge_recon_labels_val_accumulated,
            edge_same_cat_covariates_cat=edge_same_cat_covariates_cat_val_accumulated,
            edge_incl=edge_incl_val_accumulated,
            omics_pred_dict=omics_pred_dict_val_accumulated)
        if not is_main_process():
            return
        print("\n--- MODEL EVALUATION ---")
        print(f"val AUROC score: {val_eval_dict['auroc_score']:.4f}")
        print(f"val AUPRC score: {val_eval_dict['auprc_score']:.4f}")
        print(f"val best accuracy score: {val_eval_dict['best_acc_score']:.4f}")
        print(f"val best F1 score: {val_eval_dict['best_f1_score']:.4f}")
        for modality in self.model.modalities_:
            for entity in ["target", "source"]:
                print(f"val {entity} {modality} MSE score: "
                      f"{val_eval_dict[f'{entity}_{modality}_mse_score']:.4f}")
        for i in range(self.n_cat_covariates):
            if f"cat_covariate{i}_mean_sim_diff" in val_eval_dict.keys():
                print(f"Val cat covariate{i} mean sim diff: "
                      f"{val_eval_dict[f'cat_covariate{i}_mean_sim_diff']:.4f}")
            
        # Log evaluation metrics
        if self.mlflow_experiment_id is not None:
            for key, value in val_eval_dict.items():
                mlflow.log_metric(f"val_{key}", value)

    def is_early_stopping(self) -> bool:
        """
        Check whether to apply early stopping, update learning rate and save 
        best model state.

        Runs on every process when training is distributed, and must: two of
        the three things it does are per process side effects, namely reducing
        the learning rate on that process's optimizer and recording that
        process's best model state. It reads only ´epoch_logs´, which is
        all-reduced, so every process reaches the same decision from the same
        numbers.

        Returns
        ----------
        stop_training:
            If `True`, stop NicheCompass model training.
        """
        early_stopping_metric = self.early_stopping.early_stopping_metric
        current_metric = self.epoch_logs[early_stopping_metric][-1]
        if self.early_stopping.update_state(current_metric):
            self.best_model_state_dict = copy.deepcopy(self.model.state_dict())
            self.best_epoch = self.epoch

        continue_training, reduce_lr = self.early_stopping.step(current_metric)
        if reduce_lr:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] *= self.early_stopping.lr_factor
            if is_main_process():
                print(f"New learning rate is {param_group['lr']}.\n")
        stop_training = not continue_training
        return stop_training