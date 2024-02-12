# Developer

## Benchmarking

```{eval-rst}
.. module:: nichecompass.benchmarking
.. currentmodule:: nichecompass

.. autosummary::
    :toctree: generated

    benchmarking.utils.compute_knn_graph_connectivities_and_distances
```

## Data

```{eval-rst}
.. module:: nichecompass.data
.. currentmodule:: nichecompass

.. autosummary::
    :toctree: generated

    data.initialize_dataloaders
    data.edge_level_split
    data.node_level_split_mask
    data.prepare_data
    data.SpatialAnnTorchDataset
```

## Models

```{eval-rst}
.. module:: nichecompass.models
.. currentmodule:: nichecompass

.. autosummary::
    :toctree: generated

    models.utils.load_saved_files
    models.utils.validate_var_names
    models.utils.initialize_model
```

## Modules

```{eval-rst}
.. module:: nichecompass.modules
.. currentmodule:: nichecompass

.. autosummary::
    :toctree: generated

    modules.VGPGAE
    modules.VGAEModuleMixin
    modules.BaseModuleMixin
    modules.compute_cat_covariates_contrastive_loss
    modules.compute_edge_recon_loss
    modules.compute_gp_group_lasso_reg_loss
    modules.compute_gp_l1_reg_loss
    modules.compute_kl_reg_loss
    modules.compute_omics_recon_nb_loss
```

## NN

```{eval-rst}
.. module:: nichecompass.nn
.. currentmodule:: nichecompass

.. autosummary::
    :toctree: generated

    nn.OneHopAttentionNodeLabelAggregator
    nn.OneHopGCNNormNodeLabelAggregator
    nn.OneHopSumNodeLabelAggregator
    nn.CosineSimGraphDecoder
    nn.FCOmicsFeatureDecoder
    nn.MaskedOmicsFeatureDecoder
    nn.Encoder
    nn.MaskedLinear
    nn.AddOnMaskedLayer
```

## Train

```{eval-rst}
.. module:: nichecompass.train
.. currentmodule:: nichecompass

.. autosummary::
    :toctree: generated

    train.Trainer
    train.eval_metrics
    train.plot_eval_metrics
```
