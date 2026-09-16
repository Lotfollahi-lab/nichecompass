# Query-to-reference mapping

Mapping a query dataset onto a trained reference goes through
`NicheCompass.load()`, which rebuilds the saved architecture against the query
AnnData, restores the reference weights, and freezes them. You then call
`model.train()` as usual, and only the parameter groups you unfroze move.

## What the freeze is for

A gene program score is interpretable only because its gene loadings define
the axis. If the loadings move, the axis moves, and comparing "GP X activity"
between reference and query stops meaning anything. Keeping the decoder fixed
is what makes reference and query scores commensurable, and it is why
`prepare_gp_analysis` carries the reference orientation sign forward instead of
recomputing it.

This also rules out the obvious-looking alternative. Simply unfreezing the
decoder does not gently adapt the reference programs: `lambda_l1_masked` and
`lambda_group_lasso` shrink those weights toward **zero**, and there is no
anchor to the reference values, so the programs are eroded rather than
adjusted.

## Parameter groups

Each argument unfreezes one group, matched by an explicit predicate on the
parameter name rather than a substring test.

| argument | what it unfreezes |
| --- | --- |
| `unfreeze_encoder_weights` | the encoder: graph convolutions, fully connected layers, and its add-on heads (`encoder.addon_conv_*`), which have to move with the hidden representation they read |
| `unfreeze_addon_gp_weights` | add-on gene programs, in the encoder (`addon_conv_*`) and the decoders (`addon_l`) — plus, for backwards compatibility, dispersion and the aggregator |
| `unfreeze_cat_covariates_embedder_weights` | the per-category covariate embedding tables only |
| `unfreeze_cat_covariates_projection` | the layers projecting those embeddings into the decoders. Separate because that projection is shared with the reference, so training it moves the reference's offset too |
| `unfreeze_dispersion` | the per-feature negative binomial dispersion (`*_theta`) |
| `unfreeze_node_label_aggregator` | the node label aggregator — only has parameters under `node_label_method="one-hop-attention"` |
| `unfreeze_all_weights` | everything, and a full refit: see below |

The prior gene program loadings have no unfreeze argument of their own. They
move only under `unfreeze_all_weights`.

`load()` prints which parameters it unfroze and records them in
`model.unfrozen_parameter_names_`. Training a model with nothing unfrozen
raises, rather than failing later inside the optimizer.

## What "frozen" does and does not mean

`requires_grad=False` stops gradient updates. It does not stop anything else,
so `load()` additionally:

- pins the running statistics of frozen normalisation layers, because buffers
  are outside the freeze and `Trainer.train()` puts the module back into train
  mode. `track_running_stats=False` alone would not do this — in training mode
  it makes the layer normalise with the current minibatch instead of the
  stored statistics — so `VGPGAE.train` re-asserts eval mode on those layers
  every epoch;
- marks which gene programs must hold their activity statistic, so that
  **pruning** cannot delete a program whose loadings are frozen. The mark is
  per program, not per model: an add-on program added for the query has
  trainable loadings and a statistic starting at zero, so it still needs the
  running average.

Pruning matters more than it looks. The active-GP decision is driven by
`running_mean_abs_mu`, an exponential moving average that would otherwise
drift to query statistics within tens of steps; the decision then zeroes that
same statistic and the dynamic decoder masks, irreversibly. The statistic is a
persistent buffer and the masks are saved alongside the checkpoint in
`gp_analysis_dynamic_masks_`, so both survive a save and load. On a frozen model that would delete reference gene
programs on the basis of query data — from the query checkpoint and from any
joint reference+query model built from it. Pruning is therefore confined to
training a model that is actually being fitted.

## Which configuration to use

**Comparing query cells on the reference's terms**, with no adaptation:

```python
model = NicheCompass.load(dir_path=..., adata=adata_query)
```

Everything frozen. Do not call `train()`; just read the latent.

**Adapting to a query batch effect only** — the documented tutorial setting:

```python
model = NicheCompass.load(
    dir_path=..., adata=adata_query,
    unfreeze_cat_covariates_embedder_weights=True)
```

Be aware of what this can and cannot do. The covariate embedding reaches the
encoder only when `cat_covariates_embeds_injection` contains `"encoder"`. With
the default decoder-only injection the query latent is the unmodified
reference encoder applied to the query, so **no trainable parameter can change
the query gene program scores at all** — training fits a reconstruction offset
downstream of the latent. If you want the covariate embedding to absorb
technical shift out of the latent, as scArches and expiMap do, pass
`cat_covariates_embeds_injection=["encoder", "gene_expr_decoder"]` when you
train the reference.

**Adapting to a different tissue architecture:**

```python
model = NicheCompass.load(
    dir_path=..., adata=adata_query,
    unfreeze_encoder_weights=True,
    unfreeze_cat_covariates_embedder_weights=True)
```

This is the setting spatial data usually wants and the one that could not be
expressed before. A query tissue can differ in cell density and neighbourhood
composition, not just in batch chemistry, and the only components that can
absorb that are in the encoder. The loadings stay fixed, so the axes keep their
meaning and their inherited orientation. Watch for overfitting on a small
query — there is no anchor pulling the encoder back toward the reference, so
use early stopping.

**Learning new de novo programs for the query:**

```python
model = NicheCompass.load(
    dir_path=..., adata=adata_query,
    n_addon_gps=10, gp_names_key="nichecompass_gp_names",
    unfreeze_addon_gp_weights=True)
```

One caveat worth stating plainly: under a frozen decoder the add-on rows are
the only trainable sink for platform shift, batch effects and reference misfit,
and nothing in the objective penalises an add-on program for duplicating a
prior one. Check the correlation between add-on and prior activities before
reading an add-on program as new biology.

## Gene program provenance

`prepare_gp_analysis` marks a program `inherited` when the model is frozen and
the program's identity digest matches a reference row, and then reuses the
reference orientation sign. Because `freeze_` stays `True` under partial
unfreezing, that flag alone does not establish that a particular program's
loadings are unchanged — so the inherited sign is checked against the sign the
current loadings imply. On disagreement the recomputed sign wins and
`inherited` becomes `False`; both are reported, in `inherited_sign` and
`inherited_sign_disagreed`.
