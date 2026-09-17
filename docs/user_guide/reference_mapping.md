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
| `unfreeze_graph_adapters` | the encoder's graph adapters, if the model has any. Implied by `unfreeze_encoder_weights` |
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

This works only if the covariate embedding reaches the encoder, which means
`cat_covariates_embeds_injection` must contain `"encoder"`. That is now the
default, but it is a property of the **reference**: it adds the embedding
width to the encoder's input dimension, so it changes `fc_l1`'s shape and
cannot be switched on for an existing checkpoint. `load()` warns when nothing
you unfroze can reach the latent, which is what a decoder-only reference plus
this setting amounts to — the query latent is then the unmodified reference
encoder applied to the query, and training fits a reconstruction offset
downstream of it. For such a reference, `unfreeze_encoder_weights` is the
available lever.

### What encoder injection can express

More than a per-sample offset, and this is worth being precise about. The
embedding is concatenated to the encoder input, passed through `fc_l1` and its
**ReLU**, and only then aggregated over the spatial graph. Because of that
nonlinearity, each neighbour's covariate contribution is
`relu(W_x x_j + W_c c + b) − relu(W_x x_j + b)`, which depends on that
neighbour's own expression. Aggregation therefore produces an effect that
depends on **which** cell types surround a cell, not only on how many:

```
same degree, neighbourhood all type A:  -0.348  +0.263  +0.154  ...
same degree, neighbourhood all type B:  -0.084  +0.376  -0.706  ...
```

So a frozen encoder with a trainable covariate embedding can adapt in a
composition-sensitive way. The limit is dimensional rather than qualitative:
the only trainable tensor on that route is the embedding, so the reachable
perturbations form a family with as many parameters as the embedding is wide,
pushed through a frozen network. It can tilt the latent in response to
composition; it cannot re-map composition arbitrarily. For that, unfreeze the
encoder.

The contrast is instructive. `Encoder` also supports a `"hidden"` mode, which
concatenates the embedding **after** the ReLU and feeds it straight into the
convolution. That path is linear in the embedding, so the aggregate reduces to
a degree-dependent offset and carries no composition information at all — the
effect is identical across neighbourhoods to within floating point. Only the
`"input"` mode buys the behaviour above, and it is the mode the model always
uses (`VGPGAE` never passes the argument, so `Encoder`'s own default applies).

**Bounded adaptation of an existing reference — graph adapters:**

```python
model = NicheCompass.load(
    dir_path=..., adata=adata_query,
    n_graph_adapter_hidden=32,
    unfreeze_graph_adapters=True)
```

A graph adapter is a residual bottleneck with its own message passing,
`h + up(conv(act(down(h)), edge_index))`, inserted into the encoder before the
frozen convolutions that produce `mu`. It is never applied to `mu` itself:
`mu` *is* the gene program activities, so transforming it would move the axes
the loadings define.

Three properties, and no other option here has all three:

- **It sees the real neighbourhood.** Because the message passing is inside
  the adapter, it can respond to which cell types surround a cell, not only to
  how many.
- **It starts as the identity.** `up` is zero-initialised, so attaching an
  adapter changes nothing until it is trained, and a query run departs from the
  reference gradually. An unfrozen encoder has no such anchor.
- **It can be retrofitted.** It adds parameters rather than changing the shape
  of existing ones, so it attaches to a reference that is already trained —
  which encoder covariate injection cannot do.

Use a bottleneck much narrower than the hidden width; that width is what
bounds how far the query can depart from the reference.

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
