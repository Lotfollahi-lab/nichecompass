# Gene program interpretation and differential analysis

NicheCompass now provides a shared orientation for GP activities and decoder
loadings. The analysis APIs apply it automatically for newly trained models.
Do not multiply GP columns or differential statistics by manually chosen signs.

## Start an analysis

```python
# Required once for a historical checkpoint; harmless for a new model.
quality = model.prepare_gp_analysis()

# Canonical posterior means, with columns in this exact order.
gps = model.get_active_gps()
activity, std = model.get_gp_activities(gps, return_std=True)

# Refreshes obs columns, including any previously flipped by hand.
model.add_active_gp_scores_to_obs()

# Tidy member-level loadings and the established wide summary.
features = model.get_gp_feature_table(gps)
summary = model.get_gp_summary()
quality = model.get_gp_quality_table()

# Compare each niche with the remaining cells, retaining both directions.
all_results = model.run_differential_gp_tests(
    cat_key="niche", direction="both", return_all=True)

# Or explicitly request enriched (higher canonical activity) programs.
enriched_gps = model.run_differential_gp_tests(
    cat_key="niche", direction="higher", key_added="enriched_gps")
```

`get_gp_activities()` encodes observations afresh. Pass `use_cached=True` only
when the active raw latent cache and its GP names are current. Cached values do
not include posterior standard deviations. External RNA data must use the same
feature names and order. For multimodal subsets, matching ATAC observations are
selected by name; provide `adata_atac` explicitly for new observations.
New checkpoints retain the fitted RNA and ATAC feature order. High-level analysis
rejects misordered or relabeled features, including after loading. Historical
checkpoints retain RNA order in their saved gene list; their ATAC order must be
verified against the original data when no saved feature-order metadata exists.

## What the sign convention means

A latent GP coordinate and its matching decoder column can both change sign
without changing their product. Consequently, the raw sign is not identified
by the reconstruction objective. Analysis uses `z_canonical = sign * z_raw`
and `W_canonical = sign * W_raw`. The encoder, decoder parameters, raw latent
cache, reconstructed expression and graph geometry are unchanged. Posterior
standard deviations are also unchanged.

The versioned `paper_2025` convention uses full-precision RNA loadings:

1. A prior GP with measured source members uses the sum of its source loadings.
2. A prior GP with no measured source members uses its target loadings.
3. An add-on GP uses source and target RNA loadings together.

A negative sum gives sign −1; a nonnegative sum gives +1. Having source members
whose loadings sum to zero does **not** trigger target fallback. A zero sum
never produces a zero sign or erases activities. The same sign applies to both
entities and to RNA and ATAC. ATAC feature tables apply the fitted dynamic mask.

This codifies the paper's convention, with explicit rules for zero sums. It does
not reproduce every historical hand-selected notebook flip. Recomputed plots
and simulation rankings can therefore differ from saved publication outputs.

## Interpret activities and feature loadings together

`get_gp_feature_table()` retains measured members with zero learned loadings.
Its `arm` column distinguishes positive, negative and zero loadings. Importance
is the fraction of absolute loading mass across source and target jointly,
computed separately for each modality. Zero-mass programs have zero importance.
The table and `compute_gp_gene_importances()` / `compute_gp_peak_importances()`
use full precision; rounding is a presentation choice.

A higher canonical activity does not imply that every GP member is more highly
expressed. Negative-loading members point in the opposite direction, and the
softmax decoder couples all reconstructed features. Loadings are not expression
effect sizes, causal effects, or variance explained.

The quality table reports `anchor`, `signed_mass`, `anchor_mass`,
`coherence = abs(signed_mass) / anchor_mass`, source/target disagreement,
global active status and orientation status. Zero support, coherence below 0.1,
or absolute anchor mass below 1e-8 are flagged. These are descriptive diagnostics,
not validated confidence thresholds; no program is discarded automatically.

## Differential results

The test estimates the posterior probability that a randomly sampled cell in
the focal population has higher GP activity than a randomly sampled comparison
cell. It integrates each pair of Gaussian posteriors analytically and averages
over cell pairs sampled with replacement. `n_sample` controls this Monte Carlo
average. `seed` uses a local random generator.
If both posterior variances are zero and activities are exactly equal, the
implementation splits the tie evenly between the two directions.

The complete table includes `p_higher`, `p_lower`, their log odds under the
legacy name `log_bayes_factor`, the observed posterior-mean difference,
`higher`/`lower`/`equal` direction, population sizes, GP identity and orientation
provenance. Legacy aliases `p_h0` and `p_h1` remain. Flipping orientation swaps
both probabilities and negates both signed statistics. Missing group labels
are excluded. Empty populations and invalid selections raise clear errors.

The filtered table is stored at `adata.uns[key_added]`, all tested results at
`adata.uns[key_added + "_all"]`, and analysis settings at
`adata.uns[key_added + "_params"]`. `return_all=True` returns the complete table;
the default return value remains a list of GPs passing the threshold and direction
filter. Use `comparison_cats=["rest"]` to compare against a literal category
named "rest"; the string `"rest"` means every other nonmissing category.

These probabilities describe cell distributions. They are not a donor-level
condition test, a probability about population means, or an FDR-adjusted
significance measure. Replicate-aware inference, covariate adjustment and
cross-fit stability remain separate planned extensions.

## Checkpoints, migration and plotting

Orientation metadata is saved with model attributes and mirrored in
`adata.uns["nichecompass_gp_analysis"]`. It survives saving without AnnData.
New checkpoints also retain fitted dynamic decoder masks. Old checkpoints
without this metadata retain raw high-level semantics until
`prepare_gp_analysis()` is called explicitly. Raw access remains available via
`orientation="raw"`; `get_latent_representation()` and `get_gp_data()` remain
raw by default.
Per-call `orientation="canonical"` overrides and quality-table inspection do
not change a historical model's default. Use `prepare_gp_analysis()` to opt in
persistently. Score writers track each column's convention and orientation ID
in `adata.uns["nichecompass_gp_score_columns"]["table"]`, including mixed subsets.

Metadata includes a decoder/feature fingerprint and an orientation ID. A changed
decoder or feature definition raises a stale-analysis error. After an intentional
change, call `prepare_gp_analysis(overwrite=True)` and rerun differential tests
and score exports. Training refreshes metadata automatically. Frozen reference
mappings retain the reference signs for matching GP identities. Prior GP IDs
hash the name and measured RNA membership; add-on IDs also include model identity,
so unrelated add-on programs are not treated as equivalent across fits.

`generate_enriched_gp_info_plots()` refreshes score columns and rejects a
different orientation ID in the associated test results. Rerun old differential
tests after opting a historical model into canonical analysis. Communication
networks retain their paired raw activity/loading convention, whose products
are invariant to joint sign reflection.

New differential results also fingerprint model parameters/buffers, counts,
adjacency, observation/feature order, group labels and encoder covariates. Plots
reject stale results after any of these inputs change. Input checks avoid relying
on bitwise reproducibility of GPU posterior calculations. They take time proportional
to the model and input size, without retaining additional full input copies. Rerun
legacy raw tests to obtain this provenance check.

For communication analysis on integrated datasets, pass `sample_key` so neighbors
are constructed within each independent spatial sample:

```python
from nichecompass.utils import compute_communication_gp_network

network = compute_communication_gp_network(
    gp_list=["selected_prior_GP"], model=model,
    group_key="niche", sample_key="sample", n_neighbors=90)
```

The communication graph cache checks coordinates and sample labels; singleton
samples have no edges. Score products are computed on spatial edges without a
dense all-cell outer product.

The tutorials and audited reproducibility notebooks now initialize the analysis
view after loading and request `direction="higher"` when selecting enriched
programs. Their stale saved outputs have been cleared. Rerun these analyses with
the original data to produce new figures; use Git history for the publication
workflow. Model retraining is not required for this migration.
