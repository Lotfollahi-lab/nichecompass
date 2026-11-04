# Release notes

All notable changes to this project will be documented in this file. The format
is based on [keep a changelog], and this project adheres to
[Semantic Versioning]. Full commit history is available in the [commit logs].

### 1.0.0 (27.10.2026)

NicheCompass 1.0.0 is installed from source; PyPI still provides 0.3.3 (see the
[installation guide](../installation.md)). Every tutorial requires 1.0.0.

#### Changes that affect results

Read these before comparing a model trained with 1.0.0 with one trained with 0.3.3
or earlier.

-   Early stopping is aware of gene program pruning (`prune_aware_early_stopping=True`,
    passed through `NicheCompass.train` to the trainer). Pruning makes the loss jump,
    so the best value seen before it was rarely beaten again: an unpruned state stayed
    the best one, was reloaded at the end with the decoder masks of the last epoch,
    and training stopped `patience` epochs after pruning began. The epochs before
    pruning are now a fixed warm-up that cannot stop training or lower the learning
    rate, early stopping starts with the first pruned epoch, and the best state brings
    back the decoder masks of its own epoch. `prune_aware_early_stopping=False`
    restores the behaviour of 0.3.3 exactly.
-   Pruned gene programs no longer enter the Kullback-Leibler term of the loss. Pruning
    sets their latent features to zero before the decoders, so they shaped no
    reconstruction, but 0.3.3 still added their divergence to the training loss and to
    the loss early stopping reads. The term now covers exactly the programs the decoders
    see; until pruning starts every program counts, as before. The term is weighted 1,
    against 300 and 500,000 for the reconstruction terms, so the effect is small: on a
    synthetic tissue it lowered the Kullback-Leibler term by 3 to 23% and the loss early
    stopping reads by about 0.04%, and the same number of programs was pruned. The
    latent values of pruned programs, which no analysis reads, are no longer pulled
    towards the prior. No option restores the old term.
-   Categorical covariate embeddings now enter the encoder as well as the decoders by
    default (`cat_covariates_embeds_injection=["encoder", "gene_expr_decoder",
    "chrom_access_decoder"]`). Only a reference trained this way lets query mapping
    move the query's gene program activities by adapting the covariate embedding.
    The choice changes the encoder's input size, so it has to be made when the
    reference is trained; pass `["gene_expr_decoder", "chrom_access_decoder"]`, the
    default of 0.3.3, to reproduce a model trained with it.
-   Gene program filtering and combining are fixed, so a mask built with 1.0.0 can
    differ from one built with 0.3.3 or earlier:
    `filter_and_combine_gp_dict_gps` removed both programs of every pair with
    identical source and target genes instead of keeping one, and its combine branch
    split gene names into single characters, and a chained comparison in it let
    duplicate overlap groups through; `filter_and_combine_gp_dict_gps_v2` skipped
    merging programs whose identical source gene lists had an even number of genes.
-   Query mapping keeps a frozen reference frozen. Before, the activity statistic
    behind gene program pruning kept updating on query data, so reference programs
    absent from the query were pruned and deleted from the query and joint models,
    including in validation passes; batch normalization layers switched to query
    statistics; and the frozen ATAC loadings of programs switched off by the dynamic
    mask were overwritten in place. Frozen programs are now reported inactive instead
    of pruned, batch normalization keeps the reference statistics, frozen loadings
    stay untouched, and add-on programs added for the query are trained and pruned as
    usual.
-   Analysis functions return gene program activities and loadings in one shared
    orientation, the `paper_2025` convention: a prior program's sign is set so that
    its source RNA loadings sum to zero or more (its target loadings if it has no
    measured source genes), an add-on program's so that all its RNA loadings do. New
    models use it automatically; a model saved by 0.3.3 or earlier opts in with
    `prepare_gp_analysis()`. Raw inference (`get_latent_representation()`) is
    unchanged. Do not flip score columns by hand any more.
-   Communication strengths from `compute_communication_gp_network()` now measure how
    strongly a program acts per contact between two groups of cells, not in total,
    which changes every communication network:
    -   Only the receiving cell's activity enters. The sender is scored by its
        expression of the program's source genes, and the receiver by its activity
        times its expression of the target genes, both read in the program's
        canonical direction; the activity and the target-gene term are each set to
        zero where negative, instead of their product. A cell's activity marks the receiving end of a program,
        so also multiplying in the sender's activity credited only senders that were
        themselves receivers, which favoured pairs inside one niche.
    -   Each ordered pair of groups gets the mean communication per edge of the
        spatial graph (`aggregate="mean"`, the default) instead of the sum. The sum
        grows with the size of a group and the length of a shared border, so large
        niches and their self-loops dominated the networks. `aggregate="sum"` still
        gives the total. Both are returned for every pair, with the number of edges
        and an enrichment over the program's mean edge, and pairs connected by fewer
        than `min_edges` edges (default 20) are omitted.
    -   The spatial graph is the one the model was trained on (`n_neighbors=None`,
        the default), so communication is attributed over the neighbourhood in which
        the activities were learned. An integer `n_neighbors` still builds a
        nearest-neighbour graph from the coordinates; the default used to be 90.
    -   `strength` is divided by the largest value (`normalize="global"`, the
        default) instead of min-max rescaled, so each program's weakest pair is kept
        instead of being set to zero and dropped. `strength_unscaled` keeps the
        unscaled values.

#### New features

-   Human PPI gene programs. `extract_gp_dict_from_humanppi_interactions` builds gene
    programs from the predicted human interactome of Zhang, Humphreys et al., Science
    (2025). It locates every protein from UniProt keywords, membrane topology and
    further annotation fallbacks, classifies each interaction as paracrine,
    juxtacrine, same-cell complex, extracellular assembly or intracellular (using the
    EBI Complex Portal, curated gene families and a test of whether a partner reaches
    the neighbouring cell), orients the sending partner into the source component
    from evidence, maps human genes to mouse orthologs, and records the source and
    version of every downloaded resource in a provenance file next to its cache.
    See the [human PPI guide](../user_guide/humanppi_gene_programs.md) and the
    [human PPI tutorial](../tutorials/notebooks/mouse_cns_single_sample_humanppi.ipynb).
-   Gene program analysis. `prepare_gp_analysis()`, `get_gp_quality_table()` (how
    firmly each program's sign is set), `get_gp_feature_table()` (signed loadings and
    importance of every member), `get_gp_activities()` (posterior means and standard
    deviations) are new, and `add_active_gp_scores_to_obs()` now writes canonical
    activities. `run_differential_gp_tests()`
    gains `direction`, `return_all`, `n_sample` and `seed`, reports `p_higher`,
    `p_lower`, `mean_difference` and group sizes, stores the complete table and the
    settings next to the filtered one, handles zero-variance posteriors, missing
    labels and invalid comparisons, and plots refuse results that no longer match the
    model. See the [GP analysis guide](../user_guide/gene_program_analysis.md).
-   Cell-cell communication. `compute_communication_gp_network()` gains `aggregate`,
    `min_edges`, `normalize` and `sample_key`, which removes edges between samples;
    it returns `mean_strength`, `total_strength`, `n_edges` and `enrichment` for every
    pair of groups, checks that the selected programs are eligible, and computes
    products only on graph edges, which makes it faster.
    `visualize_communication_gp_network()` draws one arrow per program between two
    groups (it drew only one of several programs that connect the same pair), gains
    `show_self_loops` and `node_labels`, which numbers long group names such as cell
    types and names them in a legend instead of crowding the circle, pairs colours
    and labels correctly and supports more than ten programs.
-   Multi-GPU training (opt-in, `multi_gpu=True`): one process per GPU with
    `DistributedDataParallel`, launched with `torchrun` or an MPI launcher, with
    `batch_size_scaling="per_process"` (default) or `"global"` and optional runtime
    profiling. See the [multi-GPU guide](../user_guide/multi_gpu_training.md).
-   Reference mapping. `load()` gains `unfreeze_encoder_weights`, which adapts the
    encoder while the loadings, and with them the meaning and orientation of every
    program, stay fixed, as well as `unfreeze_dispersion`,
    `unfreeze_node_label_aggregator` and `unfreeze_cat_covariates_projection` (the
    layers projecting the covariate embeddings into the decoders); parameters are assigned to these groups
    explicitly instead of by name fragments, and the flags of 0.3.3 keep their effect.
    `load()` prints what it unfroze and warns when no unfrozen parameter can reach the
    latent, in which case fine-tuning cannot change the query's gene program
    activities; training with nothing unfrozen stops with a message that names the
    arguments to pass. Optional graph adapters (`n_graph_adapter_hidden`,
    `unfreeze_graph_adapters`) let an already trained reference adapt to a query with
    a different neighbourhood composition while all loadings stay frozen. See the
    [reference mapping guide](../user_guide/reference_mapping.md).

#### Fixes

-   Static and add-on decoder masks are registered buffers, so moving a model to a
    GPU moves them too; `get_gp_summary()` no longer fails on a GPU model.
-   Checkpoints keep the fitted dynamic decoder masks, and add-on programs are
    expanded correctly when a reference is loaded for mapping.
-   The API documentation names `extract_gp_dict_from_mebocost_ms_interactions`
    correctly.

#### Documentation

-   New user guides for human PPI gene programs, gene program analysis, multi-GPU
    training and reference mapping.
-   All tutorials are updated for these changes, explain every analysis step, run
    from top to bottom and ship without outputs; a new tutorial builds the gene
    program mask from human PPI predictions alone.
[@sebastianbirk]

### 0.3.3 (15.05.2026)

-   Fix small bug by changing ad.read() to ad.read_h5ad().
[@sebastianbirk]


### 0.3.2 (04.11.2025)

-   Remove fixed versioning of package dependencies for uv installation with jax CUDA 12.4 support.
[@sebastianbirk]

### 0.3.1 (04.11.2025)

-   Fixed issue in retrieval of omnipath gene programs due to omnipath db version update.
[@sebastianbirk]

### 0.3.0 (07.08.2025)

-   Fixed dependency issues in optional dependencies and updated decoupler to version 2.0.0.
[@sebastianbirk]

### 0.2.4 (02.08.2025)

-   Fixed issue in pyproject.toml that prevented "pip install nichecompass[all]" from installing optional dependencies.
[@sebastianbirk]

### 0.2.3 (17.02.2025)

-   Added numpy<2 dependency as version upgrade of NumPy to major version 2 breaks required scanpy version.
[@sebastianbirk]

### 0.2.2 (09.01.2025)

-   Synced repository with Zenodo to mint DOI for publication.
[@sebastianbirk]

### 0.2.1 (15.10.2024)

-   Added a user guide to the package documentation.
[@sebastianbirk]

### 0.2.0 (22.08.2024)

-   Fixed a bug in the configuration of random seeds.
-   Fixed a bug in the definition of MEBOCOST prior gene programs.
-   Raised the default quality filter threshold for the retrieval of OmniPath gene programs.
-   Modified the GP filtering logic to combine GPs with the same source genes and drop GPs that do not have source genes if they have a specified overlap in target genes with other GPs.
-   Changed the default hyperparameters for model training based on ablation experiments.
[@sebastianbirk]

### 0.1.2 (13.02.2024)

-   The version was incremented due to package upload requirements.
[@sebastianbirk]

### 0.1.1 (13.02.2024)

-   The version was incremented due to package upload requirements.
[@sebastianbirk]

### 0.1.0 (13.02.2024)

-   First NicheCompass version.
[@sebastianbirk]

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
[commit logs]: https://github.com/Lotfollahi-lab/nichecompass/commits
[@sebastianbirk]: https://github.com/sebastianbirk
