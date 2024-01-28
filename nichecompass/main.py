import typer
import json
import pickle
from pprint import pprint
from datetime import datetime
from wonderwords import RandomWord
import os
import anndata as ad
import pandas as pd
import scanpy as sc
import squidpy as sq
import scipy.sparse as sp
from scipy.special import erfc

from nichecompass.models import NicheCompass
from nichecompass.utils import (add_gps_from_gp_dict_to_adata,
                                extract_gp_dict_from_collectri_tf_network,
                                extract_gp_dict_from_mebocost_es_interactions,
                                extract_gp_dict_from_nichenet_lrt_interactions,
                                extract_gp_dict_from_omnipath_lr_interactions,
                                filter_and_combine_gp_dict_gps,
                                get_unique_genes_from_gp_dict)


app = typer.Typer()


@app.command()
def skeleton(output: str):
    """Creates a skeleton configuration file with sensible defaults."""

    default_config = {
        "artefact_directory": "artefacts",
        "gene_programs": {
            "sources": ["omnipath", "nichenet", "mebocost", "brain_marker"],
            "species": "mouse",
            "filter_mode": "subset",
            "gene_orthologs_mapping_file_path": "human_mouse_gene_orthologs.csv",
            "export_file_path": "gene_programs.pkl",
            "nichenet": {
                "keep_target_genes_ratio": 0.01,
                "max_target_genes_per_gene_program": 100
            },
            "mebocost": {
                "metabolite_enzyme_sensor_directory_path": "metabolite_enzyme_sensor_gps"
            },
        },
        "dataset": {
            "file_path": "spatial_dataset.h5ad",
            "library_key": "",
            "spatial_key": "",
            "export_file_path": "spatial_dataset_built.h5ad",
            "gene_programs": "gene_programs.pkl",
            "n_neighbors": 12,
            "gene_filters": {
                "n_highly_variable": 0,
                "n_spatially_variable": 0,
                "min_cell_gene_thresh_ratio": 0.1,
                "gene_program_relevant": True
            }
        },
        "model": {
            "dataset_file_path": "spatial_dataset_built.h5ad",
            "cat_covariates_embeds_injection": ["gene_expr_decoder"],
            "cat_covariates_keys": None,
            "cat_covariates_no_edges": None,
            "cat_covariates_embeds_nums": None,
            "n_addon_gp": 10,
            "active_gp_thresh_ratio": 0.05,
            "n_fc_layers_encoder": 1,
            "n_layers_encoder": 1,
            "conv_layer_encoder": "gcnconv",
            "n_hidden_encoder": None,
            "node_label_method": "one-hop-norm",
        },
        "training": {
            "n_epochs": 400,
            "n_epochs_all_gps": 25,
            "n_epochs_no_cat_covariates_contrastive": 5,
            "lr": 0.001,
            "lambda_edge_recon": 5000000,
            "lambda_gene_expr_recon": 3000,
            "lambda_cat_covariates_contrastive": 0,
            "contrastive_logits_pos_ratio": 0,
            "contrastive_logits_neg_ratio": 0,
            "lambda_group_lasso": 0,
            "lambda_l1_masked": 0,
            "lambda_l1_addon": 0,
            "edge_batch_size": 256,
            "node_batch_size": 256,
            "n_sampled_neighbors": 1,
        },
        "train_query": {
            "reference_model": {
                "artefact_directory": "",
                "file_path": "spatial_dataset_built.h5ad",
                "cat_covariates_keys": None
            },
            "dataset": {
                "file_path": "",
                "spatial_key": "spatial",
                "library_key": "label"
            },
            "graph": {
                "n_neighbors": 3
            },
            "training": {
                "n_epochs": 400,
                "n_epochs_all_gps": 25,
                "n_epochs_no_cat_covariates_contrastive": 5,
                "lr": 0.001,
                "lambda_edge_recon": 5000000,
                "lambda_gene_expr_recon": 3000,
                "lambda_cat_covariates_contrastive": 0,
                "contrastive_logits_pos_ratio": 0,
                "contrastive_logits_neg_ratio": 0,
                "lambda_group_lasso": 0,
                "lambda_l1_masked": 0,
                "lambda_l1_addon": 0,
                "edge_batch_size": 256,
                "node_batch_size": 256,
                "n_sampled_neighbors": 1,
                "artefact_directory": ""
            }
        },
        "clustering": {
            "adata_directory": "",
            "file_path": "",
            "primary_cluster_resolution": 0.2,
            "primary_cluster_min_cells": 10000,
            "sub_cluster_resolution": 0.2,
            "sub_cluster_min_cells": 2000
        },
        "gene_program_enrichment": {
            "adata_directory": "",
            "adata_file_path": "",
            "latent_directory": ""
        }
    }

    with open(output, 'w') as file:
        json.dump(default_config, file, indent=4)



@app.command()
def build_gene_programs(config: str):

    run_timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    adjective = RandomWord().word(word_min_length=3, word_max_length=8, include_categories=["adjectives"])
    noun = RandomWord().word(word_min_length=3, word_max_length=8, include_categories=["nouns"])
    run_label = adjective + "_" + noun
    print(f"Starting run {run_label} at {run_timestamp}...")

    print(f"Loading run configuration...")
    with open(config) as file:
        config = json.load(file)
    pprint(config)

    print("Building gene programs...")
    gene_programs = {}
    genes = set()

    if "omnipath" in config["gene_programs"]["sources"]:
        omnipath_gene_programs = extract_gp_dict_from_omnipath_lr_interactions(
            species=config["gene_programs"]["species"],
            gene_orthologs_mapping_file_path=config["gene_programs"]["gene_orthologs_mapping_file_path"],
            plot_gp_gene_count_distributions=False)
        gene_programs.update(omnipath_gene_programs)

    if "nichenet" in config["gene_programs"]["sources"]:
        nichenet_gene_programs = extract_gp_dict_from_nichenet_lrt_interactions(
            species=config["gene_programs"]["species"],
            version="v2",
            keep_target_genes_ratio=config["gene_programs"]["nichenet"]["keep_target_genes_ratio"],
            max_n_target_genes_per_gp=config["gene_programs"]["nichenet"]["max_target_genes_per_gene_program"],
            gene_orthologs_mapping_file_path=config["gene_programs"]["gene_orthologs_mapping_file_path"],
            plot_gp_gene_count_distributions=False)
        gene_programs.update(nichenet_gene_programs)

    if "mebocost" in config["gene_programs"]["sources"]:
        mebocost_gene_programs = extract_gp_dict_from_mebocost_es_interactions(
            dir_path=config["gene_programs"]["mebocost"]["metabolite_enzyme_sensor_directory_path"],
            species=config["gene_programs"]["species"],
            plot_gp_gene_count_distributions=False)
        gene_programs.update(mebocost_gene_programs)

    if "collectri" in config["gene_programs"]["sources"]:
        collectri_gene_programs = extract_gp_dict_from_collectri_tf_network(
            species=config["gene_programs"]["species"],
            plot_gp_gene_count_distributions=False)
        gene_programs.update(collectri_gene_programs)

    if "brain_marker" in config["gene_programs"]["sources"]:
        # Add spatial layer marker gene GPs
        # Load experimentially validated marker genes
        validated_marker_genes_df = pd.read_csv(f"marker_gps/Validated_markers_MM_layers.tsv",
                                                sep="\t",
                                                header=None,
                                                names=["gene_name", "ensembl_id", "layer"])
        validated_marker_genes_df = validated_marker_genes_df[["layer", "gene_name"]]
        # Load ranked marker genes and get top 100 per layer
        ranked_marker_genes_df = pd.DataFrame()
        for ranked_marker_genes_file_name in [
            "Ranked_mm_L2L3.tsv",
            "Ranked_mm_L4.tsv",
            "Ranked_mm_L5.tsv",
            "Ranked_mm_L6.tsv",
            "Ranked_mm_L6b.tsv"]:
            ranked_marker_genes_layer_df = pd.read_csv(
                f"marker_gps/{ranked_marker_genes_file_name}",
                sep="\t",
                header=None,
                names=["ensembl_id", "gene_name", "layer"])
            ranked_marker_genes_layer_df = ranked_marker_genes_layer_df[:100]  # filter top 100 genes
            ranked_marker_genes_layer_df = ranked_marker_genes_layer_df[["layer", "gene_name"]]
            ranked_marker_genes_df = pd.concat([ranked_marker_genes_df, ranked_marker_genes_layer_df])
        marker_genes_df = pd.concat([validated_marker_genes_df, ranked_marker_genes_df])
        marker_genes_grouped_df = marker_genes_df.groupby("layer")["gene_name"].agg(list).reset_index()
        marker_genes_grouped_df.columns = ["layer", "marker_genes"]
        marker_genes_grouped_df["layer"] = marker_genes_grouped_df["layer"] + "_marker_GP"
        marker_genes_gp_dict = {}
        for layer, marker_genes in zip(marker_genes_grouped_df["layer"], marker_genes_grouped_df["marker_genes"]):
            marker_genes_gp_dict[layer] = {
                "sources": marker_genes,
                "targets": marker_genes,
                "sources_categories": ["marker"] * len(marker_genes),
                "targets_categories": ["marker"] * len(marker_genes)}
        gene_programs.update(marker_genes_gp_dict)

    print("Merging and filtering gene programs...")
    filtered_gene_programs = filter_and_combine_gp_dict_gps(
        gp_dict=gene_programs,
        gp_filter_mode=config["gene_programs"]["filter_mode"],
        overlap_thresh_source_genes=0.9,
        overlap_thresh_target_genes=0.9,
        overlap_thresh_genes=0.9)

    print("Exporting gene programs...")
    os.makedirs(os.path.join(config["artefact_directory"], run_label), exist_ok=True)
    with open(os.path.join(config["artefact_directory"], run_label, "gene_programs.pkl"), "wb") as file:
        pickle.dump(filtered_gene_programs, file, pickle.HIGHEST_PROTOCOL)

    return run_label


@app.command()
def build_dataset(config: str):

    run_timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    adjective = RandomWord().word(word_min_length=3, word_max_length=8, include_categories=["adjectives"])
    noun = RandomWord().word(word_min_length=3, word_max_length=8, include_categories=["nouns"])
    run_label = adjective + "_" + noun
    print(f"Starting run {run_label} at {run_timestamp}...")

    print("Loading run configuration...")
    with open(config) as file:
        config = json.load(file)
    pprint(config)

    print("Loading gene programs...")
    with open(config["dataset"]["gene_programs"], "rb") as file:
        gene_programs = pickle.load(file)

    print("Reading dataset...")
    adata = ad.read_h5ad(config["dataset"]["file_path"])
    sq.gr.spatial_neighbors(adata,
                            coord_type="generic",
                            spatial_key=config["dataset"]["spatial_key"],
                            library_key=config["dataset"]["library_key"],
                            n_neighs=config["dataset"]["n_neighbors"])
    adjacency = adata.obsp["spatial_connectivities"]
    symmetrical_adjacency = adjacency.maximum(adjacency.T)
    adata.obsp["spatial_connectivities"] = symmetrical_adjacency

    print("Filtering genes...")

    gene_program_genes = get_unique_genes_from_gp_dict(
        gp_dict=gene_programs,
        retrieved_gene_entities=["sources", "targets"])

    print(f"Annotating based on expression in at least {config['dataset']['gene_filters']['min_cell_gene_thresh_ratio'] * 100}% of cells...")
    min_cells = int(adata.shape[0] * config["dataset"]["gene_filters"]["min_cell_gene_thresh_ratio"])
    adata.var["expressed"] = sc.pp.filter_genes(adata, min_cells=min_cells, inplace=False)[0].tolist()

    print(f"Annotating {config['dataset']['gene_filters']['n_highly_variable']} highly variable genes...")
    n_top_genes = (0 if config["dataset"]["gene_filters"]["n_highly_variable"] is None else config["gene_filters"]["n_highly_variable"])
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="seurat_v3",
        batch_key=config["dataset"]["library_key"])

    print(f"Annotating {config['dataset']['gene_filters']['n_spatially_variable']} spatially variable genes...")
    sq.gr.spatial_autocorr(adata, mode="moran", genes=adata.var_names)
    sv_genes = adata.uns["moranI"].index[:config["dataset"]["gene_filters"]["n_spatially_variable"]].tolist()
    adata.var["spatially_variable"] = adata.var_names.isin(sv_genes)

    print(f"Annotating genes present in gene programs...")
    adata.var["gene_program_relevant"] = adata.var.index.isin(gene_program_genes)

    print("Applying filtering...")
    adata.var["retained_gene"] = [True] * adata.shape[1]

    if config["dataset"]["gene_filters"]["min_cell_gene_thresh_ratio"] is not None:
        adata.var["retained_gene"] = adata.var["retained_gene"] & adata.var["expressed"]

    if config["dataset"]["gene_filters"]["n_highly_variable"] is not None:
        adata.var["retained_gene"] = adata.var["retained_gene"] & adata.var["highly_variable"]

    if config["dataset"]["gene_filters"]["n_spatially_variable"] is not None:
        adata.var["retained_gene"] = adata.var["retained_gene"] & adata.var["spatially_variable"]

    if config["dataset"]["gene_filters"]["gene_program_relevant"] is not None:
        adata.var["retained_gene"] = adata.var["retained_gene"] & adata.var["gene_program_relevant"]

    adata = adata[:, adata.var["retained_gene"] == True]
    print(f"Retaining {len(adata.var_names)} genes.")

    print("Adding gene programs to dataset...")
    add_gps_from_gp_dict_to_adata(gp_dict=gene_programs, adata=adata)

    print("Exporting dataset...")
    os.makedirs(os.path.join(config["artefact_directory"], run_label), exist_ok=True)
    adata_basename = os.path.basename(config["dataset"]["file_path"])
    adata.write_h5ad(os.path.join(config["artefact_directory"], run_label, adata_basename))

    return run_label


@app.command()
def train_reference(config: str):

    run_timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    adjective = RandomWord().word(word_min_length=3, word_max_length=8, include_categories=["adjectives"])
    noun = RandomWord().word(word_min_length=3, word_max_length=8, include_categories=["nouns"])
    run_label = adjective + "_" + noun
    print(f"Starting run {run_label} at {run_timestamp}...")

    print("Loading run configuration...")
    with open(config) as file:
        config = json.load(file)
    pprint(config)

    print("Reading dataset...")
    adata = ad.read_h5ad(config["model"]["dataset_file_path"])

    print("Initializing model...")
    model = NicheCompass(adata,
                         counts_key=None,
                         cat_covariates_embeds_injection=config["model"]["cat_covariates_embeds_injection"],
                         cat_covariates_keys=config["model"]["cat_covariates_keys"],
                         cat_covariates_no_edges=config["model"]["cat_covariates_no_edges"],
                         cat_covariates_embeds_nums=config["model"]["cat_covariates_embeds_nums"],
                         n_addon_gp=config["model"]["n_addon_gp"],
                         active_gp_thresh_ratio=config["model"]["active_gp_thresh_ratio"],
                         n_fc_layers_encoder=config["model"]["n_fc_layers_encoder"],
                         n_layers_encoder=config["model"]["n_layers_encoder"],
                         conv_layer_encoder=config["model"]["conv_layer_encoder"],
                         n_hidden_encoder=config["model"]["n_hidden_encoder"],
                         node_label_method=config["model"]["node_label_method"])

    print("Training model...")
    model.train(n_epochs=config["training"]["n_epochs"],
                n_epochs_all_gps=config["training"]["n_epochs_all_gps"],
                n_epochs_no_cat_covariates_contrastive=config["training"]["n_epochs_no_cat_covariates_contrastive"],
                lr=config["training"]["lr"],
                lambda_edge_recon=config["training"]["lambda_edge_recon"],
                lambda_gene_expr_recon=config["training"]["lambda_gene_expr_recon"],
                lambda_cat_covariates_contrastive=config["training"]["lambda_cat_covariates_contrastive"],
                contrastive_logits_pos_ratio=config["training"]["contrastive_logits_pos_ratio"],
                contrastive_logits_neg_ratio=config["training"]["contrastive_logits_neg_ratio"],
                lambda_group_lasso=config["training"]["lambda_group_lasso"],
                lambda_l1_masked=config["training"]["lambda_l1_masked"],
                lambda_l1_addon=config["training"]["lambda_l1_addon"],
                edge_batch_size=config["training"]["edge_batch_size"],
                node_batch_size=config["training"]["node_batch_size"],
                n_sampled_neighbors=config["training"]["n_sampled_neighbors"],
                verbose=True)

    print("Computing neighbor graph...")
    sc.pp.neighbors(model.adata, use_rep="nichecompass_latent", key_added="nichecompass_latent")

    print("Computing latent umap embedding...")
    sc.tl.umap(model.adata, neighbors_key="nichecompass_latent")

    print("Exporting trained model...")
    os.makedirs(os.path.join(config["artefact_directory"], run_label), exist_ok=True)
    model.save(
        dir_path=os.path.join(config["artefact_directory"], run_label),
        adata_file_name=os.path.splitext(os.path.basename(config["model"]["dataset_file_path"]))[0] + ".h5ad",
        overwrite = True,
        save_adata = True)
    with open(os.path.join(config["artefact_directory"], run_label, "run-config.yml"), 'w') as file:
        json.dump(config, file, indent=4)

    return run_label


@app.command()
def train_query(config: str):

    run_timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    adjective = RandomWord().word(word_min_length=3, word_max_length=8, include_categories=["adjectives"])
    noun = RandomWord().word(word_min_length=3, word_max_length=8, include_categories=["nouns"])
    run_label = adjective + "_" + noun
    print(f"Starting run {run_label} at {run_timestamp}...")

    print("Loading run configuration...")
    with open(config) as file:
        config = json.load(file)
    pprint(config)

    print("Retrieving reference model genes...")
    print(config["train_query"]["reference_model"]["artefact_directory"])

    reference_model = NicheCompass.load(
        dir_path=os.path.join(config["train_query"]["reference_model"]["artefact_directory"]),
        adata_file_name=os.path.splitext(os.path.basename(config["train_query"]["reference_model"]["file_path"]))[0] + ".h5ad",
        gp_names_key="nichecompass_gp_names")
    reference_adata = reference_model.adata
    genes = reference_model.adata.var_names

    print("Reading dataset...")
    adata = ad.read_h5ad(config["train_query"]["dataset"]["file_path"])
    sq.gr.spatial_neighbors(adata,
                            coord_type="generic",
                            spatial_key=config["train_query"]["dataset"]["spatial_key"],
                            library_key=config["train_query"]["dataset"]["library_key"],
                            n_neighs=config["train_query"]["graph"]["n_neighbors"])
    adjacency = adata.obsp["spatial_connectivities"]
    symmetrical_adjacency = adjacency.maximum(adjacency.T)
    adata.obsp["spatial_connectivities"] = symmetrical_adjacency

    print("Filtering for genes used in reference...")
    adata = adata[:, genes]

    print("Adding gene programs to dataset...")
    adata.varm["nichecompass_gp_targets"] = reference_adata.varm["nichecompass_gp_targets"]
    adata.varm["nichecompass_gp_sources"] = reference_adata.varm["nichecompass_gp_sources"]
    adata.varm["nichecompass_gp_targets_categories"] = reference_adata.varm["nichecompass_gp_targets_categories"]
    adata.varm["nichecompass_gp_sources_categories"] = reference_adata.varm["nichecompass_gp_sources_categories"]
    adata.uns["nichecompass_targets_categories_label_encoder"] = reference_adata.uns["nichecompass_targets_categories_label_encoder"]
    adata.uns["nichecompass_sources_categories_label_encoder"] = reference_adata.uns["nichecompass_sources_categories_label_encoder"]
    adata.uns["nichecompass_gp_names"] = reference_adata.uns["nichecompass_gp_names"]
    adata.uns["nichecompass_genes_idx"] = reference_adata.uns["nichecompass_genes_idx"]
    adata.uns["nichecompass_target_genes_idx"] = reference_adata.uns["nichecompass_target_genes_idx"]
    adata.uns["nichecompass_source_genes_idx"] = reference_adata.uns["nichecompass_source_genes_idx"]

    print("Training model...")
    model = NicheCompass.load(
        dir_path=os.path.join(config["train_query"]["reference_model"]["artefact_directory"]),
        adata=adata,
        adata_file_name=os.path.splitext(os.path.basename(config["train_query"]["reference_model"]["file_path"]))[0] + ".h5ad",
        gp_names_key="nichecompass_gp_names",
        unfreeze_all_weights=False,
        unfreeze_cat_covariates_embedder_weights=True)

    model.train(n_epochs=config["train_query"]["training"]["n_epochs"],
                n_epochs_all_gps=config["train_query"]["training"]["n_epochs_all_gps"],
                n_epochs_no_cat_covariates_contrastive=config["train_query"]["training"]["n_epochs_no_cat_covariates_contrastive"],
                lr=config["train_query"]["training"]["lr"],
                lambda_edge_recon=config["train_query"]["training"]["lambda_edge_recon"],
                lambda_gene_expr_recon=config["train_query"]["training"]["lambda_gene_expr_recon"],
                lambda_cat_covariates_contrastive=config["train_query"]["training"]["lambda_cat_covariates_contrastive"],
                contrastive_logits_pos_ratio=config["train_query"]["training"]["contrastive_logits_pos_ratio"],
                contrastive_logits_neg_ratio=config["train_query"]["training"]["contrastive_logits_neg_ratio"],
                lambda_group_lasso=config["train_query"]["training"]["lambda_group_lasso"],
                lambda_l1_masked=config["train_query"]["training"]["lambda_l1_masked"],
                edge_batch_size=config["train_query"]["training"]["edge_batch_size"],
                node_batch_size=config["train_query"]["training"]["node_batch_size"],
                n_sampled_neighbors=config["train_query"]["training"]["n_sampled_neighbors"],
                verbose=True)

    print("Exporting trained model...")
    model.save(
        dir_path=os.path.join(config["artefact_directory"], run_label),
        adata_file_name=os.path.splitext(os.path.basename(config["train_query"]["dataset"]["file_path"]))[0] + ".h5ad",
        overwrite=True,
        save_adata=True)
    with open(os.path.join(config["artefact_directory"], run_label, "run-config.yml"), 'w') as file:
        json.dump(config, file, indent=4)

    print("Integrating reference and query adata...")

    adata_batch_list = [reference_adata, model.adata]
    reference_query_adata = ad.concat(adata_batch_list, join="inner")

    batch_connectivities = []
    len_before_batch = 0
    for i in range(len(adata_batch_list)):
        if i == 0: # first batch
            after_batch_connectivities_extension = sp.csr_matrix(
                (adata_batch_list[0].shape[0],
                (reference_query_adata.shape[0] -
                adata_batch_list[0].shape[0])))
            batch_connectivities.append(sp.hstack(
                (adata_batch_list[0].obsp["spatial_connectivities"],
                after_batch_connectivities_extension)))
        elif i == (len(adata_batch_list) - 1): # last batch
            before_batch_connectivities_extension = sp.csr_matrix(
                (adata_batch_list[i].shape[0],
                (reference_query_adata.shape[0] -
                adata_batch_list[i].shape[0])))
            batch_connectivities.append(sp.hstack(
                (before_batch_connectivities_extension,
                adata_batch_list[i].obsp["spatial_connectivities"])))
        else: # middle batches
            before_batch_connectivities_extension = sp.csr_matrix(
                (adata_batch_list[i].shape[0], len_before_batch))
            after_batch_connectivities_extension = sp.csr_matrix(
                (adata_batch_list[i].shape[0],
                (reference_query_adata.shape[0] -
                adata_batch_list[i].shape[0] -
                len_before_batch)))
            batch_connectivities.append(sp.hstack(
                (before_batch_connectivities_extension,
                adata_batch_list[i].obsp["spatial_connectivities"],
                after_batch_connectivities_extension)))
        len_before_batch += adata_batch_list[i].shape[0]
    connectivities = sp.vstack(batch_connectivities)
    reference_query_adata.obsp["spatial_connectivities"] = connectivities

    model.adata = reference_query_adata

    model.adata.varm["nichecompass_gp_targets"] = reference_adata.varm["nichecompass_gp_targets"]
    model.adata.varm["nichecompass_gp_sources"] = reference_adata.varm["nichecompass_gp_sources"]
    model.adata.uns["nichecompass_gp_names"] = reference_adata.uns["nichecompass_gp_names"]
    model.adata.uns["nichecompass_genes_idx"] = reference_adata.uns["nichecompass_genes_idx"]
    model.adata.uns["nichecompass_target_genes_idx"] = reference_adata.uns["nichecompass_target_genes_idx"]
    model.adata.uns["nichecompass_source_genes_idx"] = reference_adata.uns["nichecompass_source_genes_idx"]

    print("Computing reference query latent embedding...")
    model.adata.obsm["nichecompass_latent"], _ = model.get_latent_representation(
       adata=model.adata,
       counts_key=None,
       cat_covariates_keys=config["train_query"]["reference_model"]["cat_covariates_keys"],
       only_active_gps=True,
       return_mu_std=True,
       node_batch_size=config["train_query"]["training"]["node_batch_size"])

    print("Computing neighbor graph...")
    sc.pp.neighbors(model.adata, use_rep="nichecompass_latent", key_added="nichecompass_latent")

    print("Computing latent umap embedding...")
    sc.tl.umap(model.adata, neighbors_key="nichecompass_latent")

    print("Exporting trained model...")
    model.save(
        dir_path=os.path.join(config["artefact_directory"], run_label, "joint"),
        adata_file_name=os.path.splitext(os.path.basename(config["train_query"]["dataset"]["file_path"]))[0] + ".h5ad",
        overwrite = True,
        save_adata = True)
    with open(os.path.join(config["artefact_directory"], run_label, "run-config.yml"), 'w') as file:
        json.dump(config, file, indent=4)


@app.command()
def intersect_datasets(adata_reference_path: str, adata_query_path: str, species: str, artefact_directory: str):

    run_timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    adjective = RandomWord().word(word_min_length=3, word_max_length=8, include_categories=["adjectives"])
    noun = RandomWord().word(word_min_length=3, word_max_length=8, include_categories=["nouns"])
    run_label = adjective + "_" + noun
    print(f"Starting run {run_label} at {run_timestamp}...")

    print("Loading run configuration...")
    config = {
        "adata_reference_path": adata_reference_path,
        "adata_query_path": adata_query_path,
        "species": species,
        "artefact_directory": artefact_directory
    }
    pprint(config)

    adata_reference_basename = os.path.basename(config["adata_reference_path"])
    adata_query_basename = os.path.basename(config["adata_query_path"])

    adata_reference = ad.read_h5ad(config["adata_reference_path"])
    adata_query = ad.read_h5ad(config["adata_query_path"])

    if config["species"] == "mouse":
        # harmonise case, since many datasets uppercase mouse gene symbols
        adata_reference.var_names = [var_name.capitalize() for var_name in adata_reference.var_names.tolist()]
        adata_query.var_names = [var_name.capitalize() for var_name in adata_query.var_names.tolist()]

    intersecting_genes = set(adata_reference.var_names.tolist()).intersection(set(adata_query.var_names.tolist()))

    adata_reference = adata_reference[:, list(intersecting_genes)]
    adata_query = adata_query[:, list(intersecting_genes)]

    adata_reference.obs["experiment"] = os.path.splitext(adata_reference_basename)[0]
    adata_query.obs["experiment"] = os.path.splitext(adata_query_basename)[0]

    adata_joint = ad.concat([adata_reference, adata_query], join="inner")

    print("Exporting datasets...")

    os.makedirs(os.path.join(config["artefact_directory"], run_label), exist_ok=True)
    adata_reference.write(os.path.join(config["artefact_directory"], run_label, adata_reference_basename))
    adata_query.write(os.path.join(config["artefact_directory"], run_label, adata_query_basename))
    adata_joint.write(os.path.join(config["artefact_directory"], run_label, f"{os.path.splitext(adata_reference_basename)[0]}_{os.path.splitext(adata_query_basename)[0]}.h5ad"))

    with open(os.path.join(config["artefact_directory"], run_label, "run-config.yml"), 'w') as file:
        json.dump(config, file, indent=4)


@app.command()
def export_latent(model_directory, adata_filename, artefact_directory: str):

    run_timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    adjective = RandomWord().word(word_min_length=3, word_max_length=8, include_categories=["adjectives"])
    noun = RandomWord().word(word_min_length=3, word_max_length=8, include_categories=["nouns"])
    run_label = adjective + "_" + noun
    print(f"Starting run {run_label} at {run_timestamp}...")

    print("Loading run configuration...")
    config = {
        "model_directory": model_directory,
        "adata_filename": adata_filename,
        "artefact_directory": artefact_directory
    }

    pprint(config)
    model = NicheCompass.load(
        dir_path=config["model_directory"],
        adata_file_name=config["adata_filename"],
        gp_names_key="nichecompass_gp_names")

    adata = ad.read_h5ad(os.path.join(config["model_directory"], config["adata_filename"]))

    active_gps = list(adata.uns[model.active_gp_names_key_])
    mu, std = model.get_latent_representation(
        adata=adata,
        counts_key=model.counts_key_,
        adj_key=model.adj_key_,
        cat_covariates_keys=model.cat_covariates_keys_,
        only_active_gps=False,
        return_mu_std=True,
        node_batch_size=model.node_batch_size_
    )

    active_gps_idx, active_gps_weights, _ = model.get_gp_data(selected_gps=active_gps)

    latent = {
        "mu": mu,
        "std": std,
        "active_gps": active_gps,
        "active_gps_idx": active_gps_idx,
        "active_gps_weights": active_gps_weights
    }

    print("Exporting latent...")

    os.makedirs(os.path.join(config["artefact_directory"], run_label), exist_ok=True)
    with open(os.path.join(config["artefact_directory"], run_label, "latent.pkl"), "wb") as file:
        pickle.dump(latent, file, pickle.HIGHEST_PROTOCOL)

    print("Loading gene program summary...")

    gp_summary_df = model.get_gp_summary()

    print("Exporting gene program summary...")

    with open(os.path.join(config["artefact_directory"], run_label, "gene_program_summary.pkl"), "wb") as file:
        pickle.dump(gp_summary_df, file, pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(config["artefact_directory"], run_label, "run-config.yml"), 'w') as file:
        json.dump(config, file, indent=4)


@app.command()
def cluster(config: str):

    run_timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    adjective = RandomWord().word(word_min_length=3, word_max_length=8, include_categories=["adjectives"])
    noun = RandomWord().word(word_min_length=3, word_max_length=8, include_categories=["nouns"])
    run_label = adjective + "_" + noun
    print(f"Starting run {run_label} at {run_timestamp}...")

    print(f"Loading run configuration...")
    with open(config) as file:
        config = json.load(file)
    pprint(config)

    print(f"Loading anndata...")
    adata = ad.read_h5ad(os.path.join(config["clustering"]["adata_directory"], config["clustering"]["file_path"]))

    print(f"Performing primary clustering...")
    sc.tl.leiden(adata=adata,
                 resolution=config["clustering"]["primary_cluster_resolution"],
                 key_added="nichecompass_latent_cluster",
                 neighbors_key="nichecompass_latent")
    niche_cell_frequency = adata.obs["nichecompass_latent_cluster"].value_counts()
    high_population_niches = niche_cell_frequency[niche_cell_frequency > config["clustering"]["primary_cluster_min_cells"]].index.tolist()
    adata.obs["nichecompass_latent_cluster"] = [
        niche if is_high_population is True else "unassigned" for niche, is_high_population in zip(
            adata.obs["nichecompass_latent_cluster"].tolist(),
            adata.obs["nichecompass_latent_cluster"].isin(high_population_niches).tolist()
        )
    ]

    print(f"Performing sub clustering...")
    clustered_adata = []
    for cluster in adata.obs["nichecompass_latent_cluster"].unique().tolist():
        adata_sample = adata[adata.obs["nichecompass_latent_cluster"] == cluster]
        if cluster == "unassigned":
            adata_sample.obs["nichecompass_latent_sub_cluster"] = "unassigned"
        if cluster != "unassigned":
            sc.tl.leiden(adata=adata_sample,
                         resolution=config["clustering"]["sub_cluster_resolution"],
                         key_added="nichecompass_latent_sub_cluster",
                         neighbors_key="nichecompass_latent")
            niche_cell_frequency = adata_sample.obs["nichecompass_latent_sub_cluster"].value_counts()
            high_population_niches = niche_cell_frequency[niche_cell_frequency > config["clustering"]["sub_cluster_min_cells"]].index.tolist()
            adata_sample.obs["nichecompass_latent_sub_cluster"] = [
                sub_cluster if is_high_population is True else "unassigned" for sub_cluster, is_high_population in zip(
                    adata_sample.obs["nichecompass_latent_sub_cluster"].tolist(),
                    adata_sample.obs["nichecompass_latent_sub_cluster"].isin(high_population_niches).tolist()
                )
            ]
        adata_sample.obs["nichecompass_latent_sub_cluster_label"] = adata_sample.obs[
                                                                        "nichecompass_latent_cluster"].astype(
            str) + "_" + adata_sample.obs["nichecompass_latent_sub_cluster"].astype(str)
        clustered_adata.append(adata_sample)
        print(f"Completed cluster {cluster}")
    adata = ad.concat(clustered_adata, uns_merge="same")

    print(f"Saving dataset...")
    os.makedirs(os.path.join(config["artefact_directory"], run_label), exist_ok=True)
    adata.write(os.path.join(config["artefact_directory"], run_label, config["clustering"]["file_path"]))


@app.command()
def enrichment(config: str):

    run_timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
    adjective = RandomWord().word(word_min_length=3, word_max_length=8, include_categories=["adjectives"])
    noun = RandomWord().word(word_min_length=3, word_max_length=8, include_categories=["nouns"])
    run_label = adjective + "_" + noun
    print(f"Starting run {run_label} at {run_timestamp}...")

    print(f"Loading run configuration...")
    with open(config) as file:
        config = json.load(file)
    pprint(config)

    print(f"Loading anndata...")
    adata = ad.read_h5ad(os.path.join(config["gene_program_enrichment"]["adata_directory"], config["gene_program_enrichment"]["adata_file_path"]))

    print(f"Loading latent export...")
    with open(os.path.join(config["gene_program_enrichment"]["latent_directory"], "latent.pkl"), "rb") as file:
        latent = pickle.load(file)
    active_gps = latent["active_gps"]
    mu = latent["mu"]
    std = latent["std"]
    active_gps_idx = latent["active_gps_idx"]

    mu_active_gps = mu[:, active_gps_idx]
    std_active_gps = std[:, active_gps_idx]

    print(f"Performing primary cluster enrichment...")
    cat_values = adata.obs["nichecompass_latent_cluster"]
    cats = cat_values.unique()
    results_primary_clusters = []

    for cat in cats:
        cat_mask = (cat_values == cat)
        comparison_cat_mask = ~cat_mask
        mu_cat = mu_active_gps[cat_mask]
        std_cat = std_active_gps[cat_mask]
        mu_comparison_cat = mu_active_gps[comparison_cat_mask]
        std_comparison_cat = std_active_gps[comparison_cat_mask]
        cat_idx = np.random.choice(cat_mask.sum(), 10000)
        comparison_cat_idx = np.random.choice(cat_mask.sum(), 10000)
        mu_cat_sample = mu_cat[cat_idx]
        std_cat_sample = std_cat[cat_idx]
        mu_comparison_cat_sample = mu_comparison_cat[comparison_cat_idx]
        std_comparison_cat_sample = std_comparison_cat[comparison_cat_idx]
        to_reduce = (- (mu_cat_sample - mu_comparison_cat_sample) / np.sqrt(
            2 * (std_cat_sample ** 2 + std_comparison_cat_sample ** 2)))
        to_reduce = 0.5 * erfc(to_reduce)
        p_h0 = np.mean(to_reduce, axis=0)
        p_h1 = 1.0 - p_h0
        epsilon = 1e-12
        log_bayes_factor = np.log(p_h0 + epsilon) - np.log(p_h1 + epsilon)
        zeros_mask = ((np.abs(mu_cat_sample).sum(0) == 0) | (np.abs(mu_comparison_cat_sample).sum(0) == 0))
        p_h0[zeros_mask] = 0
        p_h1[zeros_mask] = 0
        log_bayes_factor[zeros_mask] = 0
        zipped = zip(active_gps, p_h0, p_h1, log_bayes_factor)
        cat_results = [
            {"category": cat, "gene_program": gp, "p_h0": p_h0, "p_h1": p_h1, "log_bayes_factor": log_bayes_factor} for
            gp, p_h0, p_h1, log_bayes_factor in zipped]
        for result in cat_results:
            results_primary_clusters.append(result)

    results_primary_clusters = pd.DataFrame(results_primary_clusters)
    results_primary_clusters["abs_log_bayes_factor"] = np.abs(results_primary_clusters["log_bayes_factor"])
    results_primary_clusters["type"] = "primary_cluster"

    print(f"Performing sub cluster enrichment...")
    cat_values = adata.obs["nichecompass_latent_sub_cluster_label"]
    cats = cat_values.unique()
    results_sub_clusters = []

    for cat in cats:
        cat_mask = (cat_values == cat)
        comparison_cat_mask = ~cat_mask
        mu_cat = mu_active_gps[cat_mask]
        std_cat = std_active_gps[cat_mask]
        mu_comparison_cat = mu_active_gps[comparison_cat_mask]
        std_comparison_cat = std_active_gps[comparison_cat_mask]
        cat_idx = np.random.choice(cat_mask.sum(), 10000)
        comparison_cat_idx = np.random.choice(cat_mask.sum(), 10000)
        mu_cat_sample = mu_cat[cat_idx]
        std_cat_sample = std_cat[cat_idx]
        mu_comparison_cat_sample = mu_comparison_cat[comparison_cat_idx]
        std_comparison_cat_sample = std_comparison_cat[comparison_cat_idx]
        to_reduce = (- (mu_cat_sample - mu_comparison_cat_sample) / np.sqrt(
            2 * (std_cat_sample ** 2 + std_comparison_cat_sample ** 2)))
        to_reduce = 0.5 * erfc(to_reduce)
        p_h0 = np.mean(to_reduce, axis=0)
        p_h1 = 1.0 - p_h0
        epsilon = 1e-12
        log_bayes_factor = np.log(p_h0 + epsilon) - np.log(p_h1 + epsilon)
        zeros_mask = ((np.abs(mu_cat_sample).sum(0) == 0) | (np.abs(mu_comparison_cat_sample).sum(0) == 0))
        p_h0[zeros_mask] = 0
        p_h1[zeros_mask] = 0
        log_bayes_factor[zeros_mask] = 0
        zipped = zip(active_gps, p_h0, p_h1, log_bayes_factor)
        cat_results = [
            {"category": cat, "gene_program": gp, "p_h0": p_h0, "p_h1": p_h1, "log_bayes_factor": log_bayes_factor} for
            gp, p_h0, p_h1, log_bayes_factor in zipped]
        for result in cat_results:
            results_sub_clusters.append(result)

    results_sub_clusters = pd.DataFrame(results_sub_clusters)
    results_sub_clusters["abs_log_bayes_factor"] = np.abs(results_sub_clusters["log_bayes_factor"])
    results_sub_clusters["type"] = "sub_cluster"

    print(f"Exporting results...")
    results = pd.concat([results_primary_clusters, results_sub_clusters])
    os.makedirs(os.path.join(config["artefact_directory"], run_label), exist_ok=True)
    results.to_csv(os.path.join(config["artefact_directory"], run_label, "enrichment_results.csv"))


if __name__ == "__main__":
    app()
