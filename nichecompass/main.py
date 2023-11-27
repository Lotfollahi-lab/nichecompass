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
        "gene_programs": {
            "sources": ["omnipath", "nichenet", "mebocost", "brain_marker"],
            "species": "mouse",
            "filter_mode": "subset",
            "gene_orthologs_mapping_file_path": "human_mouse_gene_orthologs.csv",
            "export_file_path": "gene_programs.pkl",
            "genes_export_file_path": "relevant_genes.pkl",
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
            "export_file_path": "spatial_dataset.pkl",
        },
        "gene_filters": {
            "n_highly_variable": 0,
            "n_spatially_variable": 0,
            "min_cell_gene_thresh_ratio": 0.1,
            "gene_program_relevant": True
        },
        "graph": {
            "n_neighbors": 12,
        },
        "model": {
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
            "node_batch_size": None,
            "n_sampled_neighbors": 1,
            "artefact_directory": "artefacts"
        }
    }

    with open(output, 'w') as file:
        json.dump(default_config, file, indent=4)



@app.command()
def build_gene_programs(config: str):

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
    with open(config["gene_programs"]["export_file_path"], "wb") as file:
        pickle.dump(filtered_gene_programs, file, pickle.HIGHEST_PROTOCOL)


@app.command()
def build_dataset(config: str):

    print("Loading run configuration...")
    with open(config) as file:
        config = json.load(file)
    pprint(config)

    print("Loading gene programs...")
    with open(config["gene_programs"]["export_file_path"], "rb") as file:
        gene_programs = pickle.load(file)

    print("Reading dataset...")
    adata = ad.read_h5ad(config["dataset"]["file_path"])
    sq.gr.spatial_neighbors(adata,
                            coord_type="generic",
                            spatial_key=config["dataset"]["spatial_key"],
                            library_key=config["dataset"]["library_key"],
                            n_neighs=config["graph"]["n_neighbors"])
    adjacency = adata.obsp["spatial_connectivities"]
    symmetrical_adjacency = adjacency.maximum(adjacency.T)
    adata.obsp["spatial_connectivities"] = symmetrical_adjacency

    print("Filtering genes...")

    gene_program_genes = get_unique_genes_from_gp_dict(
        gp_dict=gene_programs,
        retrieved_gene_entities=["sources", "targets"])

    print(f"Annotating based on expression in at least {config['gene_filters']['min_cell_gene_thresh_ratio'] * 100}% of cells...")
    min_cells = int(adata.shape[0] * config["gene_filters"]["min_cell_gene_thresh_ratio"])
    adata.var["expressed"] = sc.pp.filter_genes(adata, min_cells=min_cells, inplace=False)[0].tolist()

    print(f"Annotating {config['gene_filters']['n_highly_variable']} highly variable genes...")
    n_top_genes = (0 if config["gene_filters"]["n_highly_variable"] is None else config["gene_filters"]["n_highly_variable"])
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        flavor="seurat_v3",
        batch_key=config["dataset"]["library_key"])

    print(f"Annotating {config['gene_filters']['n_spatially_variable']} spatially variable genes...")
    sq.gr.spatial_autocorr(adata, mode="moran", genes=adata.var_names)
    sv_genes = adata.uns["moranI"].index[:config["gene_filters"]["n_spatially_variable"]].tolist()
    adata.var["spatially_variable"] = adata.var_names.isin(sv_genes)

    print(f"Annotating genes present in gene programs...")
    adata.var["gene_program_relevant"] = adata.var.index.isin(gene_program_genes)

    print("Applying filtering...")
    adata.var["retained_gene"] = [True] * adata.shape[1]

    if config["gene_filters"]["min_cell_gene_thresh_ratio"] is not None:
        adata.var["retained_gene"] = adata.var["retained_gene"] & adata.var["expressed"]

    if config["gene_filters"]["n_highly_variable"] is not None:
        adata.var["retained_gene"] = adata.var["retained_gene"] & adata.var["highly_variable"]

    if config["gene_filters"]["n_spatially_variable"] is not None:
        adata.var["retained_gene"] = adata.var["retained_gene"] & adata.var["spatially_variable"]

    if config["gene_filters"]["gene_program_relevant"] is not None:
        adata.var["retained_gene"] = adata.var["retained_gene"] & adata.var["gene_program_relevant"]

    adata = adata[:, adata.var["retained_gene"] == True]
    print(f"Retaining {len(adata.var_names)} genes.")

    print("Adding gene programs to dataset...")
    add_gps_from_gp_dict_to_adata(gp_dict=gene_programs, adata=adata)

    print("Exporting dataset...")
    adata.write_h5ad(config["dataset"]["export_file_path"])


@app.command()
def train(config: str):

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
    adata = ad.read_h5ad(config["dataset"]["export_file_path"])

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
    model.save(
        dir_path=os.path.join(config["training"]["artefact_directory"], run_label),
        adata_file_name=os.path.splitext(os.path.basename(config["dataset"]["export_file_path"]))[0] + ".h5ad",
        overwrite = True,
        save_adata = True)
    with open(os.path.join(config["training"]["artefact_directory"], run_label, "run-config.yml"), 'w') as file:
        json.dump(config, file, indent=4)


if __name__ == "__main__":
    app()
