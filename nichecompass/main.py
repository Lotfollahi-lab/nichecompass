import typer
import json
import pickle
from pprint import pprint
from datetime import datetime

import anndata as ad
import mlflow
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import squidpy as sq

from nichecompass.models import NicheCompass
from nichecompass.utils import (add_gps_from_gp_dict_to_adata,
                                add_multimodal_mask_to_adata,
                                extract_gp_dict_from_collectri_tf_network,
                                extract_gp_dict_from_mebocost_es_interactions,
                                extract_gp_dict_from_nichenet_lrt_interactions,
                                extract_gp_dict_from_omnipath_lr_interactions,
                                filter_and_combine_gp_dict_gps,
                                generate_multimodal_mapping_dict,
                                get_gene_annotations,
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
            "min_cell_gene_thresh_ratio": 0.1
        },
        "graph": {
            "n_neighbors": 12,

        }
    }

    with open(output, 'w') as file:
        json.dump(default_config, file, indent=4)





"""
        "data": {
            "dataset": None,
            "reference_batches": None, # could just assume using all in adata, note cannot be a list
            "counts_key": None # shouldn't it just assume the raw data?
            "condition_key": "batch", # these could be aggregated
            "cat_covariates_keys": None, # ^ # cannot be a list of [None]
            "cat_covariates_no_edges": None, # don't even know what this does?, cannot be a list of [None]
            "n_neighbors": 12,
            "spatial_key": "spatial", # could just be set
            "adj_key": "spatial_connectivities", # could just be set
            "mapping_entity_key": "mapping_entity", # could just be set
            "filter_genes": False, # could be moved over
            "n_hvg": 0, # could be moved over
            "n_svg": 3000, # could be moved over
            "n_svp": 3000, # could be moved over
            "gp_targets_mask_key": "nichecompass_gp_targets", # could just be set
            "gp_sources_mask_key": "nichecompass_gp_sources", # could just be set
            "gp_names_key": "nichecompass_gp_names", # could just be set
            "include_atac_modality": False, # could be removed for this cli function
            "filter_peaks": False, # ^
            "min_cell_peak_thresh_ratio": 0.0005, # ^
            "min_cell_gene_thresh_ratio": 0.0005 # don't even know what this does?
        },
        "model": {
            "model_label": "reference",
            "active_gp_names_key": "nichecompass_active_gp_names",
            "latent_key": "nichecompass_latent",
            "active_gp_thresh_ratio": 0.05,
            "n_addon_gp": 10,
            "active_gp_thresh_ratio": 0.05,
            "gene_expr_recon_dist": "nb",
            "cat_covariates_embeds_injection": ["gene_expr_decoder"], # note must be an empty list and not null
            "cat_covariates_embeds_nums": None, # cannot be a list of [None]
            "log_variational": True,
            "node_label_method": "one-hop-norm",
            "n_layers_encoder": 1,
            "n_fc_layers_encoder": 1,
            "conv_layer_encoder": "gcnconv",
            "n_hidden_encoder": None,
            "n_epochs": 400,
            "n_epochs_all_gps": 25,
            "n_epochs_no_cat_covariates_contrastive": 5,
            "lr": 0.001,
            "lambda_edge_recon": 5000000,
            "lambda_gene_expr_recon": 3000,
            "lambda_chrom_access_recon": 1000,
            "lambda_cat_covariates_contrastive": 0,
            "contrastive_logits_pos_ratio": 0,
            "contrastive_logits_neg_ratio": 0,
            "lambda_group_lasso": 0,
            "lambda_l1_masked": 0,
            "lambda_l1_addon": 0,
            "edge_batch_size": 256,
            "node_batch_size": None,
            "n_sampled_neighbors": 1
        },
        "timestamp_suffix": "" # not really needed
    }
    # should add local: {artefact_directory}

"""


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
            dir_path=config["gene_programs"]["mebocost"]["metabolite_enzyme_sensor_directory_path"], #fixme remove dependency on file
            species=config["gene_programs"]["species"],
            plot_gp_gene_count_distributions=False)
        gene_programs.update(mebocost_gene_programs)

    if "collectri" in config["gene_programs"]["sources"]:
        collectri_gene_programs = extract_gp_dict_from_collectri_tf_network(
            species=config["gene_programs"]["species"],
            plot_gp_gene_count_distributions=False)
        gene_programs.update(collectri_gene_programs)

    if "brain_marker" in config["gene_programs"]["sources"]: #fixme this should be moved to another function
        # Add spatial layer marker gene GPs
        # Load experimentially validated marker genes
        validated_marker_genes_df = pd.read_csv(f"marker_gps/Validated_markers_MM_layers.tsv", #fixme
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
        overlap_thresh_genes=0.9) #FIXME these should be defaults in the fun rather than here

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
    adata.var["expressed"] = sc.pp.filter_genes(adata, min_cells=min_cells, inplace=False)[0]

    print(f"Annotating {config['gene_filters']['n_highly_variable']} highly variable genes...")
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=config["gene_filters"]["n_highly_variable"],
        flavor="seurat_v3",
        batch_key=config["dataset"]["library_key"])

    print(f"Annotating {config['gene_filters']['n_spatially_variable']} spatially variable genes...")
    sq.gr.spatial_autocorr(adata, mode="moran", genes=adata.var_names)
    sv_genes = adata.uns["moranI"].index[:config["gene_filters"]["n_spatially_variable"]].tolist()
    adata.var["spatially_variable"] = adata.var_names.isin(sv_genes)

    print(f"Annotating genes present in gene programs...")
    adata.var["gene_program_relevant"] = adata.var.index.str.upper().isin(gene_program_genes)

    print("Applying filtering...")
    adata.var["keep_gene"] = (adata.var["expressed"] &
                              (adata.var["gene_program_relevant"] |
                              adata.var["highly_variable"] |
                              adata.var["spatially_variable"]))
    adata = adata[:, adata.var["keep_gene"] == True]
    print(f"Retaining {len(adata.var_names)} genes.")

    print("Adding gene programs to dataset...")
    add_gps_from_gp_dict_to_adata(gp_dict=gene_programs, adata=adata)

    print("Exporting dataset...")
    with open(config["dataset"]["export_file_path"], "wb") as file:
        pickle.dump(adata, file, pickle.HIGHEST_PROTOCOL)


@app.command()
def train():
    pass

    current_timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    print(f"Run timestamp: {current_timestamp}")



"""


    print("Initializing model...")
    model = NicheCompass(adata,
                         adata_atac,
                         counts_key=args.counts_key,
                         adj_key=args.adj_key,
                         cat_covariates_embeds_injection=args.cat_covariates_embeds_injection,
                         cat_covariates_keys=args.cat_covariates_keys,
                         cat_covariates_no_edges=args.cat_covariates_no_edges,
                         cat_covariates_embeds_nums=args.cat_covariates_embeds_nums,
                         gp_names_key=args.gp_names_key,
                         active_gp_names_key=args.active_gp_names_key,
                         gp_targets_mask_key=args.gp_targets_mask_key,
                         gp_sources_mask_key=args.gp_sources_mask_key,
                         latent_key=args.latent_key,
                         n_addon_gp=args.n_addon_gp,
                         active_gp_thresh_ratio=args.active_gp_thresh_ratio,
                         gene_expr_recon_dist=args.gene_expr_recon_dist,
                         n_fc_layers_encoder=args.n_fc_layers_encoder,
                         n_layers_encoder=args.n_layers_encoder,
                         conv_layer_encoder=args.conv_layer_encoder,
                         n_hidden_encoder=args.n_hidden_encoder,
                         log_variational=args.log_variational,
                         node_label_method=args.node_label_method)

    print("Training model...")
    model.train(n_epochs=args.n_epochs,
                n_epochs_all_gps=args.n_epochs_all_gps,
                n_epochs_no_cat_covariates_contrastive=args.n_epochs_no_cat_covariates_contrastive,
                lr=args.lr,
                lambda_edge_recon=args.lambda_edge_recon,
                lambda_gene_expr_recon=args.lambda_gene_expr_recon,
                lambda_chrom_access_recon=args.lambda_chrom_access_recon,
                lambda_cat_covariates_contrastive=args.lambda_cat_covariates_contrastive,
                contrastive_logits_pos_ratio=args.contrastive_logits_pos_ratio,
                contrastive_logits_neg_ratio=args.contrastive_logits_neg_ratio,
                lambda_group_lasso=args.lambda_group_lasso,
                lambda_l1_masked=args.lambda_l1_masked,
                lambda_l1_addon=args.lambda_l1_addon,
                edge_batch_size=args.edge_batch_size,
                node_batch_size=args.node_batch_size,
                n_sampled_neighbors=args.n_sampled_neighbors,
                mlflow_experiment_id=mlflow_experiment_id,
                verbose=True)

    print("Computing neighbor graph...")
    sc.pp.neighbors(model.adata,
                    use_rep=args.latent_key,
                    key_added=args.latent_key)

    print("Computing UMAP embedding...")
    sc.tl.umap(model.adata,
               neighbors_key=args.latent_key)

    print("Exporting dataset...")
    model.adata.write(
        f"{result_folder_path}/{args.dataset}_{args.model_label}.h5ad")

    print("Exporting trained model...")
    model.save(dir_path=model_folder_path,
               overwrite=True,
               save_adata=True,
               adata_file_name=f"{args.dataset}_{args.model_label}.h5ad",
               save_adata_atac=save_adata_atac,
               adata_atac_file_name=f"{args.dataset}_{args.model_label}_atac.h5ad")


"""


if __name__ == "__main__":
    app()
