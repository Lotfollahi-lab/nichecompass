from datetime import datetime

import torch.cuda

from nichecompass import config
import os
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import squidpy as sq
from matplotlib import gridspec
from nichecompass.models import NicheCompass
from nichecompass.utils import (add_gps_from_gp_dict_to_adata,
                                aggregate_obsp_matrix_per_cell_type,
                                create_cell_type_chord_plot_from_df,
                                create_new_color_dict,
                                extract_gp_dict_from_mebocost_es_interactions,
                                extract_gp_dict_from_nichenet_lrt_interactions,
                                extract_gp_dict_from_omnipath_lr_interactions,
                                filter_and_combine_gp_dict_gps,
                                generate_enriched_gp_info_plots,
                                get_unique_genes_from_gp_dict)
import nichecompass.report
import typer

app = typer.Typer()

@app.command()
def check_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')


@app.command()
def start():
    print(f"{datetime.now()}: Running NicheCompass with configuration:")
    config_object = config.Config("test_config.json")
    for key, value in config_object.options.items():
        print(key, ":", value)

    # retrieve gene programs
    print(f"{datetime.now()}: ---Retrieving gene programs---")

    print(f"{datetime.now()}: Retrieving OmniPath gene programs")
    omnipath_gp_dict = extract_gp_dict_from_omnipath_lr_interactions(
        species=config_object.options["species"],
        min_curation_effort=0,
        load_from_disk=True,
        save_to_disk=True,
        lr_network_file_path=config_object.options["omnipath_lr_network_file_path"],
        gene_orthologs_mapping_file_path=config_object.options["gene_orthologs_mapping_file_path"],
        plot_gp_gene_count_distributions=True)
    omnipath_genes = get_unique_genes_from_gp_dict(
        gp_dict=omnipath_gp_dict,
        retrieved_gene_entities=["sources", "targets"])

    print(f"{datetime.now()}: Retrieving NicheNet gene programs")
    nichenet_gp_dict = extract_gp_dict_from_nichenet_lrt_interactions(
        species=config_object.options["species"],
        version="v2",
        keep_target_genes_ratio=1,
        max_n_target_genes_per_gp=250,
        load_from_disk=True,
        save_to_disk=True,
        lr_network_file_path=config_object.options["nichenet_lr_network_file_path"],
        ligand_target_matrix_file_path=config_object.options["nichenet_ligand_target_matrix_file_path"],
        gene_orthologs_mapping_file_path=config_object.options["gene_orthologs_mapping_file_path"],
        plot_gp_gene_count_distributions=True)
    nichenet_source_genes = get_unique_genes_from_gp_dict(
        gp_dict=nichenet_gp_dict,
        retrieved_gene_entities=["sources"])

    print(f"{datetime.now()}: Retrieving MEBOCOST gene programs")
    mebocost_gp_dict = extract_gp_dict_from_mebocost_es_interactions(
        dir_path=config_object.options["mebocost_enzyme_sensor_interactions_folder_path"],
        species=config_object.options["species"],
        plot_gp_gene_count_distributions=True)
    mebocost_genes = get_unique_genes_from_gp_dict(
        gp_dict=mebocost_gp_dict,
        retrieved_gene_entities=["sources", "targets"])

    print(f"{datetime.now()}: Combining and filtering gene programs")
    combined_gp_dict = dict(omnipath_gp_dict)
    combined_gp_dict.update(nichenet_gp_dict)
    combined_gp_dict.update(mebocost_gp_dict)
    combined_new_gp_dict = filter_and_combine_gp_dict_gps(
        gp_dict=combined_gp_dict,
        gp_filter_mode="subset",
        combine_overlap_gps=True,
        overlap_thresh_source_genes=0.9,
        overlap_thresh_target_genes=0.9,
        overlap_thresh_genes=0.9)
    print(f"..{len(combined_gp_dict)} gene programs before combining and filtering")
    print(f"..{len(combined_new_gp_dict)} gene programs after combining and filtering")

    # load dataset
    print(f"{datetime.now()}: ---Processing dataset---")

    print(f"{datetime.now()}: Reading dataset")
    adata = sc.read_h5ad(config_object.options["data_path"])

    # compute spatial neighborhood
    print(f"{datetime.now()}: Computing spatial neighborhood")
    sq.gr.spatial_neighbors(adata,
                            coord_type="generic",
                            spatial_key=config_object.options["spatial_key"],
                            n_neighs=config_object.options["n_neighbors"])
    print(f"{datetime.now()}: Making adjacency matrix symmetric")
    adata.obsp[config_object.options["adj_key"]] = (
        adata.obsp[config_object.options["adj_key"]].maximum(
            adata.obsp[config_object.options["adj_key"]].T))

    print(f"{datetime.now()}: Filtering genes")
    if config_object.options["filter_genes"]:
        gp_dict_genes = get_unique_genes_from_gp_dict(
            gp_dict=combined_new_gp_dict,
            retrieved_gene_entities=["sources", "targets"])
        print(f"..starting with {len(adata.var_names)} genes")

        sc.pp.filter_genes(
            adata,
            min_cells=0)
        print(f"..retaining {len(adata.var_names)} genes expressed in >0 cell(s)")

        sc.pp.highly_variable_genes(
            adata,
            layer=config_object.options["counts_key"],
            n_top_genes=config_object.options["n_hvg"],
            flavor="seurat_v3",
            subset=False)
        gp_relevant_genes = [gene.upper() for gene in list(set(
            omnipath_genes + nichenet_source_genes + mebocost_genes))]
        adata.var["gp_relevant"] = (adata.var.index.str.upper().isin(gp_relevant_genes))
        adata.var["keep_gene"] = (adata.var["gp_relevant"] | adata.var["highly_variable"])
        adata = adata[:, adata.var["keep_gene"] == True]
        print(f"..retaining {len(adata.var_names)} highly variable or gene program relevant genes")

        adata = (adata[:, adata.var_names[adata.var_names.str.upper().isin(
            [gene.upper() for gene in gp_dict_genes])].sort_values()])
        print(f"..retaining {len(adata.var_names)} genes included in >0 gene program(s)")

    print(f"{datetime.now()}: Adding gene program dictionary as binary mask to adata")
    add_gps_from_gp_dict_to_adata(
        gp_dict=combined_new_gp_dict,
        adata=adata,
        gp_targets_mask_key=config_object.options["gp_targets_mask_key"],
        gp_targets_categories_mask_key=config_object.options["gp_targets_categories_mask_key"],
        gp_sources_mask_key=config_object.options["gp_sources_mask_key"],
        gp_sources_categories_mask_key=config_object.options["gp_sources_categories_mask_key"],
        gp_names_key=config_object.options["gp_names_key"],
        min_genes_per_gp=1,
        min_source_genes_per_gp=0,
        min_target_genes_per_gp=0,
        max_genes_per_gp=None,
        max_source_genes_per_gp=None,
        max_target_genes_per_gp=None)

    print(f"{datetime.now()}: Generating visualisation of cell-level annotated data in physical space")

    cell_type_colors = create_new_color_dict(
        adata=adata,
        cat_key=config_object.options["cell_type_key"])

    print(f"..number of nodes (observations): {adata.layers['counts'].shape[0]}")
    print(f"..number of node features (genes): {adata.layers['counts'].shape[1]}")

    ax = sc.pl.spatial(adata,
                        color=config_object.options["cell_type_key"],
                        palette=cell_type_colors,
                        spot_size=config_object.options["spot_size"],
                        return_fig=True)
    fig = ax[0].figure

    report_image = nichecompass.report.ReportItemImage(fig=fig, alt="Spatial map", caption="A spatial map")
    report_section = nichecompass.report.ReportSection(title="Spatial map", description="A spatial map")
    report_section.add_item(report_image)
    report = nichecompass.report.Report()
    report.add_section(report_section)

    with open("report.html", "w") as report_file:
        report_file.write(report.render())

    print(f"{datetime.now()}: Initializing NicheCompass model")

    lambda_edge_recon = 0
    lambda_gene_expr_recon = 300000

    model = NicheCompass(adata,
                         counts_key=config_object.options["counts_key"],
                         adj_key=config_object.options["adj_key"],
                         gp_names_key=config_object.options["gp_names_key"],
                         active_gp_names_key=config_object.options["active_gp_names_key"],
                         gp_targets_mask_key=config_object.options["gp_targets_mask_key"],
                         gp_targets_categories_mask_key=config_object.options["gp_targets_categories_mask_key"],
                         gp_sources_mask_key=config_object.options["gp_sources_mask_key"],
                         gp_sources_categories_mask_key=config_object.options["gp_sources_categories_mask_key"],
                         latent_key=config_object.options["latent_key"],
                         active_gp_thresh_ratio=config_object.options["active_gp_thresh_ratio"],
                         node_label_method=config_object.options["node_label_method"])

    print(f"{datetime.now()}: Training NicheCompass model")

    model.train(n_epochs=config_object.options["n_epochs"],
                n_epochs_all_gps=config_object.options["n_epochs_all_gps"],
                lr=config_object.options["lr"],
                lambda_edge_recon=config_object.options["lambda_edge_recon"],
                lambda_gene_expr_recon=config_object.options["lambda_gene_expr_recon"],
                lambda_l1_masked=config_object.options["lambda_l1_masked"],
                l1_targets_categories=config_object.options["l1_targets_categories"],
                l1_sources_categories=config_object.options["l1_sources_categories"],
                edge_batch_size=config_object.options["edge_batch_size"],
                use_cuda_if_available=config_object.options["use_cuda_if_available"],
                verbose=False)

    sc.pp.neighbors(model.adata,
                    use_rep=config_object.options["latent_key"],
                    key_added=config_object.options["latent_key"])

    print(f"{datetime.now()}: Generating latent space UMAP")

    sc.tl.umap(model.adata,
               neighbors_key=config_object.options["latent_key"])

    print(f"{datetime.now()}: Saving NicheCompass model")

    model.save(dir_path=config_object.options["model_folder_path"],
               overwrite=True,
               save_adata=True,
               adata_file_name="adata.h5ad")

    print(f"{datetime.now()}: Done")


if __name__ == "__main__":
    app()
