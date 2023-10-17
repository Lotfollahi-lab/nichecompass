import config
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

def main():
    print("Running NicheCompass with configuration:")
    config_object = config.Config("test_config.json")
    for key, value in config_object.options.items():
        print(key, ":", value)

    # retrieve gene programs
    print("---Retrieving gene programs---")

    print("Retrieving OmniPath gene programs")
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

    print("Retrieving NicheNet gene programs")
    nichenet_gp_dict = extract_gp_dict_from_nichenet_lrt_interactions(
        species=config_object.options["species"],
        version="v2",
        keep_target_genes_ratio=1,
        max_n_target_genes_per_gp=250,
        load_from_disk=False,
        save_to_disk=True,
        lr_network_file_path=config_object.options["nichenet_lr_network_file_path"],
        ligand_target_matrix_file_path=config_object.options["nichenet_ligand_target_matrix_file_path"],
        gene_orthologs_mapping_file_path=config_object.options["gene_orthologs_mapping_file_path"],
        plot_gp_gene_count_distributions=True)
    nichenet_source_genes = get_unique_genes_from_gp_dict(
        gp_dict=nichenet_gp_dict,
        retrieved_gene_entities=["sources"])

    print("Retrieving MEBOCOST gene programs")
    mebocost_gp_dict = extract_gp_dict_from_mebocost_es_interactions(
        dir_path=config_object.options["mebocost_enzyme_sensor_interactions_folder_path"],
        species=config_object.options["species"],
        plot_gp_gene_count_distributions=True)
    mebocost_genes = get_unique_genes_from_gp_dict(
        gp_dict=mebocost_gp_dict,
        retrieved_gene_entities=["sources", "targets"])

    print("Combining and filtering gene programs")
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
    print(f"{len(combined_gp_dict)} gene programs before combining and filtering")
    print(f"{len(combined_new_gp_dict)} gene programs after combining and filtering")

    # load dataset
    print("Reading dataset")
    adata = sc.read_h5ad(f"{so_data_folder_path}/{dataset}_batch1.h5ad")

    # compute spatial neighborhood
    print("Computing spatial neighborhood")
    sq.gr.spatial_neighbors(adata,
                            coord_type="generic",
                            spatial_key=config_object.options["spatial_key"],
                            n_neighs=config_object.options["n_neighbors"])
    print("Making adjacency matrix symmetric")
    adata.obsp[config_object.options["adj_key"]] = (
        adata.obsp[config_object.options["adj_key"]].maximum(
            adata.obsp[config_object.options["adj_key"]].T))

    print("Done")


if __name__ == '__main__':
    main()
