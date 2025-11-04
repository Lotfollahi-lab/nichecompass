"""
This module contains utilities to retrieve interpretable prior knowledge gene
programs for use by the NicheCompass model.
"""

import copy
from typing import Literal, Optional

import decoupler as dc
import numpy as np
import omnipath as op
import pandas as pd
from anndata import AnnData

from .utils import load_R_file_as_df, create_gp_gene_count_distribution_plots


def add_gps_from_gp_dict_to_adata(
        gp_dict: dict,
        adata: AnnData,
        genes_uppercase: bool=True,
        gp_targets_mask_key: str="nichecompass_gp_targets",
        gp_targets_categories_mask_key: str="nichecompass_gp_targets_categories",
        targets_categories_label_encoder_key: str="nichecompass_targets_categories_label_encoder",
        gp_sources_mask_key: str="nichecompass_gp_sources",
        gp_sources_categories_mask_key: str="nichecompass_gp_sources_categories",
        sources_categories_label_encoder_key: str="nichecompass_sources_categories_label_encoder",
        gp_names_key: str="nichecompass_gp_names",
        source_genes_idx_key: str="nichecompass_source_genes_idx",
        target_genes_idx_key: str="nichecompass_target_genes_idx",
        genes_idx_key: str="nichecompass_genes_idx",
        min_genes_per_gp: int=1,
        min_source_genes_per_gp: int=0,
        min_target_genes_per_gp: int=0,
        max_genes_per_gp: Optional[int]=None,
        max_source_genes_per_gp: Optional[int]=None,
        max_target_genes_per_gp: Optional[int]=None,
        filter_genes_not_in_masks: bool=False,
        add_fc_gps_instead_of_gp_dict_gps: bool=False,
        plot_gp_gene_count_distributions: bool=False):
    """
    Add gene programs defined in a gene program dictionary to an AnnData object.
    This is done by converting the gene program lists of gene program target and
    source genes to binary masks and aligning the masks with genes for which
    gene expression is available in the AnnData object.

    Parts of the implementation are inspired by
    https://github.com/theislab/scarches/blob/master/scarches/utils/annotations.py#L5
    (01.10.2022).

    Parameters
    ----------
    gp_dict:
        Nested dictionary containing the gene programs with keys being gene 
        program names and values being dictionaries with keys ´targets´ and 
        ´sources´, where ´targets´ contains a list of the names of genes in the
        gene program for the reconstruction of the gene expression of the node
        itself (receiving node) and ´sources´ contains a list of the names of
        genes in the gene program for the reconstruction of the gene expression
        of the node's neighbors (transmitting nodes).
    adata:
        AnnData object to which the gene programs will be added.
    genes_uppercase:
        If `True`, convert the gene names in the adata and in the gene program
        dictionary to uppercase for comparison.
    gp_targets_mask_key:
        Key in ´adata.varm´ where the binary gene program mask for target genes
        of a gene program will be stored (target genes are used for the 
        reconstruction of the gene expression of the node itself (receiving node
        )).
    gp_sources_mask_key:
        Key in ´adata.varm´ where the binary gene program mask for source genes
        of a gene program will be stored (source genes are used for the 
        reconstruction of the gene expression of the node's neighbors 
        (transmitting nodes)).
    gp_names_key:
        Key in ´adata.uns´ where the gene program names will be stored.
    source_genes_idx_key:
        Key in ´adata.uns´ where the index of the source genes that are in the
        gene program mask will be stored.
    target_genes_idx_key:
        Key in ´adata.uns´ where the index of the target genes that are in the
        gene program mask will be stored.
    genes_idx_key:
        Key in ´adata.uns´ where the index of a concatenated vector of target
        and source genes that are in the gene program masks will be stored.
    min_genes_per_gp:
        Minimum number of genes in a gene program inluding both target and 
        source genes that need to be available in the adata (gene expression has
        been probed) for a gene program not to be discarded.
    min_source_genes_per_gp:
        Minimum number of source genes in a gene program that need to be 
        available in the adata (gene expression has been probed) for a gene 
        program not to be discarded.
    min_target_genes_per_gp:
        Minimum number of target genes in a gene program that need to be 
        available in the adata (gene expression has been probed) for a gene 
        program not to be discarded.
    max_genes_per_gp:
        Maximum number of genes in a gene program inluding both target and 
        source genes that can be available in the adata (gene expression has 
        been probed) for a gene program not to be discarded.
    max_source_genes_per_gp:
        Maximum number of source genes in a gene program that can be available 
        in the adata (gene expression has been probed) for a gene program not to
        be discarded.
    max_target_genes_per_gp:
        Maximum number of target genes in a gene program that can be available 
        in the adata (gene expression has been probed) for a gene program not to
        be discarded.
    filter_genes_not_in_masks:
        If ´True´, remove the genes that are not in the gp masks from the adata
        object.
    add_fc_gps_instead_of_gp_dict_gps:
        Note: this parameter is just used for ablation studies. If ´True´,
        ignores the gene programs from the gp dict and instead creates a mask
        of fully-connected gene programs (same amount as gps in the gp dict).
    plot_gp_gene_count_distributions:
        If ´True´, display the distribution of gene programs per number of
        source and target genes.
    """
    # Retrieve probed genes from adata
    adata_genes = (adata.var_names.str.upper() if genes_uppercase
                   else adata.var_names)
    
    # Just for ablation studies, create fully-connected mask
    if add_fc_gps_instead_of_gp_dict_gps:
        gp_targets_mask = [[1 for _, _ in gp_dict.items()]
                           for gene in adata_genes]
        gp_targets_mask = np.asarray(gp_targets_mask, dtype="int32")
        gp_sources_mask = [[1 for _, _ in gp_dict.items()]
                           for gene in adata_genes]
        gp_sources_mask = np.asarray(gp_sources_mask, dtype="int32")
        
        gp_targets_categories_mask = [[0 for _, _ in gp_dict.items()]
                                      for gene in adata_genes]
        gp_targets_categories_mask = np.asarray(gp_targets_categories_mask,
                                                dtype="int32")
        gp_sources_categories_mask = [[0 for _, _ in gp_dict.items()]
                                      for gene in adata_genes]
        gp_sources_categories_mask = np.asarray(gp_sources_categories_mask,
                                                dtype="int32")
        
        categories_label_encoder = {"fc": 0}

        adata.varm[gp_sources_mask_key] = gp_sources_mask
        adata.varm[gp_targets_mask_key] = gp_targets_mask
        adata.varm[gp_sources_categories_mask_key] = gp_sources_categories_mask
        adata.varm[gp_targets_categories_mask_key] = gp_targets_categories_mask
        adata.uns[sources_categories_label_encoder_key] = (
            categories_label_encoder)
        adata.uns[targets_categories_label_encoder_key] = (
            categories_label_encoder)
        
        # Get index of genes present in the sources and targets mask respectively
        adata.uns[source_genes_idx_key] = np.arange(
            len(adata.varm[gp_sources_mask_key]))
        adata.uns[target_genes_idx_key] = np.arange(
            len(adata.varm[gp_targets_mask_key]))
        adata.uns[genes_idx_key] = np.concatenate(
            (adata.uns[target_genes_idx_key],
             adata.uns[source_genes_idx_key] + adata.n_vars), axis=0)

        # Add gene program names
        adata.uns[gp_names_key] = np.array([f"FC_{i}_GP" for i, (_, _) in 
                                            enumerate(gp_dict.items())])
        return
    
    if genes_uppercase:
        # Convert gene program genes to uppercase
        for _, gp_genes_dict in gp_dict.items():
            gp_genes_dict["sources"] = [
                source.upper() for source in gp_genes_dict["sources"]]
            gp_genes_dict["targets"] = [
                target.upper() for target in gp_genes_dict["targets"]]

    # Create binary gene program gene masks considering only probed genes
    gp_targets_mask = [[int(gene in gp_genes_dict["targets"])
                        for _, gp_genes_dict in gp_dict.items()]
                       for gene in adata_genes]
    gp_targets_mask = np.asarray(gp_targets_mask, dtype="int32")
    gp_sources_mask = [[int(gene in gp_genes_dict["sources"])
                        for _, gp_genes_dict in gp_dict.items()]
                       for gene in adata_genes]
    gp_sources_mask = np.asarray(gp_sources_mask, dtype="int32")
    gp_mask = np.concatenate((gp_sources_mask, gp_targets_mask), axis=0)
    
    # Create gene program gene category masks considering only probed genes
    # First, get unique categories
    sources_categories = []
    targets_categories = []
    for _, gp_genes_dict in gp_dict.items():
        sources_categories.extend(gp_genes_dict["sources_categories"])
        targets_categories.extend(gp_genes_dict["targets_categories"])
    sources_categories = list(set(sources_categories))
    targets_categories = list(set(targets_categories))
    
    # Second, create and store categories label encoders
    sources_categories_label_encoder = {
        k: v for k, v in zip(sources_categories, range(1, len(sources_categories) + 1))}
    targets_categories_label_encoder = {
        k: v for k, v in zip(targets_categories, range(1, len(targets_categories) + 1))}
    adata.uns[sources_categories_label_encoder_key] = sources_categories_label_encoder
    adata.uns[targets_categories_label_encoder_key] = targets_categories_label_encoder

    # Third, create new gp dict with label encoded categories
    category_encoded_gp_dict = copy.deepcopy(gp_dict)
    for _, gp_genes_dict in category_encoded_gp_dict.items():
        gp_genes_dict["targets_categories"] = [
            targets_categories_label_encoder.get(target) for target in
            gp_genes_dict["targets_categories"]]
        gp_genes_dict["sources_categories"] = [
            sources_categories_label_encoder.get(source) for source in
            gp_genes_dict["sources_categories"]]

    # Fourth, use label encoded gp dict to create category masks
    # (encode with category 0 if gene is not in mask)
    gp_targets_categories_mask = [
        [gp_genes_dict["targets_categories"][gp_genes_dict["targets"].index(gene)]
         if gene in gp_genes_dict["targets"] else 0
         for _, gp_genes_dict in category_encoded_gp_dict.items()]
        for gene in adata_genes]
    gp_targets_categories_mask = np.asarray(gp_targets_categories_mask, dtype="int32")

    gp_sources_categories_mask = [
        [gp_genes_dict["sources_categories"][gp_genes_dict["sources"].index(gene)]
         if gene in gp_genes_dict["sources"] else 0
         for _, gp_genes_dict in category_encoded_gp_dict.items()]
        for gene in adata_genes]
    gp_sources_categories_mask = np.asarray(gp_sources_categories_mask,
                                            dtype="int32")

    # Filter gene programs for min genes and max genes
    gp_mask_filter = gp_mask.sum(0) >= min_genes_per_gp
    if max_genes_per_gp is not None:
        gp_mask_filter &= gp_mask.sum(0) <= max_genes_per_gp
    gp_targets_mask_filter = gp_targets_mask.sum(0) >= min_target_genes_per_gp
    if max_target_genes_per_gp is not None:
        gp_targets_mask_filter &= (gp_targets_mask.sum(0)
                                   <= max_target_genes_per_gp)
    gp_sources_mask_filter = gp_sources_mask.sum(0) >= min_source_genes_per_gp
    if max_source_genes_per_gp is not None:
        gp_sources_mask_filter &= (gp_sources_mask.sum(0)
                                   <= max_source_genes_per_gp)
    gp_mask_filter &= gp_sources_mask_filter
    gp_mask_filter &= gp_targets_mask_filter
    gp_targets_mask = gp_targets_mask[:, gp_mask_filter]
    gp_sources_mask = gp_sources_mask[:, gp_mask_filter]
    gp_targets_categories_mask = gp_targets_categories_mask[:, gp_mask_filter]
    gp_sources_categories_mask = gp_sources_categories_mask[:, gp_mask_filter]

    # Add binary gene program gene masks to ´adata.varm´
    adata.varm[gp_sources_mask_key] = gp_sources_mask
    adata.varm[gp_targets_mask_key] = gp_targets_mask
    
    # Add gene program gene category masks to ´adata.varm´
    adata.varm[gp_sources_categories_mask_key] = gp_sources_categories_mask
    adata.varm[gp_targets_categories_mask_key] = gp_targets_categories_mask

    if filter_genes_not_in_masks:
        # Filter out genes not present in any of the masks
        combined_gp_mask = np.maximum(adata.varm["nichecompass_gp_sources"],
                                      adata.varm["nichecompass_gp_targets"])
        adata._inplace_subset_var(combined_gp_mask.sum(axis=1) > 0)

    # Get index of genes present in the sources and targets mask respectively
    adata.uns[source_genes_idx_key] = np.nonzero(
        adata.varm[gp_sources_mask_key].sum(axis=1))[0]
    adata.uns[target_genes_idx_key] = np.nonzero(
        adata.varm[gp_targets_mask_key].sum(axis=1))[0]
    adata.uns[genes_idx_key] = np.concatenate(
        (adata.uns[target_genes_idx_key],
         adata.uns[source_genes_idx_key] + adata.n_vars), axis=0)
         
    # Add gene program names of gene programs that passed filter to adata.uns
    removed_gp_idx = np.where(~gp_mask_filter)[0]
    adata.uns[gp_names_key] = np.array([gp_name for i, (gp_name, _) in 
                                        enumerate(gp_dict.items()) if i not in 
                                        removed_gp_idx])
    
    if plot_gp_gene_count_distributions:
        create_gp_gene_count_distribution_plots(adata=adata,
                                                gp_plot_label="AnnData")


def extract_gp_dict_from_collectri_tf_network(
        species: Literal["mouse", "human"],
        tf_network_file_path: Optional[str]="collectri_tf_network.csv",
        load_from_disk: bool=False,
        save_to_disk: bool=False,
        plot_gp_gene_count_distributions: bool=True,
        gp_gene_count_distributions_save_path: Optional[str]=None) -> dict:
    """
    Retrieve 1072 mouse or 1186 human transcription factor (TF) target gene gene
    programs from CollecTRI via decoupler. CollecTRI is a comprehensive resource
    containing a curated collection of TFs and their transcriptional targets
    compiled from 12 different resources. This collection provides an increased
    coverage of transcription factors and a superior performance in identifying
    perturbed TFs compared to the DoRothEA network and other literature based
    GRNs see
    https://decoupler-py.readthedocs.io/en/latest/notebooks/dorothea.html).

    Parameters
    ----------
    species:
        Species for which the gene programs will be extracted.
    load_from_disk:
        If ´True´, the CollecTRI TF network will be loaded from disk instead of
        from the decoupler library.
    save_to_disk:
        If ´True´, the CollecTRI TF network will additionally be stored on disk.
        Only applies if ´load_from_disk´ is ´False´.
    plot_gp_gene_count_distributions:
        If ´True´, display the distribution of gene programs per number of
        source and target genes.
    gp_gene_count_distributions_save_path:
        Path of the file where the gene program gene count distribution plot
        will be saved if ´plot_gp_gene_count_distributions´ is ´True´.

    Returns
    ----------
    gp_dict:
        Nested dictionary containing the CollecTRI TF target genes gene programs
        with keys being gene program names and values being dictionaries with
        keys ´sources´, ´targets´, ´sources_categories´, and
        ´targets_categories´, where ´sources´ and ´targets´ contain the
        CollecTRI TFs and target genes, and ´sources_categories´ and
        ´targets_categories´ contain the categories of all genes ('tf' or
        'target_gene').
    """
    if not load_from_disk:
        net = dc.op.collectri(organism=species, remove_complexes=False)
        if save_to_disk:
            net.to_csv(tf_network_file_path, index=False)
    else:
        net = pd.read_csv(tf_network_file_path)

    tf_target_genes_df = net[["source", "target"]].groupby(
        "source")["target"].agg(list).reset_index()
    
    gp_dict = {}
    for tf, target_genes in zip(tf_target_genes_df["source"],
                                tf_target_genes_df["target"]):
        gp_dict[tf + "_TF_target_genes_GP"] = {
            "sources": [],
            "targets": [tf] + target_genes,
            "sources_categories": [],
            "targets_categories": ["tf"] + ["target_gene"] * len(target_genes)}
        
    if plot_gp_gene_count_distributions:
        create_gp_gene_count_distribution_plots(
            gp_dict=gp_dict,
            gp_plot_label="CollecTRI",
            save_path=gp_gene_count_distributions_save_path)
        
    return gp_dict


def extract_gp_dict_from_nichenet_lrt_interactions(
        species: Literal["mouse", "human"],
        version: Literal["v1", "v2"]="v2",
        keep_target_genes_ratio: float=1.,
        max_n_target_genes_per_gp: int=250,
        load_from_disk: bool=False,
        save_to_disk: bool=False,
        lr_network_file_path: Optional[str]="nichenet_lr_network.csv",
        ligand_target_matrix_file_path: Optional[str]="../data/gene_programs/" \
                                                      "nichenet_ligand_target" \
                                                      "_matrix.csv",
        gene_orthologs_mapping_file_path: Optional[str]="../data/gene_" \
                                                        "annotations/human_" \
                                                        "mouse_gene_orthologs.csv",
        plot_gp_gene_count_distributions: bool=True,
        gp_gene_count_distributions_save_path: Optional[str]=None) -> dict:
    """
    Retrieve the NicheNet ligand receptor network and ligand target gene
    regulatory potential matrix as described in Browaeys, R., Saelens, W. &
    Saeys, Y. NicheNet: modeling intercellular communication by linking ligands
    to target genes. Nat. Methods 17, 159–162 (2020), and extract 1287 mouse or
    1226 human gene programs of ligands with their corresponding receptors and
    top target genes based on NicheNet regulatory potential scores.

    Parameters
    ----------
    species:
        Species for which the gps will be extracted. The default is human and, 
        if version is 'v1', human genes are mapped to mouse orthologs using a
        mapping file. NicheCompass contains a default mapping file stored under
        "<root>/data/gene_annotations/human_mouse_gene_orthologs.csv", which was
        created with Ensembl BioMart
        (http://www.ensembl.org/info/data/biomart/index.html).
    version:
        Version of the NicheNet ligand receptor network and ligand target gene
        regulatory potential matrix. ´v2´ is an improved version of ´v1´, and
        has separate files for mouse and human.
    keep_target_genes_ratio:
        Ratio of target genes that are kept compared to total target genes. This
        ratio is applied over the entire matrix (not on gene program level), and
        determines the ´all_gps_score_keep_threshold´, which will be used to
        filter target genes according to their regulatory potential scores.
    max_n_target_genes_per_gp:
        Maximum number of target genes per gene program. If a gene program has
        more target genes than ´max_n_target_genes_per_gp´, only the
        ´max_n_target_genes_per_gp´ gene programs with the highest regulatory
        potential scores will be kept. Default value is chosen based on
        MultiNicheNet specification (s. Browaeys, R. et al. MultiNicheNet: a
        flexible framework for differential cell-cell communication analysis
        from multi-sample multi-condition single-cell transcriptomics data.
        bioRxiv (2023) doi:10.1101/2023.06.13.544751).
    load_from_disk:
        If ´True´, the NicheNet files will be loaded from disk instead of the
        web.
    save_to_disk:
        If ´True´, the NicheNet files will additionally be stored on disk.
    lr_network_file_path:
        Path of the file where the NicheNet ligand receptor network will be
        stored (if ´save_to_disk´ is ´True´) or loaded from (if ´load_from_disk´
        is ´True´).
    ligand_target_matrix_file_path:
        Path of the file where the NicheNet ligand target matrix will be stored
        (if ´save_to_disk´ is ´True´) or loaded from (if ´load_from_disk´ is
        ´True´).
    gene_orthologs_mapping_file_path:
        Path of the file where the gene orthologs mapping is stored. Relevant if
        version is ´v1´ and species is ´mouse´.
    plot_gp_gene_count_distributions:
        If ´True´, display the distribution of gene programs per number of
        sources and targets.
    gp_gene_count_distributions_save_path:
        Path of the file where the gene program gene count distribution plot
        will be saved if ´plot_gp_gene_count_distributions´ is ´True´.

    Returns
    ----------
    gp_dict:
        Nested dictionary containing the NicheNet ligand receptor target gene 
        programs with keys being gene program names and values being 
        dictionaries with keys ´sources´, ´targets´, ´sources_categories´, and
        ´target_categories´, where ´sources´ contains the NicheNet ligands,
        ´targets´ contains the NicheNet receptors and target genes,
        ´sources_categories´ contains the categories of the sources, and
        ´target_categories´ contains the categories of the targets.
    """
    # Download (or load) NicheNet ligand receptor network and ligand target
    # matrix and store in df (optionally also on disk)
    if not load_from_disk:
        if version == "v1":
            lr_network_url = "https://zenodo.org/record/3260758/files/" \
                             "lr_network.rds"
            ligand_target_matrix_url = "https://zenodo.org/record/3260758/" \
                                       "files/ligand_target_matrix.rds"
        elif version == "v2" and species == "human":
            lr_network_url = "https://zenodo.org/record/7074291/files/" \
                             "lr_network_human_21122021.rds"
            ligand_target_matrix_url = "https://zenodo.org/record/7074291/" \
                                       "files/ligand_target_matrix_nsga2r_" \
                                       "final.rds"
        elif version == "v2" and species == "mouse":
            lr_network_url = "https://zenodo.org/record/7074291/files/" \
                             "lr_network_mouse_21122021.rds"
            ligand_target_matrix_url = "https://zenodo.org/record/7074291/" \
                                       "files/ligand_target_matrix_nsga2r_" \
                                       "final_mouse.rds"    
        print(f"Downloading NicheNet ligand receptor network '{version}' from "
              "the web...")
        lr_network_df = load_R_file_as_df(
            R_file_path="lr_network.rds",
            url=lr_network_url,
            save_df_to_disk=save_to_disk,
            df_save_path=lr_network_file_path) # multiple rows per ligand
        print(f"Downloading NicheNet ligand target matrix '{version}' from the "
              "web. This might take a while...")
        ligand_target_matrix_df = load_R_file_as_df(
            R_file_path="ligand_target_matrix.rds",
            url=ligand_target_matrix_url,
            save_df_to_disk=save_to_disk,
            df_save_path=ligand_target_matrix_file_path) # one column per ligand
    else:
        lr_network_df = pd.read_csv(lr_network_file_path,
                                    index_col=0) # multiple rows per ligand
        ligand_target_matrix_df = pd.read_csv(ligand_target_matrix_file_path,
                                              index_col=0) # one column per
                                                           # ligand
        
    # Group receptors by ligands to have one row per ligand
    grouped_lr_network_df = lr_network_df.groupby("from")["to"].agg(
        list).reset_index()

    # Filter ligand target matrix based on sorted potential / scores, using
    # ´keep_target_genes_ratio´ (over all gps) and ´max_n_target_genes_per_gp´
    # (over each gp separately). Each ligand (column) will make up one gp.
    # Store filter outputs as a mask dict where for each gp it is indicated
    # which genes are part of it
    per_gp_target_gene_scores = ligand_target_matrix_df.values.copy()
    all_target_gene_scores = np.squeeze(per_gp_target_gene_scores).flatten()
    per_gp_target_gene_scores_sorted = np.flip(
        np.sort(per_gp_target_gene_scores, axis=0), axis=0)
    per_gp_score_keep_threshold = pd.Series(
        per_gp_target_gene_scores_sorted[
            min(max_n_target_genes_per_gp, len(per_gp_target_gene_scores_sorted) - 1), :],
        index=ligand_target_matrix_df.columns)
    all_target_gene_scores.sort()
    all_target_gene_scores_sorted = np.flip(all_target_gene_scores)
    all_gps_score_keep_threshold = all_target_gene_scores_sorted[int(
        (len(all_target_gene_scores_sorted) - 1) * keep_target_genes_ratio)]
    ligand_target_all_gps_score_keep_threshold_mask_df = (
        ligand_target_matrix_df.applymap(
        lambda x: x > all_gps_score_keep_threshold))
    ligand_target_per_gp_score_keep_threshold_mask_df = (
        ligand_target_matrix_df.apply(
        lambda col: col > per_gp_score_keep_threshold[col.name], axis=0))
    ligand_target_combined_keep_threshold_mask_df = (
        ligand_target_all_gps_score_keep_threshold_mask_df &
        ligand_target_per_gp_score_keep_threshold_mask_df)
    
    # Extract ligands to build gene programs, add receptors and target genes,
    # and store in nested dict
    gp_dict = {}
    for ligand, gene_mask in ligand_target_combined_keep_threshold_mask_df.items():
        # Retrieve ligand receptors
        receptors = grouped_lr_network_df[
            grouped_lr_network_df["from"] == ligand]["to"].values[0]
        
        # Build gp dict using ligand in source node and receptors and target
        # genes in target node
        gp_dict[ligand + "_ligand_receptor_target_gene_GP"] = {
            "sources": [ligand],
            "targets": receptors +
                       [target for target, include in
                        gene_mask.items() if include & 
                        (target not in receptors)]} # don't duplicate receptors
        
        # Add source and target categories
        gp_dict[ligand + "_ligand_receptor_target_gene_GP"][
            "sources_categories"] = ["ligand"]
        gp_dict[ligand + "_ligand_receptor_target_gene_GP"][
            "targets_categories"] = (["receptor"] * len(receptors) +
                                     ["target_gene"] * (
            len(gp_dict[ligand + "_ligand_receptor_target_gene_GP"]["targets"]) -
            len(receptors)))
        
    if version == "v1" and species == "mouse":
        # Create mapping df to map from human genes to mouse orthologs
        mapping_df = pd.read_csv(gene_orthologs_mapping_file_path)
        grouped_mapping_df = mapping_df.groupby(
            "Gene name")["Mouse gene name"].agg(list).reset_index()
        
        # Map all genes in the gene programs to their orthologs from the mapping
        # df or capitalize them if no orthologs are found (one human gene can
        # have multiple mouse orthologs)
        for _, gp in gp_dict.items():
            gp["sources"] = [element for nested_list_l1 in [
                list_element for nested_list_l2 in [
                    grouped_mapping_df[
                        grouped_mapping_df["Gene name"] == source][
                            "Mouse gene name"].values.tolist() if
                            source in grouped_mapping_df["Gene name"].values else
                            [[source.capitalize()]] for source in gp["sources"]]
                            for list_element in nested_list_l2]
                            for element in nested_list_l1]
            gp["targets"] = [element for nested_list_l1 in [
                list_element for nested_list_l2 in [
                    grouped_mapping_df[
                        grouped_mapping_df["Gene name"] == target][
                            "Mouse gene name"].values.tolist() if
                            target in grouped_mapping_df["Gene name"].values else
                            [[target.capitalize()]] for target in gp["targets"]]
                            for list_element in nested_list_l2]
                            for element in nested_list_l1]
            gp["sources_categories"] = [element for nested_list_l1 in [
                list_element for nested_list_l2 in [
                    [source_category] * len(grouped_mapping_df[
                        grouped_mapping_df["Gene name"] == source][
                            "Mouse gene name"].values[0])
                            if source in grouped_mapping_df["Gene name"].values else
                            [source_category] for source, source_category in zip(
                                gp["sources"], gp["sources_categories"])]
                                for list_element in nested_list_l2]
                                for element in nested_list_l1]
            gp["targets_categories"] = [element for nested_list_l1 in [
                list_element for nested_list_l2 in [
                    [target_category] * len(grouped_mapping_df[
                        grouped_mapping_df["Gene name"] == target][
                            "Mouse gene name"].values[0])
                            if target in grouped_mapping_df["Gene name"].values else
                            [target_category] for target, target_category in zip(
                                gp["targets"], gp["targets_categories"])]
                                for list_element in nested_list_l2]
                                for element in nested_list_l1]
        
    if plot_gp_gene_count_distributions:
        create_gp_gene_count_distribution_plots(
            gp_dict=gp_dict,
            gp_plot_label=f"NicheNet {version.replace('_', ' ').title()}",
            save_path=gp_gene_count_distributions_save_path)
        
    return gp_dict


def extract_gp_dict_from_omnipath_lr_interactions(
        species: Literal["mouse", "human"],
        min_curation_effort: int=2,
        load_from_disk: bool=False,
        save_to_disk: bool=False,
        lr_network_file_path: Optional[str]="../data/gene_programs/" \
                                            "omnipath_lr_network.csv",
        gene_orthologs_mapping_file_path: Optional[str]="../data/gene_" \
                                                        "annotations/human_" \
                                                        "mouse_gene_orthologs.csv",
        plot_gp_gene_count_distributions: bool=True,
        gp_gene_count_distributions_save_path: Optional[str]=None) -> dict:
    """
    Retrieve 724 human ligand-receptor interactions from OmniPath and extract
    them into a gene program dictionary. OmniPath is a database of molecular
    biology prior knowledge that combines intercellular communication data from
    many different resources (all resources for intercellular communication
    included in OmniPath can be queried via
    ´op.requests.Intercell.resources()´). If ´species´ is ´mouse´, orthologs
    from human interactions are returned.

    Parts of the implementation are inspired by 
    https://workflows.omnipathdb.org/intercell-networks-py.html (01.10.2022).

    Parameters
    ----------
    species:
        Species for which the gene programs will be extracted. The default is
        human. Human genes are mapped to mouse orthologs using a mapping file.
        NicheCompass contains a default mapping file stored under
        "<root>/data/gene_annotations/human_mouse_gene_orthologs.csv", which was
        created with Ensembl BioMart
        (http://www.ensembl.org/info/data/biomart/index.html).
    min_curation_effort: 
        Indicates how many times an interaction has to be described in a 
        paper and mentioned in a database to be included in the retrieval.
    load_from_disk:
        If ´True´, the OmniPath ligand receptor interactions will be loaded from
        disk instead of from the OmniPath library.
    save_to_disk:
        If ´True´, the OmniPath ligand receptor interactions will additionally 
        be stored on disk. Only applies if ´load_from_disk´ is ´False´.
    lr_network_file_path:
        Path of the file where the OmniPath ligand receptor interactions will be
        stored (if ´save_to_disk´ is ´True´) or loaded from (if ´load_from_disk´
        is ´True´).
    gene_orthologs_mapping_file_path:
        Path of the file where the gene orthologs mapping is stored if species
        is ´mouse´.
    plot_gp_gene_count_distributions:
        If ´True´, display the distribution of gene programs per number of
        source and target genes.
    gp_gene_count_distributions_save_path:
        Path of the file where the gene program gene count distribution plot
        will be saved if ´plot_gp_gene_count_distributions´ is ´True´.

    Returns
    ----------
    gp_dict:
        Nested dictionary containing the OmniPath ligand-receptor interaction
        gene programs with keys being gene program names and values being
        dictionaries with keys ´sources´, ´targets´, ´sources_categories´, and
        ´targets_categories´, where ´sources´ contains the OmniPath ligands,
        ´targets´ contains the OmniPath receptors, ´sources_categories´ contains
        the categories of the sources, and ´targets_categories´ contains
        the categories of the targets.
    """
    if not load_from_disk:
        # Define intercell_network categories to be retrieved (see
        # https://workflows.omnipathdb.org/intercell-networks-py.html,
        # https://omnipath.readthedocs.io/en/latest/api/omnipath.interactions.import_intercell_network.html#omnipath.interactions.import_intercell_network)
        intercell_df = op.interactions.import_intercell_network(
            include=["omnipath", "pathwayextra", "ligrecextra"])
        lr_interaction_df = intercell_df[
            (intercell_df["category_intercell_source"] == "ligand")
            & (intercell_df["category_intercell_target"] == "receptor")]
        if save_to_disk:
            lr_interaction_df.to_csv(lr_network_file_path, index=False)
    else:
        lr_interaction_df = pd.read_csv(lr_network_file_path, index_col=0)

    # Only keep curated interactions (see
    # https://r.omnipathdb.org/reference/filter_intercell_network.html)
    lr_interaction_df = lr_interaction_df[
        lr_interaction_df["curation_effort"] >= min_curation_effort]

    # Group receptors by ligands
    grouped_lr_interaction_df = lr_interaction_df.groupby(
        "genesymbol_intercell_source")["genesymbol_intercell_target"].agg(
            list).reset_index()
    
    # Resolve protein complexes into individual genes
    def compute_elementwise_func(lst, func):
        return [func(item) for item in lst]

    def resolve_protein_complexes(x):
        if x:
            if "COMPLEX:" not in x:
                return [x]
            else:
                return x.removeprefix("COMPLEX:").split("_")
        else:
            return []
        
    grouped_lr_interaction_df["sources"] = grouped_lr_interaction_df[
        "genesymbol_intercell_source"].apply(
            lambda x: list(set(resolve_protein_complexes(x))))
    grouped_lr_interaction_df["sources_categories"] = grouped_lr_interaction_df[
        "sources"].apply(lambda x: ["ligand"] * len(x))
    grouped_lr_interaction_df["targets"] = grouped_lr_interaction_df[
        "genesymbol_intercell_target"].apply(
            lambda x: list(set([element for sublist in compute_elementwise_func(x, resolve_protein_complexes) for element in sublist])))
    grouped_lr_interaction_df["targets_categories"] = grouped_lr_interaction_df[
        "targets"].apply(lambda x: ["receptor"] * len(x))

    # Extract gene programs and store in nested dict
    gp_dict = {}
    for _, row in grouped_lr_interaction_df.iterrows():
        gp_dict[row["genesymbol_intercell_source"] +
                "_ligand_receptor_GP"] = {
                    "sources": row["sources"],
                    "targets": row["targets"],
                    "sources_categories": row["sources_categories"],
                    "targets_categories": row["targets_categories"]}
        
    if species == "mouse":
        # Create mapping df to map from human genes to mouse orthologs
        mapping_df = pd.read_csv(gene_orthologs_mapping_file_path)
        grouped_mapping_df = mapping_df.groupby(
            "Gene name")["Mouse gene name"].agg(list).reset_index()
        
        # Map all genes in the gene programs to their orthologs from the mapping
        # df or capitalize them if no orthologs are found (one human gene can
        # have multiple mouse orthologs)
        for _, gp in gp_dict.items():
            gp["sources"] = [element for nested_list_l1 in [
                list_element for nested_list_l2 in [
                    grouped_mapping_df[
                        grouped_mapping_df["Gene name"] == source][
                            "Mouse gene name"].values.tolist() if
                            source in grouped_mapping_df["Gene name"].values else
                            [[source.capitalize()]] for source in gp["sources"]]
                            for list_element in nested_list_l2]
                            for element in nested_list_l1]
            gp["targets"] = [element for nested_list_l1 in [
                list_element for nested_list_l2 in [
                    grouped_mapping_df[
                        grouped_mapping_df["Gene name"] == target][
                            "Mouse gene name"].values.tolist() if
                            target in grouped_mapping_df["Gene name"].values else
                            [[target.capitalize()]] for target in gp["targets"]]
                            for list_element in nested_list_l2]
                            for element in nested_list_l1]
            gp["sources_categories"] = ["ligand"] * len(gp["sources"])
            gp["targets_categories"] = ["receptor"] * len(gp["targets"])
    
    if plot_gp_gene_count_distributions:
        create_gp_gene_count_distribution_plots(
            gp_dict=gp_dict,
            gp_plot_label="OmniPath",
            save_path=gp_gene_count_distributions_save_path)
        
    return gp_dict


def extract_gp_dict_from_mebocost_ms_interactions(
        species: Literal["mouse", "human"],
        dir_path: str="../data/gene_programs/metabolite_enzyme_sensor_gps",
        plot_gp_gene_count_distributions: bool=True,
        gp_gene_count_distributions_save_path: Optional[str]=None) -> dict:
    """
    Retrieve 115 mouse or 116 human metabolite-sensor interactions based on the
    Human Metabolome Database (HMDB) data curated in Chen, K. et al. MEBOCOST:
    Metabolite-mediated cell communication modeling by single cell
    transcriptome. Research Square (2022) doi:10.21203/rs.3.rs-2092898/v1.
    Gene expression of enzymes involved in reactions with metabolite products is
    used as proxy for metabolite presence.
    
    This data is available in the NicheCompass package under 
    ´..data/gene_programs/metabolite_enzyme_sensor_gps´.

    Parameters
    ----------
    species:
        Species for which to retrieve metabolite-sensor interactions.
    dir_path:
        Path of the directory where the metabolite gene programs are stored.
    plot_gp_gene_count_distributions:
        If ´True´, display the distribution of gene programs per number of
        target and source genes.
    gp_gene_count_distributions_save_path:
        Path of the file where the gene program gene count distribution plot
        will be saved if ´plot_gp_gene_count_distributions´ is ´True´.

    Returns
    ----------
    gp_dict:
        Nested dictionary containing the MEBOCOST metabolite-sensor interaction
        gene programs with keys being gene program names and values being 
        dictionaries with keys ´sources´, ´targets´, ´sources_categories´, and
        ´targets_categories´, where ´sources´ contains the MEBOCOST enzymes,
        ´targets´ contains the MEBOCOST sensors, ´sources_categories´ contains
        the categories of the sources, and ´targets_categories´ contains
        the categories of the targets.
    """
    # Read data from directory
    if species == "human":
        metabolite_enzymes_df = pd.read_csv(
            dir_path + "/human_metabolite_enzymes.tsv", sep="\t")
        metabolite_sensors_df = pd.read_csv(
            dir_path + "/human_metabolite_sensors.tsv", sep="\t")
    elif species == "mouse":
        metabolite_enzymes_df = pd.read_csv(
            dir_path + "/mouse_metabolite_enzymes.tsv", sep="\t")
        metabolite_sensors_df = pd.read_csv(
            dir_path + "/mouse_metabolite_sensors.tsv", sep="\t")
    else:
        raise ValueError("Species should be either human or mouse.")

    # Retrieve metabolite names
    metabolite_names_df = (metabolite_sensors_df[["HMDB_ID",
                                                  "standard_metName"]]
                           .drop_duplicates()
                           .set_index("HMDB_ID"))

    # Keep only enzymes for which the metabolite is the product (filter enzymes
    # for which the metabolite is the substrate)
    metabolite_enzymes_df = metabolite_enzymes_df[
        metabolite_enzymes_df["direction"] == "product"]

    # Retrieve metabolite enzyme and sensor genes
    metabolite_enzymes_unrolled = []
    for _, row in metabolite_enzymes_df.iterrows():
        genes = row["gene"].split("; ")
        for gene in genes:
            tmp = row.copy()
            tmp["gene"] = gene
            metabolite_enzymes_unrolled.append(tmp)
    metabolite_enzymes_df = pd.DataFrame(metabolite_enzymes_unrolled)
    metabolite_enzymes_df["gene_name"] = metabolite_enzymes_df["gene"].apply(
        lambda x: x.split("[")[0])
    metabolite_enzymes_df = (metabolite_enzymes_df.groupby(["HMDB_ID"])
                             .agg({"gene_name": lambda x: sorted(
                                x.unique().tolist())})
                             .rename({"gene_name": "enzyme_genes"}, axis=1)
                             .reset_index()).set_index("HMDB_ID")
    metabolite_sensors_df = (metabolite_sensors_df.groupby(["HMDB_ID"])
                             .agg({"Gene_name": lambda x: sorted(
                                x.unique().tolist())})
                             .rename({"Gene_name": "sensor_genes"}, axis=1)
                             .reset_index()).set_index("HMDB_ID")

    # Combine enzyme and sensor genes based on metabolite names (sensor genes
    # are not available for most metabolites)
    metabolite_df = metabolite_enzymes_df.join(
        other=metabolite_sensors_df,
        how="inner").join(metabolite_names_df).set_index("standard_metName")

    # Convert to gene program dictionary format
    met_interaction_dict = metabolite_df.to_dict()
    gp_dict = {}
    for metabolite, enzyme_genes in met_interaction_dict["enzyme_genes"].items():
        gp_dict[metabolite + "_metabolite_enzyme_sensor_GP"] = {
            "sources": enzyme_genes,
            "sources_categories": ["enzyme"] * len(enzyme_genes)}
    for metabolite, sensor_genes in met_interaction_dict["sensor_genes"].items():
        gp_dict[metabolite + "_metabolite_enzyme_sensor_GP"][
            "targets"] = sensor_genes
        gp_dict[metabolite + "_metabolite_enzyme_sensor_GP"][
            "targets_categories"] = ["sensor"] * len(sensor_genes)

    if plot_gp_gene_count_distributions:
        create_gp_gene_count_distribution_plots(
            gp_dict=gp_dict,
            gp_plot_label="MEBOCOST",
            save_path=gp_gene_count_distributions_save_path)

    return gp_dict


def filter_and_combine_gp_dict_gps(
        gp_dict: dict,
        gp_filter_mode: Optional[Literal["subset", "superset"]]=None,
        combine_overlap_gps: bool=True,
        overlap_thresh_source_genes: float=1.,
        overlap_thresh_target_genes: float=1.,
        overlap_thresh_genes: float=1.,
        verbose: bool=False) -> dict:
    """
    Filter and combine the gene programs in a gene program dictionary based on
    overlapping genes.

    Parameters
    ----------
    gp_dict:
        Nested dictionary containing the gene programs with keys being gene 
        program names and values being dictionaries with keys ´targets´ and 
        ´sources´, where ´targets´ contains a list of the names of genes in the
        gene program for the reconstruction of the gene expression of the node
        itself (receiving node) and ´sources´ contains a list of the names of
        genes in the gene program for the reconstruction of the gene expression
        of the node's neighbors (transmitting nodes).
    gp_filter_mode:
        If `None` (default), do not filter any gene programs. If `subset`, 
        remove gene programs that are subsets of other gene programs from the 
        gene program dictionary. If `superset`, remove gene programs that are 
        supersets of other gene programs instead.
    combine_overlap_gps:
        If `True`, combine gene programs that overlap according to the defined
        thresholds.
    overlap_thresh_source_genes:
        If `combine_overlap_gps` is `True`, the minimum ratio of source 
        genes that need to overlap between two gene programs for them to be 
        combined.
    overlap_thresh_target_genes:
        If `combine_overlap_gps` is `True`, the minimum ratio of target 
        genes that need to overlap between two gene programs for them to be 
        combined.
    overlap_thresh_genes:
        If `combine_overlap_gps` is `True`, the minimum ratio of total genes
        (source genes & target genes) that need to overlap between two gene 
        programs for them to be combined.
    verbose:
        If `True`, print gene programs that are removed and combined.

    Returns
    ----------
    new_gp_dict:
        Modified gene program dictionary with gene programs filtered according 
        to ´gp_filter_mode´ and combined according to ´combine_overlap_gps´,
        ´overlap_thresh_source_genes´, ´overlap_thresh_target_genes´, and 
        ´overlap_thresh_genes´.
    """
    new_gp_dict = gp_dict.copy()

    # Remove gps that are subsets or supersets of other gps from the gp dict
    if gp_filter_mode != None:
        for i, (gp_i, gp_genes_dict_i) in enumerate(gp_dict.items()):
            source_genes_i = set([gene.upper() for gene in 
                                  gp_genes_dict_i["sources"]])
            target_genes_i = set([gene.upper() for gene in 
                                  gp_genes_dict_i["targets"]])
            for j, (gp_j, gp_genes_dict_j) in enumerate(gp_dict.items()):
                if i != j:
                    source_genes_j = set([gene.upper() for gene in 
                                          gp_genes_dict_j["sources"]])
                    target_genes_j = set([gene.upper() for gene in
                                          gp_genes_dict_j["targets"]])
                    if gp_filter_mode == "subset":
                        if (source_genes_j.issubset(source_genes_i) &
                            target_genes_j.issubset(target_genes_i)):
                                new_gp_dict.pop(gp_j, None)
                                if verbose:
                                    print(f"Removing GP '{gp_j}' as it is a "
                                          f"subset of GP '{gp_i}'.")
                    elif gp_filter_mode == "superset":
                        if (source_genes_j.issuperset(source_genes_i) &
                            target_genes_j.issuperset(target_genes_i)):
                                new_gp_dict.pop(gp_j, None)
                                if verbose:
                                    print(f"Removing GP '{gp_j}' as it is a "
                                          f"superset of GP '{gp_i}'.")

    # Combine overlap gps in the gp dict (overlap ratios are calculated 
    # based on average gene numbers of the compared gene programs)
    if combine_overlap_gps:
        # First, get all overlap gps per gene program (this includes
        # duplicate overlaps and unresolved cross overlaps (i.e. GP A might 
        # overlap with GP B and GP B might overlap with GP C while GP A and GP C
        # do not overlap)
        all_overlap_gps = []
        for i, (gp_i, gp_genes_dict_i) in enumerate(new_gp_dict.items()):
            source_genes_i = set([gene.upper() for gene in 
                                  gp_genes_dict_i["sources"]])
            target_genes_i = set([gene.upper() for gene in 
                                  gp_genes_dict_i["targets"]])
            gp_overlap_gps = [gp_i]
            for j, (gp_j, gp_genes_dict_j) in enumerate(new_gp_dict.items()):
                if i != j:
                    source_genes_j = set([gene.upper() for gene in 
                                          gp_genes_dict_j["sources"]])
                    target_genes_j = set([gene.upper() for gene in
                                          gp_genes_dict_j["targets"]])
                    source_genes_overlap = list(source_genes_i & source_genes_j)
                    target_genes_overlap = list(target_genes_i & target_genes_j)
                    n_source_gene_overlap = len(source_genes_overlap)
                    n_target_gene_overlap = len(target_genes_overlap)
                    n_gene_overlap = (n_source_gene_overlap + 
                                      n_target_gene_overlap)
                    n_avg_source_genes = (len(source_genes_i) + 
                                          len(source_genes_j)) / 2
                    n_avg_target_genes = (len(target_genes_i) + 
                                          len(target_genes_j)) / 2
                    n_avg_genes = n_avg_source_genes + n_avg_target_genes
                    if n_avg_source_genes > 0:
                        ratio_shared_source_genes = (n_source_gene_overlap / 
                                                     n_avg_source_genes)
                    else: 
                        ratio_shared_source_genes = 1
                    if n_avg_target_genes > 0:
                        ratio_shared_target_genes = (n_target_gene_overlap /
                                                     n_avg_target_genes)
                    else:
                        ratio_shared_target_genes = 1
                    ratio_shared_genes = n_gene_overlap / n_avg_genes
                    if ((ratio_shared_source_genes >= 
                         overlap_thresh_source_genes) &
                        (ratio_shared_target_genes >= 
                         overlap_thresh_target_genes) &
                        (ratio_shared_genes >= overlap_thresh_genes)):
                            gp_overlap_gps.append(gp_j)
            if len(gp_overlap_gps) > 1:
                all_overlap_gps.append(set(gp_overlap_gps))

        # Second, clean up duplicate overlaps 
        all_unique_overlap_gps = []
        _ = [all_unique_overlap_gps.append(overlap_gp) for overlap_gp in 
             all_overlap_gps if overlap_gp not in all_unique_overlap_gps]

        # Third, split overlaps into no cross and cross overlaps
        no_cross_overlap_gps = []
        cross_overlap_gps = []
        for i, overlap_gp_i in enumerate(all_unique_overlap_gps):
            if all(overlap_gp_j.isdisjoint(overlap_gp_i) for 
            j, overlap_gp_j in enumerate(all_unique_overlap_gps) 
            if i != j):
                no_cross_overlap_gps.append(overlap_gp_i)
            else:
                cross_overlap_gps.append(overlap_gp_i)

        # Fourth, resolve cross overlaps by sequentally combining them (until
        # convergence)
        sequential_overlap_gps = list(cross_overlap_gps)
        while True:
            new_sequential_overlap_gps = []
            for i, overlap_gp_i in enumerate(sequential_overlap_gps):
                paired_overlap_gps = [overlap_gp_i.union(overlap_gp_j) for 
                                      j, overlap_gp_j in 
                                      enumerate(sequential_overlap_gps) 
                                      if (i != j) & 
                                      (overlap_gp_i.intersection(overlap_gp_j) 
                                       != set())]
                paired_overlap_gps_union = set().union(*paired_overlap_gps)
                if (paired_overlap_gps_union != set() &
                paired_overlap_gps_union not in new_sequential_overlap_gps):
                    new_sequential_overlap_gps.append(paired_overlap_gps_union)
            if (sorted([list(gp) for gp in new_sequential_overlap_gps]) == 
            sorted([list(gp) for gp in sequential_overlap_gps])):
                break
            else:
                sequential_overlap_gps = list(new_sequential_overlap_gps)

        # Fifth, add overlap gps to gp dict and remove component gps
        final_overlap_gps = [list(overlap_gp) for overlap_gp in 
                             no_cross_overlap_gps]
        _ = [final_overlap_gps.append(list(overlap_gp)) for overlap_gp in 
             sequential_overlap_gps if list(overlap_gp) not in 
             final_overlap_gps]

        for overlap_gp in final_overlap_gps:
            new_gp_name = "_".join([gp[:-3] for gp in overlap_gp]) + "_GP"
            new_gp_sources = []
            new_gp_targets = []
            new_gp_sources_categories = []
            new_gp_targets_categories = []
            for gp in overlap_gp:
                for new_gp_source, new_gp_source_category in zip(
                    gp_dict[gp]["sources"], gp_dict[gp]["sources_categories"]):
                    if new_gp_source not in new_gp_sources:
                        new_gp_sources.extend(new_gp_source)
                        new_gp_sources_categories.extend(new_gp_source_category)
                for new_gp_target, new_gp_target_category in zip(
                    gp_dict[gp]["targets"], gp_dict[gp]["targets_categories"]):
                    if new_gp_target not in new_gp_targets:
                        new_gp_targets.extend(new_gp_target)
                        new_gp_targets_categories.extend(new_gp_target_category)
                new_gp_dict.pop(gp, None)
                if verbose:
                    print(f"Removing GP '{gp}' as it is a component of the "
                          f"combined GP '{new_gp_name}'.")
            new_gp_dict[new_gp_name] = {"sources": new_gp_sources}
            new_gp_dict[new_gp_name]["targets"] = new_gp_targets
            new_gp_dict[new_gp_name]["sources_categories"] = (
                new_gp_sources_categories)
            new_gp_dict[new_gp_name]["targets_categories"] = (
                new_gp_targets_categories)
    return new_gp_dict


def filter_and_combine_gp_dict_gps_v2(
        gp_dicts: list,
        overlap_thresh_target_genes: float=1.,
        verbose: bool=False) -> dict:
    """
    Combine gene program dictionaries and filter them based on gene overlaps.

    Parameters
    ----------
    gp_dicts:
        List of gene program dictionaries with keys being gene program names and
        values being dictionaries with keys ´sources´, ´targets´,
        ´sources_categories´, and ´targets_categories´, where ´targets´ contains
        a list of the names of genes in the gene program for the reconstruction
        of the gene expression of the node itself (receiving node) and ´sources´
        contains a list of the names of genes in the gene program for the
        reconstruction of the gene expression of the node's neighbors
        (transmitting nodes).
    overlap_thresh_target_genes:
        The minimum ratio of target genes that need to overlap between a GP
        without source genes and another GP for the GP to be dropped.
        Gene programs with different source genes are never combined or dropped.
    verbose:
        If `True`, print gene programs that are dropped and combined.

    Returns
    ----------
    new_gp_dict:
        Combined gene program dictionary with filtered gene programs.
    """
    # Combine gene program dictionaries
    combined_gp_dict = {}
    for i, gp_dict in enumerate(gp_dicts):
        combined_gp_dict.update(gp_dict)

    new_gp_dict = combined_gp_dict.copy()

    # Combine gene programs with overlapping genes
    all_combined = False
    while not all_combined:
        all_combined = True
        combined_gp_dict = new_gp_dict.copy()
        for i, (gp_i, gp_genes_dict_i) in enumerate(combined_gp_dict.items()):
            source_genes_i = [
                gene for gene in gp_genes_dict_i["sources"]]
            target_genes_i = [
                gene for gene in gp_genes_dict_i["targets"]]
            target_genes_categories_i = [
                target_gene_category for target_gene_category in
                gp_genes_dict_i["targets_categories"]]
            for j, (gp_j, gp_genes_dict_j) in enumerate(
                combined_gp_dict.items()):
                if j != i:
                    source_genes_j = [
                        gene for gene in gp_genes_dict_j["sources"]]
                    target_genes_j = [
                        gene for gene in gp_genes_dict_j["targets"]]
                    target_genes_categories_j = [
                        target_gene_category for target_gene_category in
                        gp_genes_dict_j["targets_categories"]]

                    if ((source_genes_i == source_genes_j) &
                        len(source_genes_i) > 0):
                        # if source genes are exactly the same, combine gene
                        # programs
                        all_combined = False
                        if verbose:
                            print(f"Combining {gp_i} and {gp_j}.")
                        source_genes = source_genes_i
                        target_genes = target_genes_i
                        target_genes_categories = target_genes_categories_i
                        for target_gene, target_gene_category in zip(
                            target_genes_j, target_genes_categories_j):
                            if target_gene not in target_genes:
                                target_genes.extend([target_gene])
                                target_genes_categories.extend(
                                    [target_gene_category])
                        new_gp_dict.pop(gp_i, None)
                        new_gp_dict.pop(gp_j, None)
                        if (gp_j.split("_")[0] + 
                            "_combined_GP") not in new_gp_dict.keys():
                            new_gp_name = gp_i.split("_")[0] + "_combined_GP"
                            new_gp_dict[new_gp_name] = {"sources": source_genes}
                            new_gp_dict[new_gp_name]["targets"] = target_genes
                            new_gp_dict[new_gp_name][
                                "sources_categories"] = gp_genes_dict_i[
                                    "sources_categories"]
                            new_gp_dict[new_gp_name][
                                "targets_categories"] = target_genes_categories
                            
                    elif len(source_genes_i) == 0:
                        target_genes_overlap = list(
                            set(target_genes_i) & set(target_genes_j))
                        n_target_gene_overlap = len(target_genes_overlap)
                        n_target_genes = len(target_genes_i)
                        ratio_shared_target_genes = (n_target_gene_overlap /
                                                     n_target_genes)
                        if ratio_shared_target_genes >= overlap_thresh_target_genes:
                            # if source genes not existent and target genes
                            # overlap more than specified, drop gene program
                            if gp_j in new_gp_dict.keys():
                                if verbose:
                                    print(f"Dropping {gp_i}.")
                                new_gp_dict.pop(gp_i, None)
                    else:
                        # otherwise do not combine or drop gene programs
                        pass

    return new_gp_dict


def get_unique_genes_from_gp_dict(
        gp_dict: dict,
        retrieved_gene_entities: list=["sources", "targets"],
        retrieved_gene_categories: Optional[list]=None) -> list:
    """
    Return all unique genes of a gene program dictionary.

    Parameters
    ----------
    gp_dict:
        The gene program dictionary from which to retrieve the unique genes.
    retrieved_gene_entities:
        A list that contains all gene entities ("sources", "targets")
        for which unique genes of the gene program dictionary should be
        retrieved.
    retrieved_gene_categories:
        A list that contains all gene categories for which unique genes of the
        gene program dictionary should be retrieved. If `None`, all gene
        categories are included.

    Returns
    ----------
    unique_genes:
        A list of unique genes used in the gene program dictionary.
    """
    gene_list = []

    for _, gp in gp_dict.items():
        for gene_entity in retrieved_gene_entities:
            genes = gp[gene_entity]
            gene_categories = gp[f"{gene_entity}_categories"]
            if retrieved_gene_categories is not None:
                genes = [gene for gene, gene_category in zip(genes, gene_categories) if
                         gene_category in retrieved_gene_categories]
            gene_list.extend(genes)
    unique_genes = list(set(gene_list))
    unique_genes.sort()
    return unique_genes