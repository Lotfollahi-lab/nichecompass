from typing import Literal, Optional

import numpy as np
import omnipath as op
import pandas as pd
from anndata import AnnData

from .utils import _load_R_file_as_df


def add_gps_from_gp_dict_to_adata(
        adata: AnnData,
        gp_dict: dict,
        genes_uppercase: bool=True,
        gp_targets_varm_key: str="autotalker_gp_targets",
        gp_sources_varm_key: str="autotalker_gp_sources",
        gp_names_uns_key: str="autotalker_gp_names",
        min_genes_per_gp: int=0,
        max_genes_per_gp: Optional[int]=None):
    """
    Add gene programs defined in a gene program dictionary to an AnnData object
    by converting the gene program lists of gene program target and source genes
    to binary masks and aligning the masks with genes for which gene expression
    is available in the AnnData object. Inspired by
    https://github.com/theislab/scarches/blob/master/scarches/utils/annotations.py#L5.

    Parameters
    ----------
    adata:
        AnnData object to which the gene programs will be added.
    gp_dict:
        Nested dictionary containing the gene programs with keys being gene 
        program names and values being dictionaries with keys ´targets´ and 
        ´sources´, where ´targets´ contains a list of the names of genes in the
        gene program for the reconstruction of the gene expression of the node
        itself (receiving node) and ´sources´ contains a list of the names of
        genes in the gene program for the reconstruction of the gene expression
        of the node's neighbors (transmitting nodes).
    genes_uppercase:
        If `True`, convert the gene names in adata to uppercase for comparison
        with the gene program dictionary (e.g. if adata contains mouse data).
    gp_targets_varm_key:
        Key in adata.varm where the binary gene program mask for target genes
        of a gene program will be stored (target genes are used for the 
        reconstruction of the gene expression of the node itself (receiving node
        )).
    gp_sources_varm_key:
        Key in adata.varm where the binary gene program mask for source genes
        of a gene program will be stored (source genes are used for the 
        reconstruction of the gene expression of the node'sneighbors 
        (transmitting nodes).
    gp_names_uns_key:
        Key in adata.uns where the gene program names will be stored.
    min_genes_per_gp:
        Minimum number of genes in a gene program inluding both target and 
        source genes that need to be available in the adata (gene expression has
        been probed) for a gene program not to be discarded.
    max_genes_per_gp:
        Maximum number of genes in a gene program including both target and 
        source genes that can be available in the adata (gene expression has 
        been probed) for a gene program not to be discarded.
    """
    # Retrieve probed genes from adata
    adata_genes = (adata.var_names.str.upper() if genes_uppercase 
                                               else adata.var_names)

    # Create binary gene program masks considering only probed genes
    gp_targets_mask = [[int(gene in gp_genes_dict["targets"]) 
               for _, gp_genes_dict in gp_dict.items()]
               for gene in adata_genes]
    gp_targets_mask = np.asarray(gp_targets_mask, dtype="int32")

    gp_sources_mask = [[int(gene in gp_genes_dict["sources"]) 
               for _, gp_genes_dict in gp_dict.items()]
               for gene in adata_genes]
    gp_sources_mask = np.asarray(gp_sources_mask, dtype="int32")
    
    gp_mask = np.concatenate((gp_sources_mask, gp_targets_mask), axis=0)

    # Filter gene programs
    gp_mask_filter = gp_mask.sum(0) > min_genes_per_gp
    if max_genes_per_gp is not None:
        gp_mask_filter &= gp_mask.sum(0) < max_genes_per_gp
    gp_targets_mask = gp_targets_mask[:, gp_mask_filter]
    gp_sources_mask = gp_sources_mask[:, gp_mask_filter]

    # Add binary gene program masks to adata.varm
    adata.varm[gp_sources_varm_key] = gp_sources_mask
    adata.varm[gp_targets_varm_key] = gp_targets_mask

    # Add gene program names of gene programs that passed filter to adata.uns
    removed_gp_idx = np.where(~gp_mask_filter)[0]
    adata.uns[gp_names_uns_key] = [gp_name for i, (gp_name, _) in 
                          enumerate(gp_dict.items()) if i not in removed_gp_idx]


def extract_gp_dict_from_nichenet_ligand_target_mx(
        keep_target_ratio: float=0.1,
        load_from_disk: bool=False,
        save_to_disk: bool=False,
        file_path: Optional[str]="nichenet_ligand_target_matrix.csv"):
    """
    Retrieve NicheNet ligand target matrix as described in Browaeys, R., 
    Saelens, W. & Saeys, Y. NicheNet: modeling intercellular communication by 
    linking ligands to target genes. Nat. Methods 17, 159–162 (2020) and extract
    a gene program dictionary from the matrix based on ´keep_target_ratio´.

    Parameters
    ----------
    keep_target_ratio:
        Ratio of target genes that are kept compared to total target genes. This
        ratio determines the ´score_keep_threshold´, which will be used to
        filter target genes according to their scores.
    load_from_disk:
        If ´True´, the NicheNet ligand target matrix will be loaded from disk
        instead of from the web.
    save_to_disk:
        If ´True´, the NicheNet ligand target matrix will additionally be stored
        on disk.
    file_path:
        Path of the file where the NicheNet ligand target matrix will be stored
        (if ´save_to_disk´ is ´True´) or is loaded from (if ´load_from_disk´ is
        ´True´).

    Returns
    ----------
    gp_dict:
        Nested dictionary containing the NicheNet ligand target genes gene 
        programs with keys being gene program names and values being 
        dictionaries with keys ´targets´ and ´sources´, where ´targets´ contains
        the NicheNet target genes and ´sources´ contains the NicheNet ligands.
    """
    # Download or load NicheNet ligand target matrix and store in df (optionally
    # also on disk)
    if not load_from_disk:
        print("Downloading NicheNet ligand target matrix from the web. This "
              "might take a while...")
        ligand_target_df = _load_R_file_as_df(
            R_file_path="ligand_target_matrix.rds",
            url="https://zenodo.org/record/3260758/files/ligand_target_matrix.rds",
            save_df_to_disk=save_to_disk,
            df_save_path=file_path)
    else:
        ligand_target_df = pd.read_csv(file_path, index_col=0)

    # Filter NicheNet ligand target matrix based on scores and 
    # ´keep_target_ratio´
    all_target_gene_scores = np.squeeze(ligand_target_df.values).flatten()
    all_target_gene_scores.sort()
    all_target_gene_scores_sorted = np.flip(all_target_gene_scores)
    score_keep_threshold = all_target_gene_scores_sorted[int(
        len(all_target_gene_scores_sorted) * keep_target_ratio)]
    ligand_target_df = ligand_target_df.applymap(
        lambda x: x > score_keep_threshold)

    # Extract gene programs and store in nested dict
    ligand_target_dict = ligand_target_df.to_dict()
    gp_dict = {}
    for ligand in ligand_target_dict.keys():
        gp_dict[ligand + "_ligand_targetgenes_GP"] = {
            "sources": [ligand],
            "targets": [target for target, include in 
                       ligand_target_dict[ligand].items() if include == True]}
    return gp_dict


def extract_gp_dict_from_omnipath_lr_interactions(
        min_curation_effort: int=0,
        load_from_disk: bool=False,
        save_to_disk: bool=False,
        file_path: Optional[str]="omnipath_lr_interactions.csv"):
    """
    Retrieve ligand-receptor interactions from OmniPath and extract a gene
    program dictionary. OmniPath is a database of molecular biology prior 
    knowledge that combines intercellular communication data from many different
    resources (all resources for intercellular communication included in 
    OmniPath can be queried via ´op.requests.Intercell.resources()´).

    Parameters
    ----------
    min_curation_effort: 
        Indicates how many times an interaction has to be described in a 
        paper and mentioned in a database to be included in the retrieval.
    load_from_disk:
        If ´True´, the OmniPath ligand-receptor interactions will be loaded from
        disk instead of from the omnipath library.
    save_to_disk:
        If ´True´, the OmniPath ligand-receptor interactions will additionally 
        be stored on disk.
    file_path:
        Path of the file where the OmniPath ligand-receptor interactions will be
        stored (if ´save_to_disk´ is ´True´) or is loaded from 
        (if ´load_from_disk´ is ´True´).    

    Returns
    ----------
    gp_dict:
        Nested dictionary containing the OmniPath ligand-receptor interaction
        gene programs with keys being gene program names and values being 
        dictionaries with keys ´targets´ and ´sources´, where ´targets´ contains
        the OmniPath receptors and ´sources´ contains the OmniPath ligands.
    """
    if not load_from_disk:
        # Define intercell_network categories to be retrieved
        intercell_df = op.interactions.import_intercell_network(
            include=['omnipath', 'pathwayextra', 'ligrecextra'])
    
        # Set transmitters to be ligands and receivers to be receptors
        lr_interaction_df = intercell_df[
            (intercell_df["category_intercell_source"] == "ligand") &
            (intercell_df["category_intercell_target"] == "receptor")]

        if save_to_disk:
            lr_interaction_df.to_csv(file_path, index=False)

    else:
        lr_interaction_df = pd.read_csv(file_path, index_col=0)

    # Filter as per ´min_curation_effort´
    lr_interaction_df = lr_interaction_df[
        lr_interaction_df["curation_effort"] >= min_curation_effort]

    lr_interaction_df = lr_interaction_df[
        ["genesymbol_intercell_source", "genesymbol_intercell_target"]]

    lr_interaction_dict = lr_interaction_df.set_index(
        "genesymbol_intercell_source")["genesymbol_intercell_target"].to_dict()

    # Dictionary comprehension to convert dictionary values to lists and split
    # "COMPLEX:receptor1_receptor2" into ["receptor1", "receptor2"]
    lr_interaction_dict = {key: ([value] if "COMPLEX:" not in value 
        else value.removeprefix("COMPLEX:").split("_")) 
        for key, value in lr_interaction_dict.items()}

    # Extract gene programs and store in nested dict
    gp_dict = {}
    for ligand, receptor in lr_interaction_dict.items():
        gp_dict[ligand + "_ligand_receptor_GP"] = {
            "sources": [ligand],
            "targets": receptor}
    return gp_dict


def extract_gp_dict_from_mebocost_es_interactions(
        species: Literal["mouse", "human"]):
    """
    Retrieve metabolite enzyme-sensor interactions from the Human Metabolome
    Database (HMDB) data curated in Chen, K. et al. MEBOCOST: 
    Metabolite-mediated cell communication modeling by single cell transcriptome.
    Research Square (2022) doi:10.21203/rs.3.rs-2092898/v1. 
    This data is available in the Autotalker package under 
    ´datasets/gp_data/metabolite_enzyme_sensor_gps´.

    Parameters
    ----------
    species:
        Species for which to retrieve metabolite enzyme-sensor interactions.

    Returns
    ----------
    gp_dict:
        Nested dictionary containing the MEBOCOST enzyme-sensor interaction
        gene programs with keys being gene program names and values being 
        dictionaries with keys ´targets´ and ´sources´, where ´targets´ contains
        the MEBOCOST sensor genes and ´sources´ contains the MEBOCOST enzyme
        genes.    
    """
    # Read data from directory
    dir_path = "datasets/gp_data/metabolite_enzyme_sensor_gps/"
    if species == "human":
        metabolite_enzymes_df = pd.read_csv(
            dir_path + "human_metabolite_enzymes.tsv", sep="\t")
        metabolite_sensors_df = pd.read_csv(
            dir_path + "human_metabolite_sensors.tsv", sep="\t")
    elif species == "mouse":
        metabolite_enzymes_df = pd.read_csv(
            dir_path + "mouse_metabolite_enzymes.tsv", sep="\t")
        metabolite_sensors_df = pd.read_csv(
            dir_path + "mouse_metabolite_sensors.tsv", sep="\t")
    else:
        raise KeyError("Species should be either human or mouse!")

    # Retrieve metabolite names
    metabolite_names_df = (metabolite_sensors_df[["HMDB_ID", "standard_metName"]]
                          .drop_duplicates()
                          .set_index("HMDB_ID"))

    # Retrieve metabolite enzyme and sensor genes
    metabolite_enzymes_unrolled = []
    for _, line in metabolite_enzymes_df.iterrows():
        genes = line["gene"].split("; ")
        for gene in genes:
            tmp = line.copy()
            tmp["gene"] = gene
            metabolite_enzymes_unrolled.append(tmp)
    metabolite_enzymes_df = pd.DataFrame(metabolite_enzymes_unrolled)
    metabolite_enzymes_df["gene_name"] = metabolite_enzymes_df["gene"].apply(
        lambda x: x.split("[")[0])

    metabolite_enzymes_df = (metabolite_enzymes_df.groupby(["HMDB_ID"])
                             .agg({"gene_name": lambda x: x.tolist()})
                             .rename({"gene_name": "enzyme_genes"}, axis=1)
                             .reset_index()).set_index("HMDB_ID")

    metabolite_sensors_df = (metabolite_sensors_df.groupby(["HMDB_ID"])
                             .agg({"Gene_name": lambda x: x.tolist()})
                             .rename({"Gene_name": "sensor_genes"}, axis=1)
                             .reset_index()).set_index("HMDB_ID")

    # Combine metabolite names and enzyme and sensor genes
    metabolite_interaction_df = metabolite_enzymes_df.join(
        other=metabolite_sensors_df,
        how="inner").join(metabolite_names_df).set_index("standard_metName")

    metabolite_interaction_gp_dict = metabolite_interaction_df.to_dict()

    # Convert to gene program dictionary format
    gp_dict = {}
    for metabolite, enzyme_genes in metabolite_interaction_gp_dict["enzyme_genes"].items():
        gp_dict[metabolite + "_metabolite_enzyme_sensor_GP"] = {
            "sources": enzyme_genes}
    for metabolite, sensor_genes in metabolite_interaction_gp_dict["sensor_genes"].items():
        gp_dict[metabolite + "_metabolite_enzyme_sensor_GP"]["targets"] = sensor_genes

    return gp_dict

