from typing import Optional

import numpy as np
import omnipath as op
import pandas as pd
from anndata import AnnData

from ._utils import _load_R_file_as_df


def extract_gps_from_omnipath_lr_interactions(
        min_curation_effort: int=0):
    """
    Retrieve ligand-receptor interactions from OmniPath, a database of molecular
    biology prior knowledge that combines intercellular communication data from
    many different resources (all resources for intercellular communication 
    included in omnipath can be queried via 
    ´op.requests.Intercell.resources()´).

    Parameters
    ----------
    min_curation_effort: 
        Indicates how many times an interaction has to be described in a 
        paper and mentioned in a database to be included in the retrieval.

    Returns
    ----------
    lr_interaction_dict:
        Dictionary containing ligand-receptor interactions.
    """
    # Define intercell_network categories to be retrieved
    intercell_df = op.interactions.import_intercell_network(
        include=['omnipath', 'pathwayextra', 'ligrecextra'])

    # Set transmitters to be ligands and receivers to be receptors
    lr_interaction_df = intercell_df[
        (intercell_df["category_intercell_source"] == "ligand") &
        (intercell_df["category_intercell_target"] == "receptor")]

    # Split COMPLEX receptors


    # Filter as per ´min_curation_effort´
    lr_interaction_df = lr_interaction_df[
        lr_interaction_df["curation_effort"] >= min_curation_effort]
        
    lr_interaction_df = lr_interaction_df[
        ["genesymbol_intercell_source", "genesymbol_intercell_target"]]

    lr_gp_dict = lr_interaction_df.set_index(
        "genesymbol_intercell_source")["genesymbol_intercell_target"].to_dict()

    # Dictionary comprehension to convert dictionary values to lists and split
    # "COMPLEX:receptor1_receptor2" into ["receptor1", "receptor2"]
    lr_gp_dict = {key: ([value] if "COMPLEX:" not in value 
        else value.removeprefix("COMPLEX:").split("_")) 
        for key, value in lr_gp_dict.items()}
    
    return lr_gp_dict




def download_nichenet_ligand_target_mx(
        save_path: str="ligand_target_matrix.csv"):
    """
    Download NicheNet ligand target matrix as described in Browaeys, R., 
    Saelens, W. & Saeys, Y. NicheNet: modeling intercellular communication by 
    linking ligands to target genes. Nat. Methods 17, 159–162 (2020).

    Parameters
    ----------
    save_path:
        Path where to store the ligand target matrix csv file.
    """
    _load_R_file_as_df(
        R_file_path="ligand_target_matrix.rds",
        url="https://zenodo.org/record/3260758/files/ligand_target_matrix.rds",
        save_df_to_disk=True,
        df_save_path=save_path)


def extract_gps_from_ligand_target_mx(path: str,
                                      keep_target_ratio: float=0.1):
    """
    Extract gene programs from a ligand target matrix stored in a csv file.

    Parameters
    ----------
    path:
        Path where the ligand target matrix is stored.
    keep_target_ratio:
        Ration of target genes that are kept compared to total target genes.

    Returns
    ----------
    ligand_target_dict:
        Dictionary containing the ligand target gene programs.
    """
    ligand_target_df = pd.read_csv(path, index_col=0)
    all_target_gene_scores = np.squeeze(ligand_target_df.values).flatten()
    all_target_gene_scores.sort()
    all_target_gene_scores_sorted = np.flip(all_target_gene_scores)
    score_keep_threshold = all_target_gene_scores_sorted[int(
        len(all_target_gene_scores_sorted) * keep_target_ratio)]
    ligand_target_df = ligand_target_df.applymap(
        lambda x: x > score_keep_threshold)
    ligand_target_dict = ligand_target_df.to_dict()
    for ligand in ligand_target_dict.keys():
        ligand_target_dict[ligand] = [target for target, bool_value in 
                                      ligand_target_dict[ligand].items() if 
                                      bool_value == True]
    return ligand_target_dict


def add_binary_gp_mask_to_adata(adata: AnnData,
                                gp_dict: dict,
                                genes_uppercase: bool=True,
                                min_genes_per_gp: int=0,
                                max_genes_per_gp: Optional[int]=None,
                                varm_key: str="autotalker_gps",
                                uns_key: str="gene_programs"):
    """
    Convert a gene program dictionary to a binary gene program mask and add it 
    to an AnnData object. Adapted from
    https://github.com/theislab/scarches/blob/master/scarches/utils/annotations.py#L5.

    Parameters
    ----------
    adata:
        AnnData object to which the gene program mask will be added.
    gp_dict:
        Dictionary containing the gene programs with keys being gene program
        names and values being the names of genes that are part of the gene
        program.
    genes_uppercase:
        If `True` convert the gene names in adata to uppercase for comparison
        with the gene program dictionary (e.g. for mouse data).
    min_genes_per_gp:
        Minimum number of genes in a gene program that need to be available 
        in the adata (gene expression has been probed) for a gene program not
        to be discarded.
    max_genes_per_gp:
        Maximum number of genes in a gene program that can be available 
        in the adata (gene expression has been probed) for a gene program not
        to be discarded.
    varm_key:
        Key in adata.varm where the binary gene program mask will be stored.
    uns_key:
        Key in adata.uns where the gene program names will be stored.
    """
    # Retrieve probed genes from adata
    adata_genes = (adata.var_names.str.upper() if genes_uppercase 
                                               else adata.var_names)

    # Create binary gene program mask considering only probed genes
    gp_mask = [[int(gene in gp_genes) for _, gp_genes in gp_dict.items()]
               for gene in adata_genes]
    gp_mask = np.asarray(gp_mask, dtype="int32")

    # Filter gene programs with less than ´min_genes_per_gp'in ´adata´
    gp_mask_filter = gp_mask.sum(0) > min_genes_per_gp

    # Filter gene programs with more than ´max_genes_per_gp'in ´adata´
    if max_genes_per_gp is not None:
        gp_mask_filter &= gp_mask.sum(0) < max_genes_per_gp

    # Add binary gene program mask to adata.varm
    gp_mask = gp_mask[:, gp_mask_filter]
    adata.varm[varm_key] = gp_mask

    # Add gene program names of gene programs that passed filter to adata.uns
    removed_gp_idx = np.where(~gp_mask_filter)[0]
    adata.uns[uns_key] = [gp_name for i, (gp_name, _) in 
                          enumerate(gp_dict.items()) if i not in removed_gp_idx]