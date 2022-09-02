from typing import Optional

import numpy as np
import pandas as pd
from anndata import AnnData

from ._utils import _load_R_file_as_df


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