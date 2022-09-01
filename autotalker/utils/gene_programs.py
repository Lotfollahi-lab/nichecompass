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


def mask_adata_with_gp_dict(adata: AnnData,
                            gp_dict: dict,
                            min_genes=0,
                            max_genes=None,
                            varm_key="autotalker_gps",
                            uns_key="gene_programs",
                            clean=True,
                            genes_uppercase=True):
    """
    Add a gene program mask (dictionary) to an AnnData object. Adapted from
    https://github.com/theislab/scarches/blob/master/scarches/utils/annotations.py#L5.
    """

    adata_genes = adata.var_names.str.upper() if genes_uppercase else adata.var_names

    # Create mask of gene programs 
    I = [[int(gene in gp) for _, gp in gp_dict.items()] for gene in adata_genes]
    I = np.asarray(I, dtype="int32")

    mask = I.sum(0) > min_genes
    if max_genes is not None:
        mask &= I.sum(0) < max_genes
    I = I[:, mask]
    adata.varm[varm_key] = I
    adata.uns[uns_key] = [gp[0] for i, (gp_name, gp) in enumerate(gp_dict.items()) if i not in np.where(~mask)[0]]