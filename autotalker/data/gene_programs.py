import anndata as ad
import numpy as np
import pandas as pd
import pyreadr

from ._utils import load_R_file_as_df


def download_nichenet_ligand_target_mx(
        save_path: str="ligand_target_matrix.csv"):
    """
        
    """
    url = "https://zenodo.org/record/3260758/files/ligand_target_matrix.rds"
    load_R_file_as_df(
        R_file_path="ligand_target_matrix.rds",
        url="https://zenodo.org/record/3260758/files/ligand_target_matrix.rds",
        save_df_to_disk=True,
        df_save_path=save_path)


def extract_gene_programs_from_ligand_target_mx(path: str,
                                                keep_target_ratio: float=0.1):
    """
    
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


# add binary I of size n_vars x number of annotated terms in files
# if I[i,j]=1 then gene i is active in annotation j
def add_gene_program_node_mask(adata: ad.AnnData,
                                           file_name: str,
                                           min_genes=0,
                                           max_genes=None,
                                           varm_key="I",
                                           uns_key="gene_programs",
                                           clean=True,
                                           genes_uppercase=True):
    """

    """
    gp_dict = extract_gene_programs_from_ligand_target_mx(file_name)s

    var_names = adata.var_names.str.upper() if genes_uppercase else adata.var_names
    I = [[int(gene in term) for term in annot] for gene in var_names]
    I = [[int(sequenced_gene in gene_program_gene) for gene_program in gp_dict.keys()] for sequenced_gene in var_names]
    I = np.asarray(I, dtype='int32')

    mask = I.sum(0) > min_genes
    if max_genes is not None:
        mask &= I.sum(0) < max_genes
    I = I[:, mask]
    adata.varm[varm_key] = I
    adata.uns[uns_key] = [term[0] for i, term in enumerate(annot) if i not in np.where(~mask)[0]]