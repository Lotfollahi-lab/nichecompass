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


def extract_mask_from_ligand_target_mx(path: str, n_genes_per_ligand: int=10):
    """
    
    """
    df = pd.read_csv(path)
    idx_targets_sorted = np.argsort(-df.values, axis=1)[:, :n_genes_per_ligand]
    df_sorted = pd.DataFrame(df.columns.values[idx_targets_sorted],
                             index=df.index + "_ligand")





# add binary I of size n_vars x number of annotated terms in files
# if I[i,j]=1 then gene i is active in annotation j
def add_gp_annotations(adata: ad.AnnData,
                       file_name: str,
                       min_genes=0,
                       max_genes=None,
                       varm_key="I",
                       uns_key="terms",
                       clean=True,
                       genes_use_upper=True):
    """
    Add gene program annotations stored in a csv file to an AnnData object.
    Inspired by https://github.com/theislab/scarches.

    Parameters
    ----------
    adata
        Annotated data matrix.
    files
        Paths to text files with annotations. The function considers rows to be gene sets
        with name of a gene set in the first column followed by names of genes.
    min_genes
        Only include gene sets which have the total number of genes in adata
        greater than this value.
    max_genes
        Only include gene sets which have the total number of genes in adata
        less than this value.
    varm_key
        Store the binary array I of size n_vars x number of annotated terms in files
        in `adata.varm[varm_key]`. if I[i,j]=1 then the gene i is present in the annotation j.
    uns_key
        Store gene sets' names in `adata.uns[uns_key]`.
    clean
        If 'True', removes the word before the first underscore for each term name (like 'REACTOME_')
        and cuts the name to the first thirty symbols.
    genes_use_upper
        if 'True', converts genes' names from files and adata to uppercase for comparison.
    """
    files = [files] if isinstance(files, str) else files
    annot = []

    for file in files:
        with open(file) as f:
            p_f = [l.upper() for l in f] if genes_use_upper else f
            terms = [l.strip('\n').split() for l in p_f]

        if clean:
            terms = [[term[0].split('_', 1)[-1][:30]]+term[1:] for term in terms if term]
        annot+=terms

    var_names = adata.var_names.str.upper() if genes_use_upper else adata.var_names
    I = [[int(gene in term) for term in annot] for gene in var_names]
    I = np.asarray(I, dtype='int32')

    mask = I.sum(0) > min_genes
    if max_genes is not None:
        mask &= I.sum(0) < max_genes
    I = I[:, mask]
    adata.varm[varm_key] = I
    adata.uns[uns_key] = [term[0] for i, term in enumerate(annot) if i not in np.where(~mask)[0]]