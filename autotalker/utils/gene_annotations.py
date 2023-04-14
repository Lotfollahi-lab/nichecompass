"""
This module contains utiilities to add gene annotations (gene positions in the
chromosomes) as prior knowledge for use by the Autotalker model.
"""

import os
from typing import Callable, Optional

import numpy as np
import pandas as pd
from anndata import AnnData
from scglue import genomics

def get_gene_annotations(
        adata: AnnData,
        gtf_file_path: Optional[os.PathLike]="gencode.vM32.chr_patch_hapl_scaff.annotation.gtf.gz",
        adata_join_col_name: str=None,
        gtf_join_col_name: Optional[str]="gene_name",
        by_func: Optional[Callable] = None) -> None:
    """
    Get genomic annotations of genes by joining with a GTF file from GENCODE.
    The GFT file is provided but can also be downloaded from 
    https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M32/gencode.vM32.chr_patch_hapl_scaff.annotation.gff3.gz.

    Parts of the implementation are adapted from 
    Cao, Z.-J. & Gao, G. Multi-omics single-cell data integration and regulatory
    inference with graph-linked embedding. Nat. Biotechnol. 40, 1458–1466 (2022)
    (https://github.com/gao-lab/GLUE/blob/master/scglue/data.py#L86); 14.04.23).

    Parameters
    ----------
    adata
        Input dataset
    adata_join_col_name:
        Specify a column in ``adata.var`` used to merge with GTF attributes,
        otherwise ``adata.var_names`` is used by default.
    gtf
        Path to the GTF file
    gtf_join_col_name:
        Specify a field in the GTF attributes used to merge with ``adata.var``,
        e.g. "gene_id", "gene_name".
    by_func
        Specify an element-wise function used to transform merging fields,
        e.g. removing suffix in gene IDs.

    Note
    ----
    The genomic locations are converted to 0-based as specified
    in bed format rather than 1-based as specified in GTF format.
    """
    gene_names = (adata.var_names if adata_join_col_name is None 
                  else adata.var[adata_join_col_name])
    gtf = genomics.read_gtf(
        gtf_file_path).query("feature == 'gene'").split_attribute()
    if by_func:
        by_func = np.vectorize(by_func)
        gene_names = by_func(gene_names)
        gtf[gtf_join_col_name] = by_func(gtf[gtf_join_col_name])
    gtf = gtf.sort_values("seqname").drop_duplicates(
        subset=[gtf_join_col_name], keep="last") # typically scaffolds come 
                                                 # first, chromosomes come last

    merge_df = pd.concat([
        pd.DataFrame(gtf.to_bed(name=gtf_join_col_name)),
        pd.DataFrame(gtf).drop(columns=genomics.Gtf.COLUMNS) # only use splitted attributes
    ], axis=1).set_index(gtf_join_col_name).reindex(
        gene_names).set_index(adata.var.index)

    adata.var = pd.concat([adata.var, merge_df], axis=1)