"""
This module contains utiilities to add gene annotations (gene positions in the
chromosomes) as prior knowledge for use by the Autotalker model.
"""

import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData
from scglue import genomics

def get_gene_annotations(
        adata: AnnData,
        adata_atac: Optional[AnnData]=None,
        gtf_file_path: Optional[os.PathLike]="../datasets/ga_data/gencode.vM32.chr_patch_hapl_scaff.annotation.gtf.gz",
        adata_join_col_name: str=None,
        gtf_join_col_name: Optional[str]="gene_name",
        by_func: Optional[Callable]=None,
        drop_unannotated_genes: bool=True) -> Tuple[AnnData, AnnData]:
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
    drop_unannotated_genes:
        If `True`, drop genes for which no annotation was found.

    Returns
    ----------
    adata:
        The annotated RNA-seq AnnData object.
    adata_atac:
        The annotated ATAC-seq AnnData object.

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

    if drop_unannotated_genes:
        adata = adata[:, adata.var["chrom"].notnull()]

    if adata_atac is not None:
        split = adata_atac.var_names.str.split(r"[:-]")
        adata_atac.var["chrom"] = split.map(lambda x: x[0])
        adata_atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
        adata_atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)

    return adata, adata_atac

def generate_multimodal_pairing_dict(
        adata_rna: AnnData, 
        adata_atac: AnnData,
        gene_region: str="combined",
        promoter_len: int=2000,
        extend_range: int=0,
        extend_fn: Callable[[int], float]=genomics.dist_power_decay,
        uppercase=True) -> dict:
    """
    Build guidance graph anchored on RNA genes.

    Parts of the implementation are adapted from 
    Cao, Z.-J. & Gao, G. Multi-omics single-cell data integration and regulatory
    inference with graph-linked embedding. Nat. Biotechnol. 40, 1458–1466 (2022)
    (https://github.com/gao-lab/GLUE/blob/master/scglue/data.py#L86); 14.04.23).

    Parameters
    ----------
    adata_rna
        Anchor RNA dataset
    adata_atac:
        
        Other datasets
    gene_region
        Defines the genomic region of genes, must be one of
        ``{"gene_body", "promoter", "combined"}``.
    promoter_len
        Defines the length of gene promoters (bp upstream of TSS)
    extend_range
        Maximal extend distance beyond gene regions
    extend_fn
        Distance-decreasing weight function for the extended regions
        (by default :func:`dist_power_decay`)
    signs
        Sign of edges between RNA genes and features in each ``*others``
        dataset, must have the same length as ``*others``. Signs must be
        one of ``{-1, 1}``. By default, all edges have positive signs of ``1``.
    propagate_highly_variable
        Whether to propagate highly variable genes to other datasets,
        datasets in ``*others`` would be modified in place.

    Returns
    ----------
    multimodal_dict:
        Dictionary that maps genes to other modalities.

    Note
    ----
    In this function, features in the same dataset can only connect to
    anchor genes via the same edge sign. For more flexibility, please
    construct the guidance graph manually.
    """
    signs = [1] * len(adata_atac)
    if len(adata_atac) != len(signs):
        raise RuntimeError("Length of ``others`` and ``signs`` must match!")
    if set(signs).difference({-1, 1}):
        raise RuntimeError("``signs`` can only contain {-1, 1}!")

    rna_bed = genomics.Bed(adata_rna.var.assign(name=adata_rna.var_names))
    atac_bed = genomics.Bed(adata_atac.var.assign(name=adata_atac.var_names))

    if gene_region == "promoter":
        rna_bed = rna_bed.strand_specific_start_site().expand(promoter_len, 0)
    elif gene_region == "combined":
        rna_bed = rna_bed.expand(promoter_len, 0)
    elif gene_region != "gene_body":
        raise ValueError("Unrecognized `gene_range`!")
    graph = genomics.window_graph(
        rna_bed, atac_bed, window_size=extend_range,
        attr_fn=lambda l, r, d, s=signs: {
            "dist": abs(d), "weight": extend_fn(abs(d)), "sign": s
        }
    )
    gene_peak_edges_list = list(graph.edges)

    multimodal_dict = {}
    for gene, peak, _ in gene_peak_edges_list:
        if uppercase:
            gene = gene.upper()
        if gene in multimodal_dict:
            multimodal_dict[gene].append(peak)
        else:
            multimodal_dict[gene] = [peak]
    return multimodal_dict


def add_multimodal_mask_to_adata(
        adata: AnnData,
        adata_atac: AnnData,
        gene_peak_mapping_dict: dict,
        filter_peaks_based_on_genes: bool=True,
        filter_hvg_peaks: bool=False,
        n_hvg_peaks: int=4000,
        condition_key: bool="batch",
        gp_targets_mask_key: str="autotalker_gp_targets",
        gp_sources_mask_key: str="autotalker_gp_sources",
        gp_names_key: str="autotalker_gp_names",
        ca_targets_mask_key: str="autotalker_ca_targets",
        ca_sources_mask_key: str="autotalker_ca_sources",
        source_peaks_idx_key: str="autotalker_source_peaks_idx",
        target_peaks_idx_key: str="autotalker_target_peaks_idx",
        peaks_idx_key: str="autotalker_peaks_idx") -> AnnData:
    """
    Retrieve chromatin accessibility target and source masks from gene program
    masks stored in ´adata´ by mapping the genes from the gene programs to
    the peaks defined in a mapping dictionary. Only consider peaks that are in
    ´adata_atac´ and store the results as sparse boolean matrices in
    ´adata_atac.varm´.

    Parameters
    ----------
    adata:
        AnnData object with gene program masks stored in 
        ´adata.varm[gp_targets_mask_key]´ and ´adata.varm[gp_sources_mask_key]´,
        and gene program names stored in ´adata.uns[gp_names_key]´.
    adata_atac:
        AnnData object to which the chromatin accessibility masks will be added.
    gene_peak_mapping_dict:
        A mapping dictionary with genes as keys and the corresponding list of
        peaks as values.
    filter_peaks_based_on_genes:
        If `True`, filter `adata_atac` to only keep peaks that are mapped to
        genes in `gene_peak_mapping_dict`.
    filter_hvg_peaks:
        If `True`, filter `adata_atac` to only keep the ´n_hvg_peaks´ highly
        variable peaks. Is applied after gene-based peak filter.
    n_hvg_peaks:
        Number of highly variable peaks to be filtered if ´filter_hvg_peaks´ is
        ´True´.
    condition_key:
        Key in ´adata.obs´ where the conditions for highly variable peak
        filtering are stored if ´filter_hvg_peaks´ is ´True´.
    gp_targets_mask_key:
        Key in ´adata.varm´ where the binary gene program mask for target genes
        of a gene program is stored.
    gp_sources_mask_key:
        Key in ´adata.varm´ where the binary gene program mask for source genes
        of a gene program is stored.
    gp_names_key:
        Key in ´adata.uns´ where the gene program names are stored.
    ca_targets_mask_key:
        Key in ´adata_atac.varm´ where the boolean chromatin accessibility
        mask for target peaks of a gene program will be stored.
    ca_sources_mask_key:
        Key in ´adata_atac.varm´ where the boolean chromatin accessibility
        mask for source peaks of a gene program will be stored.
    source_peaks_idx_key:
        Key in ´adata_atac.uns´ where the index of the source peaks that are in
        the chromatin accessibility mask will be stored.
    target_peaks_idx_key:
        Key in ´adata_atac.uns´ where the index of the target peaks that are in
        the chromatin accessibility mask will be stored.
    peaks_idx_key:
        Key in ´adata_atac.uns´ where the index of a concatenated vector of
        target and source peaks that are in the chromatin accessibility masks
        will be stored.

    Returns
    -------
    adata_atac:
        The modified AnnData object with added chromatin accessibility masks.
    """
    if filter_peaks_based_on_genes:
        all_gene_peaks = list(
            set(peak for gene_peaks in gene_peak_mapping_dict.values() for peak
                in gene_peaks))
        adata_atac = adata_atac[:, adata_atac.var_names.isin(all_gene_peaks)]

    if filter_hvg_peaks:
        print("Filtering peaks...")
        print(f"Starting with {len(adata_atac.var_names)} peaks.")
        sc.pp.highly_variable_genes(adata_atac,
                                    n_top_genes=n_hvg_peaks,
                                    flavor="seurat_v3",
                                    batch_key=condition_key)
        print(f"Keeping {len(adata_atac.var_names)} highly variable peaks.")

    # Create mapping dict for computationally efficient mapping of peaks to
    # their index in ´adata_atac.var_names´    
    peak_idx_mapping_dict = {value: index for index, value in 
                             enumerate(adata_atac.var_names)}
    
    for entity in ["targets", "sources"]:
        if entity == "targets":
            gp_mask_key = gp_targets_mask_key
            ca_mask_key = ca_targets_mask_key
        elif entity == "sources":
            gp_mask_key = gp_sources_mask_key
            ca_mask_key = ca_sources_mask_key
 
        # Get all corresponding peaks for each gene in a gene program and remove
        # duplicate peaks
        genes_rep = np.tile(adata.var_names,
                            (adata.varm[gp_mask_key].shape[1], 1)).T
                
        all_gp_genes = [[gene.upper() for gene in gene_list if gene is not None]
                        for gene_list in np.where(
                            adata.varm[gp_mask_key], genes_rep, None).T]
        
        if entity == "targets":
            all_target_gp_peaks = [list(set([peak for gene_peaks in 
                                    [gene_peak_mapping_dict.get(gene, []) for 
                                    gene in genes] for peak in gene_peaks]))
                            for genes in all_gp_genes]
            
            peak_dict = {adata.uns[gp_names_key][gp_idx]:{
                entity: target_gp_peaks} for gp_idx, target_gp_peaks in
                enumerate(all_target_gp_peaks)}

        elif entity == "sources":
            all_source_gp_peaks = [list(set([peak for gene_peaks in 
                                    [gene_peak_mapping_dict.get(gene, []) for 
                                    gene in genes] for peak in gene_peaks]))
                            for genes in all_gp_genes]
            
            for gp_idx, source_gp_peaks in enumerate(all_source_gp_peaks):
                peak_dict[adata.uns[gp_names_key][gp_idx]][entity] = (
                    source_gp_peaks)
    
        # Create binary chromatin accessibility masks and add them to
        # ´adata_atac.varm´
        peak_idx = [peak_idx_mapping_dict[peak] for gp_peak_dict
                    in peak_dict.values() for peak in gp_peak_dict[entity]]
        gp_idx = [gp_idx for gp_idx, gp_peak_dict in
                  enumerate(peak_dict.values()) for _ in
                  range(len(gp_peak_dict[entity]))]

        adata_atac.varm[ca_mask_key] = sp.csr_matrix(
            (np.ones(len(peak_idx), dtype=bool), (peak_idx, gp_idx)),
            shape=(adata_atac.shape[1], adata.varm[gp_mask_key].shape[1]),
            dtype=bool)

    # Get index of peaks present in the sources and targets mask respectively
    # Most peaks will not be present in any mask
    adata_atac.uns[source_peaks_idx_key] = np.nonzero(
        adata_atac.varm[ca_sources_mask_key].sum(axis=1))[0]
    adata_atac.uns[target_peaks_idx_key] = np.nonzero(
        adata_atac.varm[ca_targets_mask_key].sum(axis=1))[0]
    adata_atac.uns[peaks_idx_key] = np.concatenate(
        (adata_atac.uns[source_peaks_idx_key],
         adata_atac.uns[target_peaks_idx_key] + adata_atac.n_vars), axis=0)
    
    return adata_atac