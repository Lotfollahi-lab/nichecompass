"""
This module contains utilities to add positional bp annotations to genes and
peaks and link genes to peaks based on spatial proximity for creating masks used
by the NicheCompass model.
"""

import os
from typing import Callable, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
from anndata import AnnData


def get_gene_annotations(
        adata: AnnData,
        adata_atac: Optional[AnnData]=None,
        gtf_file_path: Optional[os.PathLike]="../data/gene_annotations/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz",
        adata_join_col_name: Optional[str]=None,
        gtf_join_col_name: Optional[str]="gene_name",
        by_func: Optional[Callable]=None,
        drop_unannotated_genes: bool=True) -> Tuple[AnnData, AnnData]:
    """
    Get genomic annotations including chromosomal bp positions of genes by
    joining with a GTF file from GENCODE. The GFT file is provided but can also
    be downloaded from
    https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M32/gencode.vM32.chr_patch_hapl_scaff.annotation.gff3.gz.
    for example.

    Parts of the implementation are adapted from
    Cao, Z.-J. & Gao, G. Multi-omics single-cell data integration and regulatory
    inference with graph-linked embedding. Nat. Biotechnol. 40, 1458–1466 (2022)
    -> https://github.com/gao-lab/GLUE/blob/master/scglue/data.py#L86; 14.04.23.

    Parameters
    ----------
    adata:
        AnnData rna object for which to get gene annotations.
    adata_join_col_name:
        Column in ´adata.var´ that is used to merge with GTF file. If ´None´,
        ´adata.var_names´ is used.
    gtf_file_path:
        Path to the GTF file used to get gene annotations.
    gtf_join_col_name:
        Column in GTF file that is used to merge with ´adata.var´, e.g.
        ´gene_id´, or ´gene_name´.
    by_func:
        An element-wise function used to transform merging fields, e.g. for
        removing suffix in gene IDs.
    drop_unannotated_genes:
        If ´True´, drop genes for which no annotation was found.

    Returns
    ----------
    adata:
        The annotated AnnData rna object.
    adata_atac:
        The annotated AnnData atac object.

    Note
    ----------
    The genomic locations are converted to 0-based as specified in bed format
    rather than 1-based as specified in GTF format.
    """

    try:
        from scglue import genomics
    except ImportError:
        raise ImportError("optional dependency `scglue` is required for this function")

    gene_names = (adata.var_names if adata_join_col_name is None
                  else adata.var[adata_join_col_name])

    gtf = genomics.read_gtf(
        gtf_file_path).query("feature == 'gene'").split_attribute()

    if by_func:
        by_func = np.vectorize(by_func)
        gene_names = by_func(gene_names)
        gtf[gtf_join_col_name] = by_func(gtf[gtf_join_col_name])

    # Drop duplicates. Typically scaffolds come first, chromosomes come last
    gtf = gtf.sort_values("seqname").drop_duplicates(
        subset=[gtf_join_col_name], keep="last")

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


def generate_multimodal_mapping_dict(
        adata: AnnData,
        adata_atac: AnnData,
        gene_region: Literal["combined", "promoter", "gene_body"]="combined",
        promoter_len: int=2000,
        extend_range: int=0,
        extend_fn: Callable[[int], float]=None,
        uppercase: bool=True) -> dict:
    """
    Build a mapping dict to map peaks to genes based on chromosomal bp position
    overlaps.

    Parts of the implementation are adapted from
    Cao, Z.-J. & Gao, G. Multi-omics single-cell data integration and regulatory
    inference with graph-linked embedding. Nat. Biotechnol. 40, 1458–1466 (2022)
    -> https://github.com/gao-lab/GLUE/blob/master/scglue/genomics.py#L473;
    14.04.23.

    Parameters
    ----------
    adata
        AnnData rna object with genes as features.
    adata_atac:
        AnnData atac object with peaks as features.
    gene_region:
        Defines what should be considered to determine the bp positions for
        genes. Allowed values are ´gene_body´, ´promoter´, or ´combined´.
    promoter_len:
        Defines the length of gene promoters (bp upstream of TSS).
    extend_range
        Maximum extended bidirectional range in bps beyond overlap for peaks and
        genes to be mapped (an edge to be created in the mapping graph).
    extend_fn
        Distance-decreasing weight function for the extended regions (edges will
        have a weight lower than 1, but for now this is ignored).
    uppercase:
        Convert genes to upper case in the returned mapping dict.

    Returns
    ----------
    multimodal_mapping_dict:
        Dictionary that maps genes to atac omics features (peaks).
    """

    try:
        from scglue import genomics
    except ImportError:
        raise ImportError("optional dependency `scglue` is required for this function")

    if extend_fn is None:
        extend_fn = genomics.dist_power_decay

    # Get chromosome start and end bp positions of genes and peak regions
    rna_bed = genomics.Bed(adata.var.assign(name=adata.var_names))
    atac_bed = genomics.Bed(adata_atac.var.assign(name=adata_atac.var_names))

    # Based on specified gene region, modify bp positions
    if gene_region == "promoter":
        # Remove gene body and only keep upstream promoter
        rna_bed = rna_bed.strand_specific_start_site().expand(promoter_len, 0)
    elif gene_region == "combined":
        # Expand gene body by upstream promoter
        rna_bed = rna_bed.expand(promoter_len, 0)
    elif gene_region != "gene_body":
        raise ValueError("Unrecognized ´gene_region´!")

    # Create networkx graph that maps genes and peaks based on overlap in bp
    # positions.
    graph = genomics.window_graph(
        left=rna_bed,
        right=atac_bed,
        window_size=extend_range,
        attr_fn=lambda l, r, d: {"dist": abs(d), "weight": extend_fn(abs(d))})

    # Get edges (all edges will get a weight of 1)
    gene_peak_edges_list = list(graph.edges)

    # Build the mapping dictionary
    multimodal_mapping_dict = {}
    for gene, peak, _ in gene_peak_edges_list:
        if uppercase:
            gene = gene.upper()
        if gene in multimodal_mapping_dict:
            multimodal_mapping_dict[gene].append(peak)
        else:
            multimodal_mapping_dict[gene] = [peak]
    return multimodal_mapping_dict


def add_multimodal_mask_to_adata(
        adata: AnnData,
        adata_atac: AnnData,
        gene_peak_mapping_dict: dict,
        filter_peaks_based_on_genes: bool=True,
        filter_hvg_peaks: bool=False,
        n_hvg_peaks: int=4000,
        batch_key: bool="batch",
        gene_peaks_mask_key: str="nichecompass_gene_peaks",
        gp_targets_mask_key: str="nichecompass_gp_targets",
        gp_sources_mask_key: str="nichecompass_gp_sources",
        gp_names_key: str="nichecompass_gp_names",
        ca_targets_mask_key: str="nichecompass_ca_targets",
        ca_sources_mask_key: str="nichecompass_ca_sources",
        source_peaks_idx_key: str="nichecompass_source_peaks_idx",
        target_peaks_idx_key: str="nichecompass_target_peaks_idx",
        peaks_idx_key: str="nichecompass_peaks_idx") -> Tuple[AnnData, AnnData]:
    """
    Retrieve atac target and source gene program masks from the rna gene program
    masks stored in ´adata´. This is achieved by mapping the genes from the gene
    programs to the peaks defined in the mapping dictionary. Only consider peaks
    that are in ´adata_atac´ and store the results as sparse boolean matrices in
    ´adata_atac.varm´. Also store a gene peak mapping mask in ´adata´.

    Parameters
    ----------
    adata:
        AnnData rna object with rna gene program masks stored in
        ´adata.varm[gp_targets_mask_key]´ and ´adata.varm[gp_sources_mask_key]´,
        and gene program names stored in ´adata.uns[gp_names_key]´.
    adata_atac:
        AnnData atac object to which the atac gene program masks will be added.
    gene_peak_mapping_dict:
        A mapping dictionary with uppercase genes as keys and the corresponding
        list of peaks as values.
    filter_peaks_based_on_genes:
        If ´True´, filter ´adata_atac´ to only keep peaks that are mapped to
        genes in ´gene_peak_mapping_dict´.
    filter_hvg_peaks:
        If ´True´, filter ´adata_atac´ to only keep the ´n_hvg_peaks´ highly
        variable peaks. Is applied after gene-based peak filter.
    n_hvg_peaks:
        Number of highly variable peaks to be filtered if ´filter_hvg_peaks´ is
        ´True´.
    batch_key:
        Key in ´adata.obs´ where the batches for highly variable peak
        filtering are stored if ´filter_hvg_peaks´ is ´True´.
    gene_peaks_mask_key:
        Key in ´adata.varm´ where the binary mapping mask from genes to peaks
        will be stored.
    gp_targets_mask_key:
        Key in ´adata.varm´ where the binary gene program mask for target genes
        of a gene program is stored.
    gp_sources_mask_key:
        Key in ´adata.varm´ where the binary gene program mask for source genes
        of a gene program is stored.
    gp_names_key:
        Key in ´adata.uns´ where the gene program names are stored.
    ca_targets_mask_key:
        Key in ´adata_atac.varm´ where the binary gene program mask for target
        peaks of a gene program will be stored.
    ca_sources_mask_key:
        Key in ´adata_atac.varm´ where the binary gene program mask for source
        peaks of a gene program will be stored.
    source_peaks_idx_key:
        Key in ´adata_atac.uns´ where the index of the source peaks that are in
        the atac source mask will be stored.
    target_peaks_idx_key:
        Key in ´adata_atac.uns´ where the index of the target peaks that are in
        the atac target mask will be stored.
    peaks_idx_key:
        Key in ´adata_atac.uns´ where the index of a concatenated vector of
        target and source peaks that are in the atac masks will be stored.

    Returns
    -------
    adata:
        The modified AnnData rna object with the gene peak mask stored.
    adata_atac:
        The modified AnnData atac object with atac gene program masks stored.
    """

    try:
        from scglue import genomics
    except ImportError:
        raise ImportError("optional dependency `scglue` is required for this function")

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
                                    batch_key=batch_key)
        print(f"Keeping {len(adata_atac.var_names)} highly variable peaks.")

    # Create mapping dict with adata indices instead of gene and peak names
    uppercase_sorted_gene_list = [
        gene.upper() for gene in adata.var_names.tolist()]
    sorted_peak_list = adata_atac.var_names.tolist()
    gene_idx_peak_idx_mapping_dict = {
        uppercase_sorted_gene_list.index(gene): [
            sorted_peak_list.index(peak) for peak in peaks]
        for gene, peaks in gene_peak_mapping_dict.items()}

    # Convert mapping dict into boolean mask and store in sparse format
    gene_peak_mask = np.zeros(
        (len(uppercase_sorted_gene_list), len(sorted_peak_list)),
        dtype=np.int32)
    for gene_idx, peak_idx in gene_idx_peak_idx_mapping_dict.items():
        gene_peak_mask[gene_idx, peak_idx] = 1
    adata.varm[gene_peaks_mask_key] = sp.csr_matrix(gene_peak_mask)

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

        # Retrieve all genes for each gene program
        genes_rep = np.tile(adata.var_names,
                            (adata.varm[gp_mask_key].shape[1], 1)).T
        all_gp_genes = [[gene.upper() for gene in gene_list if gene is not None]
                        for gene_list in np.where(
                            adata.varm[gp_mask_key], genes_rep, None).T]

        if entity == "targets":
            # Retrieve all peaks for each gene program
            all_target_gp_peaks = [list(set([peak for gene_peaks in
                                    [gene_peak_mapping_dict.get(gene, []) for
                                    gene in genes] for peak in gene_peaks]))
                            for genes in all_gp_genes]

            # Create gp peak dict with only target peaks
            peak_dict = {adata.uns[gp_names_key][gp_idx]:{
                entity: target_gp_peaks} for gp_idx, target_gp_peaks in
                enumerate(all_target_gp_peaks)}

        elif entity == "sources":
            all_source_gp_peaks = [list(set([peak for gene_peaks in
                                    [gene_peak_mapping_dict.get(gene, []) for
                                    gene in genes] for peak in gene_peaks]))
                            for genes in all_gp_genes]

            # Add all source peaks to gp peak dict
            for gp_idx, source_gp_peaks in enumerate(all_source_gp_peaks):
                peak_dict[adata.uns[gp_names_key][gp_idx]][entity] = (
                    source_gp_peaks)

        # Create binary atac decoder masks and add them to ´adata_atac.varm´
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
        (adata_atac.uns[target_peaks_idx_key],
         adata_atac.uns[source_peaks_idx_key] + adata_atac.n_vars), axis=0)
    return adata, adata_atac
