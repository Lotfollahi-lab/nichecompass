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
        adata_atac: Optional[AnnData]=None,
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

    if adata_atac is not None:
        split = adata_atac.var_names.str.split(r"[:-]")
        adata_atac.var["chrom"] = split.map(lambda x: x[0])
        adata_atac.var["chromStart"] = split.map(lambda x: x[1]).astype(int)
        adata_atac.var["chromEnd"] = split.map(lambda x: x[2]).astype(int)

def generate_multimodal_pairing_graph(
    adata_rna: AnnData, 
    adata_atac: AnnData,
    gene_region: str = "combined",
    promoter_len: int = 2000,
    extend_range: int = 0,
    extend_fn: Callable[[int], float] = dist_power_decay,
    signs: Optional[List[int]] = None) -> nx.MultiDiGraph:
    r"""
    Build guidance graph anchored on RNA genes

    Parameters
    ----------
    rna
        Anchor RNA dataset
    *others
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
    corrupt_rate
        **CAUTION: DO NOT USE**, only for evaluation purpose
    random_state
        **CAUTION: DO NOT USE**, only for evaluation purpose

    Returns
    -------
    graph
        Prior regulatory graph

    Note
    ----
    In this function, features in the same dataset can only connect to
    anchor genes via the same edge sign. For more flexibility, please
    construct the guidance graph manually.
    """
    signs = signs or [1] * len(adata_atac)
    if len(others) != len(signs):
        raise RuntimeError("Length of ``others`` and ``signs`` must match!")
    if set(signs).difference({-1, 1}):
        raise RuntimeError("``signs`` can only contain {-1, 1}!")

    rna_bed = Bed(rna.var.assign(name=rna.var_names))
    other_beds = [Bed(other.var.assign(name=other.var_names)) for other in others]
    if gene_region == "promoter":
        rna_bed = rna_bed.strand_specific_start_site().expand(promoter_len, 0)
    elif gene_region == "combined":
        rna_bed = rna_bed.expand(promoter_len, 0)
    elif gene_region != "gene_body":
        raise ValueError("Unrecognized `gene_range`!")
    graphs = [window_graph(
        rna_bed, other_bed, window_size=extend_range,
        attr_fn=lambda l, r, d, s=sign: {
            "dist": abs(d), "weight": extend_fn(abs(d)), "sign": s
        }
    ) for other_bed, sign in zip(other_beds, signs)]
    graph = compose_multigraph(*graphs)

    corrupt_num = round(corrupt_rate * graph.number_of_edges())
    if corrupt_num:
        rna_anchored_guidance_graph.logger.warning("Corrupting guidance graph!")
        rs = get_rs(random_state)
        rna_var_names = rna.var_names.tolist()
        other_var_names = reduce(add, [other.var_names.tolist() for other in others])

        corrupt_remove = set(rs.choice(graph.number_of_edges(), corrupt_num, replace=False))
        corrupt_remove = set(edge for i, edge in enumerate(graph.edges) if i in corrupt_remove)
        corrupt_add = []
        while len(corrupt_add) < corrupt_num:
            corrupt_add += [
                (u, v) for u, v in zip(
                    rs.choice(rna_var_names, corrupt_num - len(corrupt_add)),
                    rs.choice(other_var_names, corrupt_num - len(corrupt_add))
                ) if not graph.has_edge(u, v)
            ]

        graph.add_edges_from([
            (add[0], add[1], graph.edges[remove])
            for add, remove in zip(corrupt_add, corrupt_remove)
        ])
        graph.remove_edges_from(corrupt_remove)

    rgraph = graph.reverse()
    nx.set_edge_attributes(graph, "fwd", name="type")
    nx.set_edge_attributes(rgraph, "rev", name="type")
    graph = compose_multigraph(graph, rgraph)
    all_features = set(chain.from_iterable(
        map(lambda x: x.var_names, [rna, *others])
    ))
    for item in all_features:
        graph.add_edge(item, item, weight=1.0, sign=1, type="loop")
    return graph