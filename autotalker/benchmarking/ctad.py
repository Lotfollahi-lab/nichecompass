import numpy as np
import scanpy as sc
import squidpy as sq
from anndata import AnnData


def compute_avg_ctad_metric(
        adata: AnnData,
        cell_type_key: str="celltype_mapped_refined",
        spatial_key: str="spatial",
        latent_rep_key: str="autotalker_latent",
        seed: int=42):
    """
    Compute multiple cell-type affinity distances (ctads) by varying the number
    of neighbors used for neighorhood graph construction from 1 to 10 and return
    the average ctad metric.

    Parameters
    ----------
    adata:
        AnnData object with cell type annotations stored in 
        ´adata.obs[cell_type_key]´, spatial coordinates stored in 
        ´adata.obsm[spatial_key]´ and the latent representation from the model
        stored in adata.obsm[latent_rep_key].
    cell_type_key:
        Key under which the cell type annotations are stored in ´adata.obs´.
    spatial_key:
        Key under which the spatial coordinates are stored in ´adata.obsm´.
    latent_rep_key:
        Key under which the latent representation from the model is stored in 
        ´adata.obsm´.
    seed:
        Random seed to get reproducible results.

    Returns
    ----------
    avg_ctad_metric:
        Average cell-type affinity distance computed over different graphs with
        varying number of neighbors.
    """
    ctad_list = []
    
    for n_neighbors in range(1,10):
        ctad_list.append(compute_cell_type_affinity_distance(
            adata=adata,
            cell_type_key=cell_type_key,
            spatial_key=spatial_key,
            latent_rep_key=latent_rep_key,
            neighborhood_graph_n_neighs=n_neighbors,
            seed=seed))

    avg_ctad_metric = np.mean(ctad_list)
    return avg_ctad_metric


def compute_cell_type_affinity_distance(
        adata: AnnData,
        cell_type_key: str="celltype_mapped_refined",
        spatial_key: str="spatial",
        latent_rep_key: str="autotalker_latent",
        neighborhood_graph_n_neighs: int=6,
        seed: int=42):
    """
    Compute the cell-type affinity distance benchmarking metric as first 
    introduced by Lohoff, T. et al. Integration of spatial and single-cell 
    transcriptomic data elucidates mouse organogenesis. Nat. Biotechnol. 40, 
    74–85 (2022). Note that the used implementation from squidpy slightly 
    deviates from the original method and we construct neighborhood graphs
    using the original spatial coordinates and the latent representation from
    the model respectively.

    Parameters
    ----------
    adata:
        AnnData object with cell type annotations stored in 
        ´adata.obs[cell_type_key]´, spatial coordinates stored in 
        ´adata.obsm[spatial_key]´ and the latent representation from the model
        stored in adata.obsm[latent_rep_key].
    cell_type_key:
        Key under which the cell type annotations are stored in ´adata.obs´.
    spatial_key:
        Key under which the spatial coordinates are stored in ´adata.obsm´.
    latent_rep_key:
        Key under which the latent representation from the model is stored in 
        ´adata.obsm´.
    neighborhood_graph_n_neighs:
        Number of neighbors used for the construction of the neighborhood graphs
        from the spatial coordinates and the latent representation from the 
        model.
    seed:
        Random seed to get reproducible results.

    Returns
    ----------
    cell_type_affinity_distance:
        Matrix distance between the spatial coordinate cell-type affinity matrix
        and the latent representation cell-type affinity matrix as measured by 
        the Frobenius norm of the element-wise matrix differences.
    """
    # Create graph from spatial coordinates 
    sq.gr.spatial_neighbors(adata,
                            spatial_key=spatial_key,
                            coord_type="generic",
                            n_neighs=neighborhood_graph_n_neighs,
                            key_added="ctad_spatial")

    # Calculate cell-type affinity scores for spatial neighbor graph
    spatial_nhood_enrichment_zscore_mx, _ = sq.gr.nhood_enrichment(
        adata,
        cluster_key=cell_type_key,
        connectivity_key="ctad_spatial",
        n_perms=1000,
        seed=seed,
        copy=True,
        show_progress_bar=False)

    # Create graph from latent representation
    sc.pp.neighbors(adata,
                    n_neighbors=neighborhood_graph_n_neighs,
                    use_rep=latent_rep_key,
                    random_state=seed,
                    key_added="latent")

    # Calculate cell type affinity scores for latent neighbor graph
    latent_nhood_enrichment_zscore_mx, _ = sq.gr.nhood_enrichment(
        adata,
        cluster_key=cell_type_key,
        connectivity_key="latent",
        n_perms=1000,
        seed=seed,
        copy=True,
        show_progress_bar=False)

    # Calculate Frobenius norm of matrix differences to quantify distance
    nhood_enrichment_zscore_diff_mx = (latent_nhood_enrichment_zscore_mx -
                                       spatial_nhood_enrichment_zscore_mx)
    cell_type_affinity_distance = np.linalg.norm(nhood_enrichment_zscore_diff_mx,
                                                 ord="fro")
    return cell_type_affinity_distance

    
