import anndata as ad

import torch_geometric
from torch_geometric.data import Data

from ._datasets import SpatialAnnTorchDataset
from ._datasplitters import edge_level_split
from ._datasplitters import node_level_split_mask


def prepare_data(adata: ad.AnnData,
                 adj_key: str="spatial_connectivities",
                 edge_val_ratio: float=0.1,
                 edge_test_ratio: float=0.05,
                 node_val_ratio: float=0.1,
                 node_test_ratio: float=0.0):
    """
    Prepare data for model training including train, validation, test split.

    Parameters
    ----------
    adata:
        AnnData object with sparse adjacency matrix stored in 
        adata.obsp[adj_key].
    adj_key:
        Key under which the sparse adjacency matrix is stored in adata.obsp.
    valid_frac:
        Fraction of the data that is used for validation.
    test_frac:
        Fraction of the data that is used for testing.

    Returns
    ----------
    train_data:
    val_data:
    test_data:
    """
    data_dict = {}
    dataset = SpatialAnnTorchDataset(adata, adj_key=adj_key)
    # PyG Data object (has 2 edge index pairs for one edge because of symmetry)
    data = Data(x=dataset.x,
                edge_index=dataset.edge_index)

    # Edge level split for edge reconstruction
    edge_train_data, edge_val_data, edge_test_data = edge_level_split(
        data=data,
        val_ratio=edge_val_ratio,
        test_ratio=edge_test_ratio)
    data_dict["edge_train_data"] = edge_train_data
    data_dict["edge_val_data"] = edge_val_data
    data_dict["edge_test_data"] = edge_test_data
    
    # Node level split for gene expression reconstruction (adds train, val and 
    # test masks to the PyG Data object)
    data_dict["node_masked_data"] = node_level_split_mask(
        data=data,
        val_ratio=node_val_ratio,
        test_ratio=node_test_ratio)

    return data_dict
