from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

from ._datasets import SpatialAnnDataset


def prepare_data(
    adata,
    adj_key,
    valid_frac: float = 0.1,
    test_frac: float = 0.05):
    """
    Prepare data for model training.

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
        PyTorch Geometric Data object containing the training data.
    valid_data:
        PyTorch Geometric Data object containing the validation data.
    test_data:
        PyTorch Geometric Data object containing the test data.
    """
    dataset = SpatialAnnDataset(adata, adj_key=adj_key)
    data = Data(x=dataset.x, edge_index=dataset.edge_index)

    transform = RandomLinkSplit(
        num_val=valid_frac,
        num_test=test_frac,
        is_undirected=True,
        split_labels=True)

    train_data, valid_data, test_data = transform(data)

    return train_data, valid_data, test_data