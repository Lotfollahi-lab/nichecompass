import pytest
import nichecompass.dataset
from numpy.random import default_rng, choice
from numpy import array_equal
import anndata


def test_spatialdata():
    rng = default_rng(42)
    counts = rng.integers(0, 15, size=(10, 100))
    coordinates = rng.uniform(0, 10, size=(10, 2))
    cell_type = choice(["B", "T", "Monocyte"], size=10)
    adata = anndata.AnnData(
        counts,
        obsm={"spatial": coordinates},
        obs={"cell_type": cell_type})
    data_loader = nichecompass.dataset.SpatialData(
        adata,
        spatial_index="spatial",
        label_index="cell_type"
    )
    assert array_equal(data_loader.coordinates(), coordinates)
    assert array_equal(data_loader.features(), counts)
    assert array_equal(data_loader.labels(), cell_type)
