"""Communication preserves independent samples and paired activity/loadings."""

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from nichecompass.utils import compute_communication_gp_network
from nichecompass.utils.analysis import _communication_spatial_graph
from tests.test_gp_analysis import model


@pytest.fixture
def communication_model(model):
    k = list(model.adata.uns[model.gp_names_key_]).index("source_neg")
    with torch.no_grad():
        model.model.target_rna_decoder.nb_means_normalized_decoder.masked_l.weight[1, k] = -10
    active = model.get_active_gps()
    model.adata.uns[model.active_gp_names_key_] = active
    model.adata.obsm[model.latent_key_] = -np.arange(1, 9)[:, None] * np.ones((8, len(active)))
    # Two physically unrelated slides overlap in coordinate space; rows are interleaved.
    model.adata.obs["sample"] = ["one", "two"] * 4
    model.adata.obsm["spatial"] = np.array([
        [0., 0.], [.01, 0.], [100., 0.], [100.01, 0.],
        [200., 0.], [200.01, 0.], [300., 0.], [300.01, 0.]])
    return model


def assert_within_sample(graph, labels):
    edges = graph.tocoo()
    assert graph.nnz > 0
    assert np.all(np.asarray(labels)[edges.row] == np.asarray(labels)[edges.col])


def test_communication_respects_interleaved_samples_and_edge_products(communication_model):
    m = communication_model
    # A historical global graph must not bypass the new sample restriction.
    m.adata.uns["spatial_cci"] = {"params": {"n_neighbors": 2}}
    m.adata.obsp["spatial_cci_connectivities"] = m.adata.obsp[m.adj_key_].copy()
    compute_communication_gp_network(["source_neg"], m, "group", n_neighbors=2, sample_key="sample")
    graph = m.adata.obsp["source_neg_connectivities"]
    assert_within_sample(graph, m.adata.obs["sample"])
    expected = np.outer(m.adata.obs.source_neg_source_score, m.adata.obs.source_neg_target_score)
    expected *= m.adata.obsp["spatial_cci_connectivities"].toarray() > 0
    np.testing.assert_allclose(graph.toarray(), expected)


def test_spatial_cache_invalidates_coordinates_samples_neighbors_and_order(communication_model):
    adata = communication_model.adata
    graph = _communication_spatial_graph(adata, 2, "sample")
    first = adata.uns["spatial_cci"]["input_fingerprint"]
    assert _communication_spatial_graph(adata, 2, "sample") is graph

    adata.obsm["spatial"][0, 0] = 299
    new_graph = _communication_spatial_graph(adata, 2, "sample")
    assert adata.uns["spatial_cci"]["input_fingerprint"] != first
    assert (new_graph != graph).nnz
    assert_within_sample(new_graph, adata.obs["sample"])

    first = adata.uns["spatial_cci"]["input_fingerprint"]
    adata.obs["sample"] = ["one", "one", "two", "two"] * 2
    new_graph = _communication_spatial_graph(adata, 2, "sample")
    assert adata.uns["spatial_cci"]["input_fingerprint"] != first
    assert_within_sample(new_graph, adata.obs["sample"])

    first = adata.uns["spatial_cci"]["input_fingerprint"]
    _communication_spatial_graph(adata, 3, "sample")
    assert adata.uns["spatial_cci"]["input_fingerprint"] != first

    first = adata.uns["spatial_cci"]["input_fingerprint"]
    reordered = adata[[7, 1, 3, 0, 6, 2, 5, 4]].copy()
    new_graph = _communication_spatial_graph(reordered, 3, "sample")
    assert reordered.uns["spatial_cci"]["input_fingerprint"] != first
    assert_within_sample(new_graph, reordered.obs["sample"])


def test_tiny_samples_and_zero_expression(communication_model):
    m = communication_model
    m.adata.obs["sample"] = ["pair", "single_a", "pair", "single_b",
                            "single_c", "single_d", "single_e", "single_f"]
    graph = _communication_spatial_graph(m.adata, 90, "sample")
    assert set(zip(*graph.nonzero())) == {(0, 2), (2, 0)}
    m.adata.X = sp.csr_matrix(m.adata.shape, dtype=np.float32)
    result = compute_communication_gp_network(["source_neg"], m, "group", n_neighbors=90, sample_key="sample")
    assert result.empty
    assert np.isfinite(m.adata.obs.source_neg_source_score).all()
    assert m.adata.obsp["source_neg_connectivities"].nnz == 0
    m.adata.obs["sample"] = [f"single_{i}" for i in range(m.adata.n_obs)]
    assert _communication_spatial_graph(m.adata, 90, "sample").nnz == 0


def test_legacy_graph_without_coordinates_is_not_reused_for_partition(model):
    model.adata.uns["spatial_cci"] = {"params": {"n_neighbors": 3}}
    model.adata.obsp["spatial_cci_connectivities"] = model.adata.obsp[model.adj_key_].copy()
    assert _communication_spatial_graph(model.adata, 3, None).nnz > 0
    with pytest.raises(ValueError, match="spatial coordinates"):
        _communication_spatial_graph(model.adata, 3, "group")


def test_invalid_spatial_inputs_fail_clearly(communication_model):
    adata = communication_model.adata
    for n_neighbors in (1, 2.5, True):
        with pytest.raises(ValueError, match="n_neighbors"):
            _communication_spatial_graph(adata, n_neighbors, "sample")
    with pytest.raises(ValueError, match="sample_key"):
        _communication_spatial_graph(adata, 2, "unknown")
    adata.obs.loc[adata.obs_names[0], "sample"] = None
    with pytest.raises(ValueError, match="missing"):
        _communication_spatial_graph(adata, 2, "sample")
    adata.obsm["spatial"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _communication_spatial_graph(adata, 2, None)
