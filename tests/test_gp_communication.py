"""Communication preserves independent samples and paired activity/loadings."""

import numpy as np
import pandas as pd
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


def test_weakest_nonzero_pair_survives_normalization(communication_model):
    """The old min-max rescaling subtracted the minimum, so the weakest group
    pair became exactly 0.0 and the positivity filter discarded it on every
    call, however strong it really was."""
    m = communication_model
    result = compute_communication_gp_network(
        ["source_neg"], m, "group", n_neighbors=2, sample_key="sample",
        min_edges=1)
    assert not result.empty
    # Nothing that carries communication is dropped, and nothing that does not
    # is kept.
    assert (result["strength_unscaled"] > 0).all()
    assert (result["strength"] > 0).all()
    # A min-max rescaling would put a zero in ´strength´ whenever more than one
    # pair survived.
    if len(result) > 1:
        assert result["strength"].min() > 0


def test_normalize_modes(communication_model):
    m = communication_model
    kwargs = dict(gp_list=["source_neg"], model=m, group_key="group",
                  n_neighbors=2, sample_key="sample", min_edges=1)
    raw = compute_communication_gp_network(**kwargs, normalize="none")
    per_gp = compute_communication_gp_network(**kwargs, normalize="per_gp")
    glob = compute_communication_gp_network(**kwargs, normalize="global")

    # ´strength_unscaled´ is the same quantity in every mode.
    for other in (per_gp, glob):
        np.testing.assert_allclose(raw["strength_unscaled"].to_numpy(),
                                   other["strength_unscaled"].to_numpy())
    np.testing.assert_allclose(raw["strength"].to_numpy(),
                               raw["strength_unscaled"].to_numpy())
    # Both scaled modes are a pure division, so zero still means zero and the
    # ordering is preserved.
    assert np.isclose(per_gp["strength"].max(), 1.0)
    assert np.isclose(glob["strength"].max(), 1.0)
    for scaled in (per_gp, glob):
        ratio = (scaled["strength"].to_numpy()
                 / scaled["strength_unscaled"].to_numpy())
        np.testing.assert_allclose(ratio, ratio[0])

    with pytest.raises(ValueError, match="normalize"):
        compute_communication_gp_network(**kwargs, normalize="minmax")


def test_store_scores_can_be_turned_off(communication_model):
    m = communication_model
    m.adata.obs.drop(columns=["source_neg_source_score"],
                     errors="ignore", inplace=True)
    compute_communication_gp_network(
        ["source_neg"], m, "group", n_neighbors=2, sample_key="sample",
        store_scores=False)
    assert "source_neg_source_score" not in m.adata.obs
    assert "source_neg_connectivities" not in m.adata.obsp


def test_filter_requires_a_category_and_validates_keys(communication_model):
    m = communication_model
    with pytest.raises(ValueError, match="filter_cat"):
        compute_communication_gp_network(
            ["source_neg"], m, "group", filter_key="sample", n_neighbors=2)
    with pytest.raises(ValueError, match="group_key"):
        compute_communication_gp_network(
            ["source_neg"], m, "not_a_column", n_neighbors=2)
    with pytest.raises(ValueError, match="filter_cat"):
        compute_communication_gp_network(
            ["source_neg"], m, "group", filter_key="sample",
            filter_cat="not_a_sample", n_neighbors=2)


def test_visualization_is_robust_and_its_legend_matches_the_edges(
        communication_model, monkeypatch):
    """The legend used to take its colours from ´set(edge_colors)´, whose
    iteration order is salted per process, and its labels from a reversed
    ´edge_types´, so the two were paired arbitrarily."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nichecompass.utils import visualize_communication_gp_network

    m = communication_model
    network = compute_communication_gp_network(
        ["source_neg"], m, "group", n_neighbors=2, sample_key="sample",
        min_edges=1)
    cats = sorted(network["source"].unique().tolist()
                  + network["target"].unique().tolist())
    cat_colors = {c: "#1f77b4" for c in set(cats)}

    captured = {}
    original = plt.Axes.legend
    def spy(self, handles, labels, *args, **kwargs):
        captured["pairs"] = [(h.get_color(), l)
                             for h, l in zip(handles, labels)]
        captured["n_handles"] = len(handles)
        captured["n_labels"] = len(labels)
        return original(self, handles, labels, *args, **kwargs)
    monkeypatch.setattr(plt.Axes, "legend", spy)

    ax = visualize_communication_gp_network(
        m.adata, network, cat_colors, cat_key="group", show=False)
    assert ax is not None
    # Every legend entry carries the colour its gene program was drawn in.
    assert captured["n_handles"] == captured["n_labels"]
    for colour, label in captured["pairs"]:
        assert label.startswith("source_neg")
        assert colour == matplotlib.colors.to_hex(
            matplotlib.cm.tab20.colors[0]) or colour.startswith("#")
    plt.close("all")

    # More gene programs than palette colours warns instead of raising
    # KeyError. The fallback palette holds 20, so this needs more than 20.
    import pandas as pd
    many = pd.concat(
        [network.assign(edge_type=f"gp{i:02d}") for i in range(23)],
        ignore_index=True)
    with pytest.warns(UserWarning, match="colors"):
        visualize_communication_gp_network(
            m.adata, many, cat_colors, cat_key="group", show=False)
    plt.close("all")

    # A category with no colour is named, rather than surfacing as a
    # matplotlib complaint about the length of its ´c´ argument.
    with pytest.raises(ValueError, match="cat_colors"):
        visualize_communication_gp_network(
            m.adata, network, {}, cat_key="group", show=False)

    with pytest.raises(ValueError, match="empty"):
        visualize_communication_gp_network(
            m.adata, network.iloc[:0], cat_colors, cat_key="group", show=False)


def _gp_column(model, gp):
    return list(model.adata.uns[model.active_gp_names_key_]).index(gp)


def test_default_graph_is_the_training_graph_without_self_edges(communication_model):
    """With ´n_neighbors=None´ the edges are those of the graph the model was
    trained on; self edges carry no communication between cells, and
    ´sample_key´ removes edges between samples."""
    m = communication_model
    result = compute_communication_gp_network(
        ["source_neg"], m, "group", min_edges=1, normalize="none")
    # The fixture graph connects every pair of the eight cells.
    n_edges = result.set_index(["source", "target"])["n_edges"]
    assert n_edges[("a", "a")] == 12 and n_edges[("a", "b")] == 16
    graph = m.adata.obsp["source_neg_connectivities"].toarray()
    expected = np.outer(m.adata.obs.source_neg_source_score,
                        m.adata.obs.source_neg_target_score)
    np.fill_diagonal(expected, 0)
    np.testing.assert_allclose(graph, expected, rtol=1e-6)

    m.adata.obsp[m.adj_key_] = sp.csr_matrix(np.ones((8, 8)))
    with_self_edges = compute_communication_gp_network(
        ["source_neg"], m, "group", min_edges=1, normalize="none")
    pd.testing.assert_frame_equal(result, with_self_edges)

    within = compute_communication_gp_network(
        ["source_neg"], m, "group", sample_key="sample", min_edges=1)
    # Group a holds cells 0-3 of the interleaved samples one, two, one, two.
    assert within.set_index(["source", "target"])["n_edges"][("a", "a")] == 4

    del m.adata.obsp[m.adj_key_]
    with pytest.raises(ValueError, match="n_neighbors"):
        compute_communication_gp_network(["source_neg"], m, "group")


def test_only_the_receivers_activity_enters(communication_model):
    """Scaling a cell's activity changes it as a receiver, never as a sender:
    the sender is scored by its expression of the source genes alone."""
    m = communication_model
    compute_communication_gp_network(["source_neg"], m, "group", min_edges=1)
    sender = m.adata.obs.source_neg_source_score.to_numpy().copy()
    receiver = m.adata.obs.source_neg_target_score.to_numpy().copy()
    assert (sender > 0).any() and (receiver > 0).any()

    m.adata.obsm[m.latent_key_][:, _gp_column(m, "source_neg")] *= 3
    compute_communication_gp_network(["source_neg"], m, "group", min_edges=1)
    np.testing.assert_allclose(m.adata.obs.source_neg_source_score, sender)
    np.testing.assert_allclose(m.adata.obs.source_neg_target_score, 3 * receiver,
                               rtol=1e-6)

    # A receiver whose activity points away from the program receives nothing.
    m.adata.obsm[m.latent_key_][:, _gp_column(m, "source_neg")] *= -1
    compute_communication_gp_network(["source_neg"], m, "group", min_edges=1)
    np.testing.assert_allclose(m.adata.obs.source_neg_source_score, sender)
    assert (m.adata.obs.source_neg_target_score == 0).all()


def test_mean_does_not_grow_with_group_size(communication_model):
    """With the same communication on every edge, the mean is the same for
    every pair of groups however large they are, the enrichment is 1, and only
    the sum follows the number of edges."""
    m = communication_model
    m.adata.X = sp.csr_matrix(np.ones(m.adata.shape, dtype=np.float32))
    m.adata.obsm[m.latent_key_] = -np.ones_like(m.adata.obsm[m.latent_key_])
    m.adata.obs["size_group"] = ["big"] * 6 + ["small"] * 2
    mean = compute_communication_gp_network(
        ["source_neg"], m, "size_group", min_edges=1, normalize="none")
    by_pair = mean.set_index(["source", "target"])
    assert by_pair.loc[("big", "big"), "n_edges"] == 30
    assert by_pair.loc[("small", "small"), "n_edges"] == 2
    np.testing.assert_allclose(by_pair["mean_strength"],
                               by_pair["mean_strength"].iloc[0])
    np.testing.assert_allclose(by_pair["enrichment"], 1.0)
    np.testing.assert_allclose(by_pair["total_strength"],
                               by_pair["mean_strength"] * by_pair["n_edges"])
    np.testing.assert_allclose(by_pair["strength_unscaled"], by_pair["mean_strength"])

    total = compute_communication_gp_network(
        ["source_neg"], m, "size_group", min_edges=1, normalize="none",
        aggregate="sum").set_index(["source", "target"])
    np.testing.assert_allclose(total["strength_unscaled"], total["total_strength"])
    assert total["strength"].idxmax() == ("big", "big")


def test_min_edges_omits_pairs_with_few_contacts(communication_model):
    m = communication_model
    m.adata.obs["size_group"] = ["big"] * 6 + ["small"] * 2
    kept = compute_communication_gp_network(
        ["source_neg"], m, "size_group", min_edges=1)
    pruned = compute_communication_gp_network(
        ["source_neg"], m, "size_group", min_edges=3)
    assert ("small", "small") in set(zip(kept.source, kept.target))
    assert ("small", "small") not in set(zip(pruned.source, pruned.target))
    assert (pruned["n_edges"] >= 3).all()


def test_a_joint_sign_flip_of_a_program_changes_nothing(communication_model):
    """The raw latent of a program can point either way. Flipping the signs of
    its loadings and activities together, as a retrained model might, leaves
    every score unchanged."""
    m = communication_model
    before = compute_communication_gp_network(
        ["source_neg"], m, "group", min_edges=1, normalize="none")
    k = list(m.adata.uns[m.gp_names_key_]).index("source_neg")
    with torch.no_grad():
        for decoder in (m.model.source_rna_decoder, m.model.target_rna_decoder):
            decoder.nb_means_normalized_decoder.masked_l.weight[:, k] *= -1
    m.adata.obsm[m.latent_key_][:, _gp_column(m, "source_neg")] *= -1
    m.prepare_gp_analysis(overwrite=True)
    after = compute_communication_gp_network(
        ["source_neg"], m, "group", min_edges=1, normalize="none")
    pd.testing.assert_frame_equal(before, after, rtol=1e-6)


def test_new_arguments_are_validated(communication_model):
    m = communication_model
    with pytest.raises(ValueError, match="aggregate"):
        compute_communication_gp_network(["source_neg"], m, "group", aggregate="median")
    for bad in (0, 2.5, True):
        with pytest.raises(ValueError, match="min_edges"):
            compute_communication_gp_network(["source_neg"], m, "group", min_edges=bad)


def test_every_program_gets_its_own_arrow(communication_model):
    """A DiGraph built from all programs kept one arrow per pair of groups, so
    a second program on the same pair was silently not drawn."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from nichecompass.utils import visualize_communication_gp_network

    m = communication_model
    one = compute_communication_gp_network(["source_neg"], m, "group", min_edges=1)
    two = pd.concat([one, one.assign(edge_type="copy_GP")], ignore_index=True)
    colors = {"a": "#1f77b4", "b": "#ff7f0e"}
    between = int((one.source != one.target).sum())
    assert between > 0 and (one.source == one.target).any()

    ax = visualize_communication_gp_network(
        m.adata, two, colors, cat_key="group", show=False, show_self_loops=False)
    arrows = [p for p in ax.patches if isinstance(p, FancyArrowPatch)]
    assert len(arrows) == 2 * between
    plt.close("all")


def test_categorical_groups_with_unused_and_missing_labels(communication_model):
    """Niche labels are usually categorical (leiden clusters). Cells without a
    label are left out, and an unused category does not become a group."""
    m = communication_model
    labels = pd.Categorical(["a", "a", "a", None, "b", "b", "b", "b"],
                            categories=["a", "b", "unused"])
    m.adata.obs["niche_cat"] = labels
    result = compute_communication_gp_network(
        ["source_neg"], m, "niche_cat", min_edges=1, normalize="none")
    assert set(result.source) | set(result.target) <= {"a", "b"}
    n_edges = result.set_index(["source", "target"])["n_edges"]
    # Three labelled cells in a, all connected: six directed edges.
    assert n_edges[("a", "a")] == 6
    assert n_edges[("a", "b")] == 12


def test_enrichment_baseline_ignores_the_sender_filter(communication_model):
    """A filtered run is read against the program's typical contact over all
    edges, not against the filtered senders' own mean."""
    m = communication_model
    full = compute_communication_gp_network(
        ["source_neg"], m, "group", min_edges=1, normalize="none")
    filtered = compute_communication_gp_network(
        ["source_neg"], m, "group", filter_key="group", filter_cat="a",
        min_edges=1, normalize="none")
    baseline = (full["total_strength"].sum() / full["n_edges"].sum())
    np.testing.assert_allclose(filtered["enrichment"],
                               filtered["mean_strength"] / baseline, rtol=1e-6)
    assert set(filtered.source) == {"a"}


def test_a_program_with_no_receiver_in_its_direction_warns(communication_model):
    m = communication_model
    k = list(m.adata.uns[m.gp_names_key_]).index("source_neg")
    with torch.no_grad():
        m.model.target_rna_decoder.nb_means_normalized_decoder.masked_l.weight[1, k] = 10
    m.prepare_gp_analysis(overwrite=True)
    with pytest.warns(UserWarning, match="canonical direction"):
        result = compute_communication_gp_network(["source_neg"], m, "group", min_edges=1)
    assert result.empty


def test_plot_reserves_categories_without_missing_labels(communication_model):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from nichecompass.utils import visualize_communication_gp_network

    m = communication_model
    m.adata.obs["niche_cat"] = pd.Categorical(["a", "a", "a", None, "b", "b", "b", "b"])
    network = compute_communication_gp_network(["source_neg"], m, "niche_cat", min_edges=1)
    visualize_communication_gp_network(
        m.adata, network, {"a": "#1f77b4", "b": "#ff7f0e"}, cat_key="niche_cat",
        reserve_all_categories=True, show=False)
    plt.close("all")
    with pytest.raises(ValueError, match="min_edges"):
        visualize_communication_gp_network(
            m.adata, network.iloc[:0], {"a": "#1f77b4"}, cat_key="niche_cat", show=False)
