"""Analysis integration tests with actual masked NicheCompass model objects."""
import pickle
from copy import deepcopy

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from nichecompass.models import NicheCompass
from nichecompass.utils import add_gps_from_gp_dict_to_adata


def fixed_posterior(model, means, std):
    def infer(**kwargs):
        _, indices = model._gp_selection(kwargs.get("selected_gps"))
        return means[:, indices], std[:, indices]
    return infer


@pytest.fixture
def model():
    adata = ad.AnnData(np.arange(48, dtype=np.float32).reshape(8, 6) % 7 + 1)
    adata.var_names = ["L", "R", "T", "A", "B", "C"]
    adata.X = sp.csr_matrix(adata.X)
    adata.layers["counts"] = adata.X.copy()
    adata.obsp["spatial_connectivities"] = sp.csr_matrix(np.ones((8, 8)) - np.eye(8))
    adata.obs["group"] = ["a"] * 4 + ["b"] * 4
    definitions = {"source_neg": (["L"], ["R"]),
                   "target_only": ([], ["T"]),
                   "balanced": (["L", "R"], ["T"]),
                   "tiny": (["L"], ["R"]), "zero": (["L"], ["R"])}
    gps = {name: {"sources": sources, "targets": targets,
                  "sources_categories": ["ligand"] * len(sources),
                  "targets_categories": ["receptor"] * len(targets)}
           for name, (sources, targets) in definitions.items()}
    add_gps_from_gp_dict_to_adata(gps, adata)
    m = NicheCompass(adata, n_addon_gp=1, n_hidden_encoder=8,
                     use_cuda_if_available=False)
    m.is_trained_ = True
    m.node_batch_size_ = 8
    m.model.eval()
    with torch.no_grad():
        m.model.running_mean_abs_mu.fill_(1)
        source = m.model.source_rna_decoder.nb_means_normalized_decoder
        target = m.model.target_rna_decoder.nb_means_normalized_decoder
        for layer in (source, target):
            layer.masked_l.weight.zero_()
            layer.addon_l.weight.zero_()
        names = list(m.adata.uns[m.gp_names_key_])
        source.masked_l.weight[0, names.index("source_neg")] = -2
        target.masked_l.weight[1, names.index("source_neg")] = 10
        target.masked_l.weight[2, names.index("target_only")] = -3
        source.masked_l.weight[0, names.index("balanced")] = -1
        source.masked_l.weight[1, names.index("balanced")] = 1
        target.masked_l.weight[2, names.index("balanced")] = -4
        source.masked_l.weight[0, names.index("tiny")] = -1e-5
        target.masked_l.weight[1, names.index("tiny")] = 4
        source.addon_l.weight[3, 0] = -1
        target.addon_l.weight[4, 0] = 4
    return m


def test_paper_policy_and_no_parameter_mutation(model):
    before = deepcopy(model.model.state_dict())
    q = model.prepare_gp_analysis().set_index("gp_name")
    assert q.orientation_sign.to_dict() == {
        "source_neg": -1, "target_only": -1, "balanced": 1,
        "tiny": -1, "zero": 1, "Add-on_0_GP": 1}
    assert q.loc["target_only", "anchor"] == "target_fallback"
    assert q.loc["balanced", "orientation_status"] == "mixed"
    assert q.loc["zero", "orientation_status"] == "zero_support"
    assert q.loc["source_neg", "source_target_disagree"]
    assert q.loc["Add-on_0_GP", "anchor"] == "all_rna"
    pd.testing.assert_frame_equal(model.prepare_gp_analysis(), q.reset_index()[model.prepare_gp_analysis().columns])
    for name, value in before.items():
        torch.testing.assert_close(model.model.state_dict()[name], value, rtol=0, atol=0)


def test_loading_views_and_decoder_invariance(model):
    model.prepare_gp_analysis()
    names = list(model.adata.uns[model.gp_names_key_])
    signs = model._gp_signs(names)
    raw = model.get_gp_data()[1]
    oriented = model.get_gp_data(orientation="canonical")[1]
    np.testing.assert_array_equal(oriented, raw * signs)
    z = torch.randn(8, len(names))
    reflected = deepcopy(model.model)
    for entity in ("source", "target"):
        original_decoder = getattr(model.model, f"{entity}_rna_decoder")
        decoder = getattr(reflected, f"{entity}_rna_decoder")
        with torch.no_grad():
            layer = decoder.nb_means_normalized_decoder
            layer.masked_l.weight *= torch.tensor(signs[:-1])
            layer.addon_l.weight *= torch.tensor(signs[-1:])
        torch.testing.assert_close(original_decoder(z, torch.zeros(8, 1)),
                                   decoder(z * torch.tensor(signs, dtype=z.dtype), torch.zeros(8, 1)))
    torch.testing.assert_close(model.model.graph_decoder(z),
                               model.model.graph_decoder(z * torch.tensor(signs, dtype=z.dtype)))
    table = model.get_gp_feature_table()
    assert np.isfinite(table.importance).all()
    assert (table.loc[table.gp_name == "zero", "importance"] == 0).all()
    assert set(table.arm) == {"positive", "negative", "zero"}
    summary = model.get_gp_summary().set_index("gp_name")
    assert summary.loc["tiny", "gp_source_genes_weights"][0] > 0
    importance = model.compute_gp_gene_importances("source_neg")
    assert importance.loc[importance.gene == "L", "gene_weight"].iloc[0] == 2


def test_real_posterior_and_idempotent_obs_refresh(model):
    names = list(model.adata.uns[model.gp_names_key_])
    raw, std = model.get_gp_activities(names, return_std=True, orientation="raw")
    canonical, canonical_std = model.get_gp_activities(names, return_std=True)
    np.testing.assert_allclose(canonical, raw * model._gp_signs(names))
    np.testing.assert_array_equal(canonical_std, std)
    selected = ["balanced", "source_neg"]
    indices = [names.index(name) for name in selected]
    np.testing.assert_allclose(model.get_gp_activities(selected), canonical[:, indices])
    selected_raw, _ = model.get_latent_representation(
        return_mu_std=True, only_active_gps=False, selected_gps=selected)
    np.testing.assert_allclose(selected_raw, raw[:, indices])
    model.add_active_gp_scores_to_obs()
    expected = model.adata.obs[model.get_active_gps()].copy()
    model.adata.obs[model.get_active_gps()] *= -1
    model.add_active_gp_scores_to_obs()
    pd.testing.assert_frame_equal(model.adata.obs[model.get_active_gps()], expected)
    with pytest.raises(ValueError, match="feature"):
        model.get_gp_activities(adata=model.adata[:, ::-1])
    with pytest.raises(ValueError, match="unique"):
        model.get_gp_activities(["tiny", "tiny"])


def test_differential_probability_direction_and_global_rng(model, monkeypatch):
    # Deterministic posteriors isolate the scientific estimand, including zero variance.
    means = np.zeros((8, 6), dtype=float)
    means[:4] = 2
    monkeypatch.setattr(model, "get_latent_representation", fixed_posterior(model, means, np.zeros_like(means)))
    np.random.seed(12)
    rng_state = np.random.get_state()
    raw = model.run_differential_gp_tests("group", "a", n_sample=19,
                                         orientation="raw", return_all=True).set_index("gene_program")
    result = model.run_differential_gp_tests("group", "a", n_sample=19,
                                            direction="higher", return_all=True).set_index("gene_program")
    for gp in result.index:
        expected = raw.loc[gp, "p_higher"] if model._gp_signs([gp])[0] == 1 else raw.loc[gp, "p_lower"]
        assert result.loc[gp, "p_higher"] == expected
    assert (model.adata.uns["nichecompass_differential_gp_test_results"].direction == "higher").all()
    assert result.loc["source_neg", "direction"] == "lower"
    assert result.loc["source_neg", "mean_difference"] == -2
    assert np.array_equal(rng_state[1], np.random.get_state()[1])
    means[:] = 0
    equal = model.run_differential_gp_tests("group", "a", return_all=True, n_sample=10)
    assert (equal.p_higher == 0.5).all()
    assert (equal.log_bayes_factor == 0).all()


def test_invalid_groups_and_missing_labels(model, monkeypatch):
    monkeypatch.setattr(model, "get_latent_representation", fixed_posterior(model, np.zeros((8, 6)), np.ones((8, 6))))
    for kwargs in ({"n_sample": 0}, {"selected_cats": "absent"},
                   {"selected_cats": "a", "comparison_cats": "a"},
                   {"direction": "enriched"}):
        with pytest.raises(ValueError):
            model.run_differential_gp_tests("group", **kwargs)
    model.adata.obs["group"] = [0, 0, 0, None, 1, 1, 1, None]
    result = model.run_differential_gp_tests("group", 0, return_all=True, n_sample=10)
    assert (result.n_focal == 3).all() and (result.n_comparison == 3).all()
    model.adata.obs["group"] = ["rest"] * 4 + ["b"] * 4
    result = model.run_differential_gp_tests("group", "b", comparison_cats=["rest"], return_all=True, n_sample=10)
    assert (result.n_comparison == 4).all()


def test_stale_decoder_and_frozen_reference_inheritance(model):
    old = model.prepare_gp_analysis()
    with torch.no_grad():
        model.model.source_rna_decoder.nb_means_normalized_decoder.masked_l.weight *= -1
    with pytest.raises(ValueError, match="stale"):
        model.get_gp_summary()
    model.freeze_ = True
    inherited = model.prepare_gp_analysis(overwrite=True)
    np.testing.assert_array_equal(inherited.orientation_sign, old.orientation_sign)
    assert inherited.inherited.all()


@pytest.mark.parametrize("save_adata", [False, True])
def test_checkpoint_roundtrip_and_legacy_opt_in(model, tmp_path, save_adata):
    model.model.target_rna_dynamic_decoder_mask[0, 1] = 0
    q = model.prepare_gp_analysis()
    model.save(str(tmp_path), overwrite=True, save_adata=save_adata)
    supplied = None if save_adata else model.adata.copy()
    loaded = NicheCompass.load(str(tmp_path), adata=supplied, use_cuda=False)
    pd.testing.assert_frame_equal(loaded.prepare_gp_analysis(), q)
    torch.testing.assert_close(loaded.model.target_rna_dynamic_decoder_mask,
                               model.model.target_rna_dynamic_decoder_mask)
    with (tmp_path / "attr.pkl").open("rb") as f:
        attrs = pickle.load(f)
    attrs = {k: v for k, v in attrs.items() if not k.startswith("gp_analysis")}
    with (tmp_path / "attr.pkl").open("wb") as f:
        pickle.dump(attrs, f)
    legacy = NicheCompass.load(str(tmp_path), adata=model.adata.copy(), use_cuda=False)
    assert legacy._gp_orientation() == "raw"
    legacy.prepare_gp_analysis()
    assert legacy._gp_orientation() == "canonical"


def test_checkpoint_addon_expansion(model, tmp_path):
    old = model.prepare_gp_analysis().set_index("gp_name")
    model.save(str(tmp_path), overwrite=True)
    mapped = NicheCompass.load(str(tmp_path), adata=model.adata.copy(), use_cuda=False,
                               n_addon_gps=1, gp_names_key=model.gp_names_key_,
                               genes_idx_key="nichecompass_genes_idx")
    q = mapped.prepare_gp_analysis(overwrite=True).set_index("gp_name")
    np.testing.assert_array_equal(q.loc[old.index, "orientation_sign"], old.orientation_sign)
    assert "Add-on_1_GP" in q.index


def test_multimodal_shared_sign_dynamic_mask_and_checkpoint(model, tmp_path, monkeypatch):
    rna = model.adata.copy()
    atac = ad.AnnData(rna.X.copy(), obs=rna.obs.copy())
    atac.var_names = [f"peak{i}" for i in range(6)]
    atac.layers["counts"] = atac.X.copy()
    rna.varm["nichecompass_gene_peaks"] = sp.eye(6, format="csr")
    for entity in ("target", "source"):
        mask = rna.varm[f"nichecompass_gp_{entity}s"]
        atac.varm[f"nichecompass_ca_{entity}s"] = sp.csr_matrix(mask)
        atac.uns[f"nichecompass_{entity}_peaks_idx"] = np.flatnonzero(mask.any(1))
    atac.uns["nichecompass_peaks_idx"] = np.r_[atac.uns["nichecompass_target_peaks_idx"],
                                                             atac.uns["nichecompass_source_peaks_idx"] + 6]
    m = NicheCompass(rna, adata_atac=atac, n_addon_gp=1, n_hidden_encoder=8,
                     use_cuda_if_available=False)
    m.is_trained_ = True
    m.node_batch_size_ = 8
    m.model.eval()
    with torch.no_grad():
        m.model.running_mean_abs_mu.fill_(1)
        for entity in ("target", "source"):
            getattr(m.model, f"{entity}_rna_decoder").load_state_dict(
                getattr(model.model, f"{entity}_rna_decoder").state_dict())
            layer = getattr(m.model, f"{entity}_atac_decoder").nb_means_normalized_decoder
            layer.masked_l.weight.fill_(3)
            layer.addon_l.weight.fill_(2)
    names = list(m.adata.uns[m.gp_names_key_])
    k = names.index("source_neg")
    m.model.target_atac_dynamic_decoder_mask[k, 1] = False
    table = m.get_gp_feature_table("source_neg")
    assert (table.orientation_sign == -1).all()
    assert table.query("modality == 'atac' and entity == 'target'").loading.iloc[0] == 0
    assert table.query("modality == 'atac' and entity == 'source'").loading.iloc[0] == -3
    assert m.get_gp_data("source_neg", orientation="canonical")[2][1, 0] == 0
    assert m.get_gp_data([], orientation="canonical")[2].shape == (12, 0)
    z = torch.randn(8, len(names))
    signs = torch.tensor(m._gp_signs(names), dtype=z.dtype)
    for entity in ("source", "target"):
        original = getattr(m.model, f"{entity}_atac_decoder")
        reflected = deepcopy(original)
        dynamic_mask = getattr(m.model, f"{entity}_atac_dynamic_decoder_mask")
        with torch.no_grad():
            layer = reflected.nb_means_normalized_decoder
            layer.masked_l.weight *= signs[:-1]
            layer.addon_l.weight *= signs[-1:]
        torch.testing.assert_close(original(z, torch.zeros(8, 1), dynamic_mask=dynamic_mask),
                                   reflected(z * signs, torch.zeros(8, 1), dynamic_mask=dynamic_mask))
    raw, std = m.get_gp_activities(names, orientation="raw", return_std=True)
    means, oriented_std = m.get_gp_activities(names, return_std=True)
    np.testing.assert_allclose(means, raw * m._gp_signs(names))
    np.testing.assert_array_equal(std, oriented_std)
    assert m.get_gp_activities(adata=m.adata[[3, 1]]).shape[0] == 2
    with pytest.raises(ValueError, match="ATAC"):
        m.get_gp_activities(adata_atac=m.adata_atac[::-1])
    m.save(str(tmp_path), overwrite=True, save_adata=True, save_adata_atac=True)
    loaded = NicheCompass.load(str(tmp_path), adata_atac_file_name="adata_atac.h5ad")
    pd.testing.assert_frame_equal(loaded.get_gp_feature_table("source_neg"), table)
    reordered_peaks = m.adata_atac[:, ::-1].copy()
    misaligned = NicheCompass.load(str(tmp_path), adata=m.adata.copy(), adata_atac=reordered_peaks)
    with pytest.raises(ValueError, match="feature names and order"):
        misaligned.prepare_gp_analysis(overwrite=True)
    import nichecompass.utils.analysis as analysis
    result = m.run_differential_gp_tests("group", selected_gps=["source_neg"], n_sample=10, return_all=True)
    m.adata.uns["nichecompass_differential_gp_test_results"] = result
    m.adata_atac.X = m.adata_atac.X.toarray()
    captured = {}
    monkeypatch.setattr(analysis, "plot_enriched_gp_info_plots_", lambda **kw: captured.update(kw))
    analysis.generate_enriched_gp_info_plots("test", m, "group",
        "nichecompass_differential_gp_test_results", "group", {"a": "red", "b": "blue"},
        n_top_peaks_per_gp=1)
    assert "adata" in captured


def test_plot_rejects_inconsistent_results(model, monkeypatch):
    from nichecompass.utils import generate_enriched_gp_info_plots
    monkeypatch.setattr(model, "get_latent_representation", fixed_posterior(model, np.zeros((8, 6)), np.ones((8, 6))))
    model.run_differential_gp_tests("group", n_sample=10, orientation="raw")
    model.prepare_gp_analysis()
    with pytest.raises(ValueError, match="orientation"):
        generate_enriched_gp_info_plots("test", model, "group",
                                       "nichecompass_differential_gp_test_results",
                                       "group", {"a": "red", "b": "blue"})


def test_train_initializes_orientation_and_keeps_raw_cache(model):
    model.is_trained_ = False
    model.train(n_epochs=1, n_epochs_all_gps=1, edge_val_ratio=0,
                node_val_ratio=0, node_batch_size=8, edge_batch_size=8,
                use_cuda_if_available=False, lambda_l1_addon=0,
                lambda_edge_recon=1, lambda_gene_expr_recon=1)
    assert model.gp_analysis_["policy"] == "paper_2025"
    raw = model.get_gp_activities(orientation="raw")
    np.testing.assert_allclose(model.adata.obsm[model.latent_key_], raw)


def test_plot_scores_and_signed_annotations_agree(model, monkeypatch):
    import nichecompass.utils.analysis as analysis
    means = np.ones((8, 6))
    means[:4] = -2
    monkeypatch.setattr(model, "get_latent_representation", fixed_posterior(model, means, np.zeros_like(means)))
    model.run_differential_gp_tests("group", "a", selected_gps=["source_neg"],
                                    direction="higher", n_sample=10)
    model.adata.obs["source_neg"] = 99  # A stale user-edited plotting column.
    captured = {}
    monkeypatch.setattr(analysis, "plot_enriched_gp_info_plots_", lambda **kw: captured.update(kw))
    analysis.generate_enriched_gp_info_plots("test", model, "group",
        "nichecompass_differential_gp_test_results", "group", {"a": "red", "b": "blue"})
    np.testing.assert_array_equal(captured["adata"].obs.source_neg, -means[:, 0])
    assert captured["adata"].uns["source_neg_source_genes_top_gene_signs"].tolist() == ["+"]
    assert captured["adata"].uns["source_neg_target_genes_top_gene_signs"].tolist() == ["-"]


def test_communication_paired_raw_convention_and_zero_expression(model):
    from nichecompass.utils import compute_communication_gp_network
    # Align source/target signs to produce nonzero communication strengths.
    k = list(model.adata.uns[model.gp_names_key_]).index("source_neg")
    with torch.no_grad():
        model.model.target_rna_decoder.nb_means_normalized_decoder.masked_l.weight[1, k] = -10
    active = model.get_active_gps()
    model.adata.uns[model.active_gp_names_key_] = active
    model.adata.obsm[model.latent_key_] = -np.arange(1, 9)[:, None] * np.ones((8, len(active)))
    model.adata.uns["spatial_cci"] = {"params": {"n_neighbors": 3}}
    model.adata.obsp["spatial_cci_connectivities"] = model.adata.obsp[model.adj_key_].copy()
    first = compute_communication_gp_network(["source_neg"], model, "group", n_neighbors=3)
    model.prepare_gp_analysis()
    model.add_active_gp_scores_to_obs(use_cached=True)
    second = compute_communication_gp_network(["source_neg"], model, "group", n_neighbors=3)
    pd.testing.assert_frame_equal(first, second)
    assert not first.empty
    for gp in ("target_only", "Add-on_0_GP"):
        with pytest.raises(ValueError, match="prior"):
            compute_communication_gp_network([gp], model, "group", n_neighbors=3)
    model.adata.X = sp.csr_matrix(model.adata.shape, dtype=np.float32)
    empty = compute_communication_gp_network(["source_neg"], model, "group", n_neighbors=3)
    assert empty.empty
    assert np.isfinite(model.adata.obs.source_neg_source_score).all()


def test_custom_keys_and_new_covariate_mapping(model, tmp_path):
    adata = model.adata.copy()
    adata.uns["programs"] = adata.uns.pop(model.gp_names_key_)
    adata.obsp["neighbors"] = adata.obsp.pop(model.adj_key_)
    adata.layers["observed_counts"] = adata.layers.pop("counts")
    adata.obs["sample"] = ["reference"] * 8
    m = NicheCompass(adata, n_addon_gp=1, n_hidden_encoder=8,
                     gp_names_key="programs", adj_key="neighbors", counts_key="observed_counts",
                     cat_covariates_keys=["sample"], use_cuda_if_available=False)
    m.is_trained_ = True
    m.node_batch_size_ = 8
    m.model.eval()
    with torch.no_grad():
        m.model.running_mean_abs_mu.fill_(1)
    old = m.prepare_gp_analysis()
    m.save(str(tmp_path), overwrite=True, save_adata=True)
    query = m.adata.copy()
    query.obs["sample"] = ["query"] * 8
    mapped = NicheCompass.load(str(tmp_path), adata=query,
                               unfreeze_cat_covariates_embedder_weights=True)
    pd.testing.assert_frame_equal(mapped.prepare_gp_analysis(), old)
    assert mapped.get_gp_activities().shape[0] == 8


def test_loading_redistribution_invalidates_old_results(model, monkeypatch):
    from nichecompass.utils import generate_enriched_gp_info_plots
    monkeypatch.setattr(model, "get_latent_representation",
                        fixed_posterior(model, np.zeros((8, 6)), np.ones((8, 6))))
    model.run_differential_gp_tests("group", n_sample=10)
    before = model.gp_analysis_["orientation_id"]
    quality = model.prepare_gp_analysis()
    k = list(model.adata.uns[model.gp_names_key_]).index("balanced")
    with torch.no_grad():
        # Same signed/absolute mass and orientation, different member loadings.
        model.model.source_rna_decoder.nb_means_normalized_decoder.masked_l.weight[:, k] *= -1
    refreshed = model.prepare_gp_analysis(overwrite=True)
    pd.testing.assert_frame_equal(quality, refreshed)
    assert model.gp_analysis_["orientation_id"] != before
    with pytest.raises(ValueError, match="orientation"):
        generate_enriched_gp_info_plots("test", model, "group",
                                       "nichecompass_differential_gp_test_results",
                                       "group", {"a": "red", "b": "blue"})


def test_mixed_score_column_provenance_and_result_serialization(model, monkeypatch, tmp_path):
    monkeypatch.setattr(model, "get_latent_representation",
                        fixed_posterior(model, np.ones((8, 6)), np.ones((8, 6))))
    model.add_active_gp_scores_to_obs()
    model.run_differential_gp_tests("group", selected_gps=["source_neg"],
                                    orientation="raw", n_sample=10)
    metadata = model.adata.uns["nichecompass_gp_score_columns"]
    assert metadata["orientation"] == "mixed"
    table = metadata["table"].set_index("gp_name")
    assert table.loc["source_neg", "orientation"] == "raw"
    assert table.loc["balanced", "orientation"] == "canonical"
    model.add_active_gp_scores_to_obs()
    assert model.adata.uns["nichecompass_gp_score_columns"]["orientation"] == "canonical"
    model.adata.write_h5ad(tmp_path / "results.h5ad")
    reloaded = ad.read_h5ad(tmp_path / "results.h5ad")
    # AnnData may normalize pandas object strings to StringDtype on read.
    pd.testing.assert_frame_equal(reloaded.uns["nichecompass_gp_score_columns"]["table"],
                                   model.adata.uns["nichecompass_gp_score_columns"]["table"], check_dtype=False)
    pd.testing.assert_frame_equal(reloaded.uns["nichecompass_differential_gp_test_results_all"],
                                   model.adata.uns["nichecompass_differential_gp_test_results_all"], check_dtype=False)


def test_gaussian_superiority_matches_independent_normal_cdf(model, monkeypatch):
    from scipy.stats import norm
    means, std = np.zeros((8, 6)), np.ones((8, 6))
    means[:4] = 1.25
    std[:4] = 2
    monkeypatch.setattr(model, "get_latent_representation", fixed_posterior(model, means, std))
    result = model.run_differential_gp_tests("group", "a", selected_gps=["source_neg"],
                                            n_sample=10, return_all=True).iloc[0]
    expected = norm.cdf(-1.25 / np.sqrt(2**2 + 1**2))
    assert result.p_higher == pytest.approx(expected, abs=1e-14)
    assert result.log_bayes_factor == pytest.approx(np.log(expected) - np.log1p(-expected), abs=1e-10)


def test_per_call_overrides_do_not_silently_migrate_legacy_default(model, monkeypatch):
    model.gp_analysis_default_orientation_ = "raw"
    monkeypatch.setattr(model, "get_latent_representation",
                        fixed_posterior(model, np.ones((8, 6)), np.ones((8, 6))))
    model.get_gp_quality_table()
    model.get_gp_summary(orientation="canonical")
    model.get_gp_activities(orientation="canonical")
    model.run_differential_gp_tests("group", orientation="canonical", n_sample=10)
    model.add_active_gp_scores_to_obs(orientation="canonical")
    assert model._gp_orientation() == "raw"
    model.prepare_gp_analysis()
    assert model._gp_orientation() == "canonical"


@pytest.mark.parametrize("change", ["encoder", "counts", "adjacency", "groups"])
def test_plot_rejects_changed_inference_inputs(model, monkeypatch, change):
    from nichecompass.utils import generate_enriched_gp_info_plots
    means = np.ones((8, 6))
    means[:4] = -2
    monkeypatch.setattr(model, "get_latent_representation", fixed_posterior(model, means, np.ones_like(means)))
    model.run_differential_gp_tests("group", "a", selected_gps=["source_neg"], n_sample=10)
    orientation_id = model.gp_analysis_["orientation_id"]
    if change == "encoder":
        with torch.no_grad():
            next(model.model.encoder.parameters()).add_(0.01)
    elif change == "counts":
        model.adata.layers[model.counts_key_].data[0] += 1
    elif change == "adjacency":
        model.adata.obsp[model.adj_key_].data[0] = 0
    else:
        model.adata.obs.loc[model.adata.obs_names[0], "group"] = "b"
    assert model.gp_analysis_["orientation_id"] == orientation_id
    with pytest.raises(ValueError, match="stale"):
        generate_enriched_gp_info_plots("test", model, "group",
            "nichecompass_differential_gp_test_results", "group", {"a": "red", "b": "blue"})


def test_input_fingerprint_survives_result_save_load(model, tmp_path, monkeypatch):
    monkeypatch.setattr(model, "get_latent_representation",
                        fixed_posterior(model, np.ones((8, 6)), np.ones((8, 6))))
    model.run_differential_gp_tests("group", n_sample=10)
    params = model.adata.uns["nichecompass_differential_gp_test_results_params"]
    before = params["input_fingerprint"]
    model.adata.obs["unrelated_annotation"] = "new"
    model.adata.obs["source_neg"] = -99
    assert model._gp_input_fingerprint(model.adata, "group") == before
    model.save(str(tmp_path), overwrite=True, save_adata=True)
    loaded = NicheCompass.load(str(tmp_path))
    assert loaded._gp_input_fingerprint(loaded.adata, "group") == before


@pytest.mark.parametrize("legacy", [False, True])
def test_loaded_checkpoint_rejects_reordered_rna_analysis(model, tmp_path, legacy):
    model.prepare_gp_analysis()
    model.save(str(tmp_path), overwrite=True)
    if legacy:
        with (tmp_path / "attr.pkl").open("rb") as handle:
            attrs = pickle.load(handle)
        with (tmp_path / "attr.pkl").open("wb") as handle:
            pickle.dump({k: v for k, v in attrs.items() if not k.startswith("gp_analysis")}, handle)
    loaded = NicheCompass.load(str(tmp_path), adata=model.adata[:, ::-1].copy())
    with pytest.raises(ValueError, match="feature names and order"):
        loaded.prepare_gp_analysis(overwrite=True)
    with pytest.raises(ValueError, match="feature names and order"):
        loaded.get_gp_activities(orientation="raw")


def test_in_place_feature_relabel_is_rejected(model):
    model.adata.var_names = model.adata.var_names[::-1]
    with pytest.raises(ValueError, match="feature names and order"):
        model.get_gp_feature_table()


@pytest.mark.parametrize("orientation", ["raw", "canonical"])
def test_empty_gp_selection_is_consistent(model, orientation):
    indices, rna, atac = model.get_gp_data([], orientation=orientation)
    assert indices.dtype.kind in "iu" and len(indices) == 0
    assert rna.shape == (12, 0) and atac is None
    assert model.get_gp_activities([], orientation=orientation).shape == (8, 0)
    assert model.get_gp_feature_table([], orientation=orientation).empty


def test_disagreement_diagnostic_uses_full_precision_sums(model):
    k = list(model.adata.uns[model.gp_names_key_]).index("balanced")
    layer = model.model.source_rna_decoder.nb_means_normalized_decoder.masked_l
    with torch.no_grad():
        layer.mask[2, k] = True
        layer.weight[:3, k] = torch.tensor([1, 1e-8, -1])
    row = model.prepare_gp_analysis().set_index("gp_name").loc["balanced"]
    assert row.signed_mass > 0
    assert row.source_target_disagree


def test_freeze_mask_does_not_change_the_input_fingerprint(model):
    """´frozen_gp_statistic_mask´ is a freeze control, not an inference input.

    It is non-persistent and ´load´ derives it from the requested unfreeze
    configuration, so a reloaded model always had it set while the model that
    wrote the results had it clear. Hashing it therefore reported every saved
    result stale after a reload.
    """
    model.prepare_gp_analysis()
    before = model._gp_input_fingerprint(model.adata, "group")
    mask = model.model.frozen_gp_statistic_mask
    assert not bool(mask.any()), "fixture should start with nothing held"
    model.model.frozen_gp_statistic_mask = torch.ones_like(mask)
    assert model._gp_input_fingerprint(model.adata, "group") == before
