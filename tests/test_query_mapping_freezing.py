"""What ´load´ freezes, and what a frozen model is allowed to change.

´requires_grad=False´ only stops gradients. These tests pin the other routes
by which a "frozen" reference model used to change on query data: the running
statistic that decides which gene programs are active, the destructive pruning
driven by it, the in-place mask write into a frozen decoder weight, and batch
norm running statistics.
"""

import numpy as np
import pytest
import torch

from nichecompass.models import NicheCompass
from nichecompass.models.basemodelmixin import (PARAMETER_GROUPS,
                                                _parameter_group_of,
                                                _parameter_groups)
from tests.test_gp_analysis import model


@pytest.fixture
def reference(model, tmp_path):
    """A trained reference model saved to disk, ready to be loaded as query."""
    model.is_trained_ = True
    model.save(str(tmp_path), overwrite=True, save_adata=True)
    return model, tmp_path


def test_every_parameter_of_a_real_model_is_classified(reference):
    """The literal cases below cannot catch a parameter the predicates do not
    recognise, because the classifier ends in an unconditional fallback to
    ´prior_gp_decoder´. Enumerate an actual model instead."""
    model, _ = reference
    groups = _parameter_groups(model.model)
    classified = {n for names in groups.values() for n in names}
    assert classified == set(dict(model.model.named_parameters()))
    # Nothing unexpected may land in the fallback group.
    for name in groups["prior_gp_decoder"]:
        assert "masked_l" in name or "nb_means" in name or "decoder" in name, (
            f"{name} fell into the prior_gp_decoder catch-all")


def test_parameter_groups_are_exact_and_exhaustive():
    """Every parameter lands in exactly one group, and the add-on tensors
    inside the encoder and the decoders are claimed by the add-on group rather
    than by the broader ones."""
    cases = {
        "encoder.conv_l1.lin.weight": "encoder",
        "encoder.fc_l2_bn.weight": "encoder",
        "encoder.addon_conv_mu.lin.weight": "addon_gp_encoder",
        "target_rna_decoder.nb_means_normalized_decoder.masked_l.weight":
            "prior_gp_decoder",
        "target_rna_decoder.nb_means_normalized_decoder.addon_l.weight":
            "addon_gp",
        "source_rna_decoder.nb_means_normalized_decoder."
        "cat_covariates_embed_l.weight": "cat_covariates_projection",
        "cat_covariate0_embedder.weight": "cat_covariates_embedder",
        "target_rna_theta": "dispersion",
        "rna_node_label_aggregator.attn": "node_label_aggregator",
    }
    for name, expected in cases.items():
        assert _parameter_group_of(name) == expected, name
        assert expected in PARAMETER_GROUPS


def test_default_load_freezes_everything_and_says_so(reference, capsys):
    model, path = reference
    loaded = NicheCompass.load(str(path), adata_file_name="adata.h5ad")
    assert all(not p.requires_grad for p in loaded.model.parameters())
    assert loaded.freeze_ is True
    assert loaded.model.freeze_ is True          # the module knows too
    assert loaded.unfrozen_parameter_names_ == []
    assert "All parameters are frozen" in capsys.readouterr().out


def test_training_a_fully_frozen_model_raises_before_it_wastes_a_job(reference):
    model, path = reference
    loaded = NicheCompass.load(str(path), adata_file_name="adata.h5ad")
    with pytest.raises(ValueError, match="frozen"):
        loaded.train(n_epochs=1, use_cuda_if_available=False)


def test_encoder_can_be_unfrozen_while_the_loadings_stay_fixed(reference):
    """The configuration query mapping needs and could not express before:
    the encoder adapts to the query's neighbourhoods, the gene program axes do
    not move, and the orientation stays inherited."""
    model, path = reference
    loaded = NicheCompass.load(str(path), adata_file_name="adata.h5ad",
                               unfreeze_encoder_weights=True)
    groups = _parameter_groups(loaded.model)
    by_name = dict(loaded.model.named_parameters())
    assert groups["encoder"], "no encoder parameters found"
    assert all(by_name[n].requires_grad for n in groups["encoder"])
    assert all(not by_name[n].requires_grad
               for n in groups["prior_gp_decoder"])
    # freeze_ stays True, so gene program orientations are still inherited.
    assert loaded.freeze_ is True


def test_unfreeze_all_clears_freeze_so_orientation_is_recomputed(reference):
    model, path = reference
    loaded = NicheCompass.load(str(path), adata_file_name="adata.h5ad",
                               unfreeze_all_weights=True)
    assert all(p.requires_grad for p in loaded.model.parameters())
    assert loaded.freeze_ is False
    assert loaded.model.freeze_ is False


def test_a_frozen_model_does_not_prune_its_reference_gene_programs(reference):
    """The defect this file exists for. A frozen query run used to spend most
    of its epochs deleting reference gene programs: the running statistic
    drifted to query data, the active mask was derived from it, and the
    dynamic decoder masks were zeroed irreversibly and then saved."""
    model, path = reference
    loaded = NicheCompass.load(str(path), adata_file_name="adata.h5ad",
                               unfreeze_cat_covariates_embedder_weights=True)
    before_stat = loaded.model.running_mean_abs_mu.detach().clone()
    before_masks = {n: b.detach().clone()
                    for n, b in loaded.model.named_buffers()
                    if "dynamic_decoder_mask" in n}
    before_loadings = {
        n: p.detach().clone() for n, p in loaded.model.named_parameters()
        if "masked_l.weight" in n}
    assert before_masks, "no dynamic masks on the model"

    # n_epochs_all_gps=0 makes use_only_active_gps True from the first epoch,
    # which is when the pruning used to fire.
    loaded.train(n_epochs=2, n_epochs_all_gps=0, use_cuda_if_available=False)

    torch.testing.assert_close(loaded.model.running_mean_abs_mu, before_stat)
    for name, before in before_masks.items():
        after = dict(loaded.model.named_buffers())[name]
        assert torch.equal(after, before), f"{name} was mutated"
    for name, before in before_loadings.items():
        after = dict(loaded.model.named_parameters())[name]
        torch.testing.assert_close(after, before)


@pytest.fixture
def reference_with_batch_norm(tmp_path):
    """A saved reference whose encoder has two fully connected layers, and
    therefore a batch norm.

    The shared ´model´ fixture leaves ´n_fc_layers_encoder´ at its default of
    1, which builds no ´fc_l2_bn´ at all - so a batch norm test written
    against it skips unconditionally and the pinning it checks has no
    executed coverage.
    """
    import anndata as ad
    import numpy as np
    import scipy.sparse as sp
    from nichecompass.models import NicheCompass
    from nichecompass.utils import add_gps_from_gp_dict_to_adata

    adata = ad.AnnData(np.arange(48, dtype=np.float32).reshape(8, 6) % 7 + 1)
    adata.var_names = ["L", "R", "T", "A", "B", "C"]
    adata.X = sp.csr_matrix(adata.X)
    adata.layers["counts"] = adata.X.copy()
    adata.obsp["spatial_connectivities"] = sp.csr_matrix(
        np.ones((8, 8)) - np.eye(8))
    adata.obs["group"] = ["a"] * 4 + ["b"] * 4
    gps = {"source_neg": {"sources": ["L"], "targets": ["R"],
                          "sources_categories": ["ligand"],
                          "targets_categories": ["receptor"]}}
    add_gps_from_gp_dict_to_adata(gps, adata)
    m = NicheCompass(adata, n_addon_gp=1, n_hidden_encoder=8,
                     n_fc_layers_encoder=2, use_cuda_if_available=False)
    m.is_trained_ = True
    m.node_batch_size_ = 8
    m.save(str(tmp_path), overwrite=True, save_adata=True)
    return m, tmp_path


def test_the_batch_norm_fixture_really_has_one(reference_with_batch_norm):
    """Guards the guard: if this ever stops finding a batch norm, the pinning
    test below is silently skipping again."""
    model, _ = reference_with_batch_norm
    bns = [m for m in model.model.modules()
           if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    assert bns, "the two-fc-layer encoder built no batch norm"


def test_frozen_batch_norm_statistics_really_do_not_drift(
        reference_with_batch_norm):
    """´track_running_stats=False´ alone does NOT keep a batch norm using its
    stored statistics - in training mode it switches the layer to the current
    minibatch. ´VGPGAE.train´ re-asserting eval mode is what actually pins
    them, and ´Trainer´ calls ´train()´ once per epoch."""
    model, path = reference_with_batch_norm
    loaded = NicheCompass.load(str(path), adata_file_name="adata.h5ad",
                               unfreeze_dispersion=True)
    bns = [m for m in loaded.model.modules()
           if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    assert bns
    for bn in bns:
        assert bn.track_running_stats is False
        assert not bn.training, "pinned layer is not in eval mode after load"

    before = [bn.running_mean.detach().clone() for bn in bns]
    # what Trainer does at the top of every epoch
    loaded.model.train()
    for bn in bns:
        assert not bn.training, (
            "train() did not re-assert eval mode on the pinned layer")
    loaded.train(n_epochs=2, use_cuda_if_available=False)
    for bn, was in zip(bns, before):
        torch.testing.assert_close(bn.running_mean, was)


def test_graph_adapter_is_the_identity_before_training():
    """The anchoring property: attaching an adapter must not change a single
    output until it is trained."""
    from nichecompass.nn import GraphAdapter
    torch.manual_seed(0)
    adapter = GraphAdapter(n_input=8, n_bottleneck=3)
    x = torch.randn(6, 8)
    torch.testing.assert_close(adapter(x), x)
    with torch.no_grad():
        adapter.up.weight.normal_()
    assert not torch.allclose(adapter(x), x)


def test_graph_adapter_does_no_message_passing_of_its_own():
    """It must not add a hop. A single layer encoder aggregates over one hop
    and the loaders sample one hop; an adapter with its own convolution would
    make the encoder two hops deep and read neighbours the sampler truncated."""
    from nichecompass.nn import GraphAdapter
    import inspect
    torch.manual_seed(0)
    adapter = GraphAdapter(n_input=6, n_bottleneck=4)
    with torch.no_grad():
        adapter.up.weight.normal_()
    # No graph argument, and no convolution submodule.
    assert list(inspect.signature(adapter.forward).parameters) == ["x"]
    assert not any("conv" in name for name, _ in adapter.named_modules())
    # Each row is transformed independently of every other row.
    x = torch.randn(5, 6)
    rowwise = torch.cat([adapter(x[i:i + 1]) for i in range(x.size(0))])
    torch.testing.assert_close(adapter(x), rowwise)


def test_composition_sensitivity_comes_from_the_following_convolution():
    """The adapter is per cell, but the convolution that already follows it
    aggregates neighbours' corrections, and those differ by cell type - so the
    adapted latent depends on WHICH cells are neighbours, with no extra hop."""
    from nichecompass.nn import GraphAdapter
    torch.manual_seed(0)
    adapter = GraphAdapter(n_input=6, n_bottleneck=4)
    with torch.no_grad():
        adapter.up.weight.normal_(0, 0.5)

    def mean_aggregate(h, neighbours):
        return torch.stack([h] + list(neighbours)).mean(0)

    type_a, type_b, focal = (torch.randn(6) for _ in range(3))
    # identical degree, different neighbourhood composition
    plain_a = mean_aggregate(focal, [type_a, type_a])
    plain_b = mean_aggregate(focal, [type_b, type_b])
    adapted_a = mean_aggregate(adapter(focal[None])[0],
                               [adapter(type_a[None])[0]] * 2)
    adapted_b = mean_aggregate(adapter(focal[None])[0],
                               [adapter(type_b[None])[0]] * 2)
    delta_a, delta_b = adapted_a - plain_a, adapted_b - plain_b
    assert not torch.allclose(delta_a, delta_b, atol=1e-6), (
        "the adapter's effect did not depend on neighbourhood composition")


def test_adapters_can_be_attached_to_a_trained_reference(reference):
    """Retrofit: the property encoder covariate injection does not have."""
    model, path = reference
    loaded = NicheCompass.load(str(path), adata_file_name="adata.h5ad",
                               n_graph_adapter_hidden=4,
                               unfreeze_graph_adapters=True)
    groups = _parameter_groups(loaded.model)
    by_name = dict(loaded.model.named_parameters())
    assert groups["graph_adapter"], "no adapter parameters were created"
    # One adapter per graph convolution stage.
    n_adapters = len({n.split("graph_adapters.")[1].split(".")[0]
                      for n in groups["graph_adapter"]})
    assert n_adapters == loaded.model.encoder.n_layers, (
        f"{n_adapters} adapters for {loaded.model.encoder.n_layers} layers")
    for name in groups["graph_adapter"]:
        assert by_name[name].requires_grad, name
    # Everything else stays frozen, including the loadings.
    for name in groups["prior_gp_decoder"] + groups["encoder"]:
        assert not by_name[name].requires_grad, name
    # Attaching them does not by itself change the latent.
    assert loaded.freeze_ is True


def test_attaching_adapters_does_not_trigger_the_latent_warning(reference, recwarn):
    model, path = reference
    NicheCompass.load(str(path), adata_file_name="adata.h5ad",
                      n_graph_adapter_hidden=4,
                      unfreeze_graph_adapters=True)
    assert not [w for w in recwarn
                if "can change the latent space" in str(w.message)]


def test_unfreezing_the_encoder_carries_its_addon_heads(reference):
    """´encoder.addon_conv_*´ reads the hidden representation the encoder
    produces. Leaving it frozen while the encoder moves would drift the add-on
    activities with nothing able to compensate at either end."""
    model, path = reference
    loaded = NicheCompass.load(str(path), adata_file_name="adata.h5ad",
                               unfreeze_encoder_weights=True)
    by_name = dict(loaded.model.named_parameters())
    groups = _parameter_groups(loaded.model)
    for name in groups["encoder"] + groups["addon_gp_encoder"]:
        assert by_name[name].requires_grad, name
    # The loadings still do not move, which is the point of the flag.
    for name in groups["prior_gp_decoder"] + groups["addon_gp_decoder"]:
        assert not by_name[name].requires_grad, name


def test_the_historical_covariate_flag_keeps_its_old_scope(reference):
    """It used to match only ´cat_covariate{i}_embedder´. The projection into
    the decoder is shared with the reference, so it needs its own opt-in."""
    model, path = reference
    loaded = NicheCompass.load(
        str(path), adata_file_name="adata.h5ad",
        unfreeze_cat_covariates_embedder_weights=True)
    groups = _parameter_groups(loaded.model)
    by_name = dict(loaded.model.named_parameters())
    for name in groups["cat_covariates_projection"]:
        assert not by_name[name].requires_grad, name
    for name in groups["cat_covariates_embedder"]:
        assert by_name[name].requires_grad, name


def test_addon_programs_added_for_the_query_still_get_a_statistic(reference):
    """Holding the activity statistic for the whole module starved add-on
    programs added at load time: their entries start at zero, and a zero
    statistic makes them either vacuously active or permanently inactive."""
    model, path = reference
    loaded = NicheCompass.load(
        str(path), adata_file_name="adata.h5ad",
        n_addon_gps=2, gp_names_key=model.gp_names_key_,
        unfreeze_addon_gp_weights=True)
    n_prior = loaded.model.n_prior_gp_
    hold = loaded.model.frozen_gp_statistic_mask
    assert hold[:n_prior].all(), "prior statistics must be held"
    assert not hold[n_prior:].any(), "add-on statistics must keep updating"

    before_prior = loaded.model.running_mean_abs_mu[:n_prior].detach().clone()
    loaded.train(n_epochs=2, n_epochs_all_gps=0, use_cuda_if_available=False)
    torch.testing.assert_close(
        loaded.model.running_mean_abs_mu[:n_prior], before_prior)
    assert (loaded.model.running_mean_abs_mu[n_prior:] != 0).any(), (
        "add-on programs never accumulated an activity statistic")


def test_warns_when_nothing_unfrozen_can_reach_the_latent(reference):
    """Hours of fine tuning that changes no gene program score.

    Keyed on a group that is EMPTY rather than on the injection list: asking
    for adapters on a model that has none unfreezes nothing, and pairing it
    with the dispersion means ´unfrozen´ is non-empty so neither the
    all-frozen notice nor ´Trainer´'s refusal fires. That is precisely the
    case the warning exists for, and the previous version of this test
    skipped unconditionally once "encoder" entered the default injection.
    """
    model, path = reference
    with pytest.warns(UserWarning, match="unfreeze_encoder_weights"):
        NicheCompass.load(
            str(path), adata_file_name="adata.h5ad",
            unfreeze_graph_adapters=True,
            unfreeze_dispersion=True)


def test_asking_for_absent_adapters_says_so(reference):
    model, path = reference
    with pytest.warns(UserWarning, match="n_graph_adapter_hidden"):
        NicheCompass.load(str(path), adata_file_name="adata.h5ad",
                          unfreeze_graph_adapters=True)


def test_a_conflicting_adapter_width_is_refused(reference):
    """Narrowing existing adapters died inside a ´torch.cat´ with a message
    naming neither the adapter nor the width."""
    model, path = reference
    wide = NicheCompass.load(str(path), adata_file_name="adata.h5ad",
                             n_graph_adapter_hidden=8)
    wide.is_trained_ = True
    wide.save(str(path / "wide"), overwrite=True, save_adata=True)
    with pytest.raises(ValueError, match="width"):
        NicheCompass.load(str(path / "wide"), adata_file_name="adata.h5ad",
                          n_graph_adapter_hidden=4)


def test_no_warning_when_the_encoder_is_unfrozen(reference, recwarn):
    model, path = reference
    NicheCompass.load(str(path), adata_file_name="adata.h5ad",
                      unfreeze_encoder_weights=True)
    assert not [w for w in recwarn
                if "can change the latent space" in str(w.message)]
