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


def test_parameter_groups_are_exact_and_exhaustive():
    """Every parameter lands in exactly one group, and the add-on tensors
    inside the encoder and the decoders are claimed by the add-on group rather
    than by the broader ones."""
    cases = {
        "encoder.conv_l1.lin.weight": "encoder",
        "encoder.fc_l2_bn.weight": "encoder",
        "encoder.addon_conv_mu.lin.weight": "addon_gp",
        "target_rna_decoder.nb_means_normalized_decoder.masked_l.weight":
            "prior_gp_decoder",
        "target_rna_decoder.nb_means_normalized_decoder.addon_l.weight":
            "addon_gp",
        "source_rna_decoder.nb_means_normalized_decoder."
        "cat_covariates_embed_l.weight": "cat_covariates_embedder",
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


def test_frozen_batch_norm_statistics_do_not_drift(reference):
    """Buffers are outside the freeze, and Trainer puts the module back into
    train mode, so a frozen batch norm used to rewrite its running statistics
    on query minibatches while the latent was read in eval mode."""
    model, path = reference
    loaded = NicheCompass.load(str(path), adata_file_name="adata.h5ad",
                               unfreeze_cat_covariates_embedder_weights=True)
    bns = [m for m in loaded.model.modules()
           if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        pytest.skip("default encoder has one fc layer, so no batch norm")
    for bn in bns:
        assert bn.track_running_stats is False
    before = [bn.running_mean.detach().clone() for bn in bns]
    loaded.train(n_epochs=2, use_cuda_if_available=False)
    for bn, was in zip(bns, before):
        torch.testing.assert_close(bn.running_mean, was)
