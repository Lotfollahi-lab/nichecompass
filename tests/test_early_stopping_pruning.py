"""Early stopping across the start of gene program pruning.

Pruning starts after ´n_epochs_all_gps´ epochs and makes the loss jump,
because the programs it removes stop contributing to the reconstruction. If
early stopping remembers the best value from before, it judges every pruned
epoch against an unpruned model: an unpruned state stays the best one, is
reloaded at the end, and training stops ´patience´ epochs after pruning began.
The reloaded state also lacked the dynamic decoder masks, which the state dict
does not hold, so it came back with the masks of the last epoch.

The tests train a tiny model and replace the loss early stopping reads with a
scripted curve, so that every decision is known in advance.
"""

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from nichecompass.models import NicheCompass
from nichecompass.train import trainer as trainer_module
from nichecompass.train.utils import EarlyStopping
from nichecompass.utils import add_gps_from_gp_dict_to_adata


def _model(n_addon_gp=1):
    adata = ad.AnnData(np.arange(48, dtype=np.float32).reshape(8, 6) % 7 + 1)
    adata.var_names = ["L", "R", "T", "A", "B", "C"]
    adata.X = sp.csr_matrix(adata.X)
    adata.layers["counts"] = adata.X.copy()
    adata.obsp["spatial_connectivities"] = sp.csr_matrix(
        np.ones((8, 8)) - np.eye(8))
    definitions = {"lr": (["L"], ["R"]), "t": ([], ["T"]),
                   "lrt": (["L", "R"], ["T"]), "ab": (["A"], ["B"]),
                   "c": ([], ["C"])}
    gps = {name: {"sources": sources, "targets": targets,
                  "sources_categories": ["ligand"] * len(sources),
                  "targets_categories": ["receptor"] * len(targets)}
           for name, (sources, targets) in definitions.items()}
    add_gps_from_gp_dict_to_adata(gps, adata)
    torch.manual_seed(0)
    return NicheCompass(adata, n_addon_gp=n_addon_gp, n_hidden_encoder=8,
                        use_cuda_if_available=False)


def _masks(model):
    return {name: buffer.detach().clone()
            for name, buffer in model.model.named_buffers()
            if "dynamic_decoder_mask" in name}


def _train(monkeypatch, curve, *, n_epochs_all_gps, prune_aware,
           cut_mask_at=None, patience=3, lr_patience=2, n_epochs=None):
    """Train with the early stopping metric replaced by ´curve´.

    Records, per judged epoch, the dynamic masks and the learning rate as
    early stopping saw them. With ´cut_mask_at´, the first program still
    connected in the decoders is cut from every dynamic mask just before that
    epoch is judged, as if pruning had removed one more program then.
    """
    model = _model()
    seen = {}
    original = trainer_module.Trainer.is_early_stopping

    def scripted(self):
        metric = self.early_stopping.early_stopping_metric
        self.epoch_logs[metric][-1] = curve[self.epoch]
        if cut_mask_at is not None and self.epoch == cut_mask_at:
            row = int(torch.nonzero(
                self.model.target_rna_dynamic_decoder_mask.any(dim=1))[0])
            with torch.no_grad():
                for name, buffer in self.model.named_buffers():
                    if "dynamic_decoder_mask" in name:
                        buffer[row, :] = 0
        seen[self.epoch] = {"masks": _masks(model),
                            "lr": self.optimizer.param_groups[0]["lr"],
                            "pruning": self.use_only_active_gps}
        return original(self)

    monkeypatch.setattr(trainer_module.Trainer, "is_early_stopping", scripted)
    model.train(n_epochs=n_epochs or len(curve),
                n_epochs_all_gps=n_epochs_all_gps, edge_val_ratio=0,
                node_val_ratio=0, node_batch_size=8, edge_batch_size=8,
                use_cuda_if_available=False, lambda_l1_addon=0,
                lambda_edge_recon=1, lambda_gene_expr_recon=1, monitor=False,
                prune_aware_early_stopping=prune_aware,
                early_stopping_kwargs={"patience": patience,
                                       "lr_patience": lr_patience})
    return model, seen


# Falls to 7 before pruning (epochs 0-3), jumps to 12 when pruning starts in
# epoch 4, reaches its pruned best of 11 in epoch 5, then rises.
JUMP = [10, 9, 8, 7, 12, 11, 11.5, 11.6, 11.7, 11.8, 11.9, 12.0]


def test_reset_restores_the_initial_state():
    es = EarlyStopping(patience=2, lr_patience=1)
    for value in (3., 2., 2.5, 2.6):
        es.step(value)
        es.update_state(value)
    es.reset()
    fresh = EarlyStopping(patience=2, lr_patience=1)
    assert vars(es) == vars(fresh)


def test_previous_behaviour_keeps_an_unpruned_state_and_stops_early(
        monkeypatch):
    """The defect, pinned so that ´prune_aware_early_stopping=False´ keeps
    reproducing it exactly."""
    model, seen = _train(monkeypatch, JUMP, n_epochs_all_gps=4,
                         prune_aware=False, cut_mask_at=6)
    trainer = model.trainer
    assert trainer.best_epoch == 3                 # the last unpruned epoch
    assert not seen[trainer.best_epoch]["pruning"]
    assert trainer.epoch == 6                      # 3 epochs after the best
    # The reloaded model mixes the weights of epoch 3 with the masks of the
    # last epoch, in which one more program had been cut.
    final = _masks(model)
    assert any(not torch.equal(final[n], seen[3]["masks"][n]) for n in final)
    assert all(torch.equal(final[n], seen[6]["masks"][n]) for n in final)


def test_best_state_comes_from_a_pruned_epoch(monkeypatch):
    model, seen = _train(monkeypatch, JUMP, n_epochs_all_gps=4,
                         prune_aware=True, cut_mask_at=6)
    trainer = model.trainer
    assert trainer.best_epoch == 5
    assert seen[trainer.best_epoch]["pruning"]
    # Patience counts from the pruned best, not from the unpruned one.
    assert trainer.epoch == 8


def test_reloaded_state_brings_back_the_masks_of_its_epoch(monkeypatch):
    model, seen = _train(monkeypatch, JUMP, n_epochs_all_gps=4,
                         prune_aware=True, cut_mask_at=6)
    best = model.trainer.best_epoch
    final = _masks(model)
    assert any(not torch.equal(seen[best]["masks"][n], seen[8]["masks"][n])
               for n in final), "the test needs masks that changed after the best epoch"
    for name in final:
        torch.testing.assert_close(final[name], seen[best]["masks"][name],
                                   rtol=0, atol=0)


def test_saved_checkpoint_keeps_the_best_epochs_masks(monkeypatch, tmp_path):
    model, seen = _train(monkeypatch, JUMP, n_epochs_all_gps=4,
                         prune_aware=True, cut_mask_at=6)
    best = model.trainer.best_epoch
    model.save(str(tmp_path), overwrite=True, save_adata=True)
    loaded = NicheCompass.load(str(tmp_path), adata_file_name="adata.h5ad")
    for name, mask in _masks(loaded).items():
        torch.testing.assert_close(mask, seen[best]["masks"][name],
                                   rtol=0, atol=0)


# Rising from the start: early stopping would end training long before
# pruning (epoch 6) if the warm-up counted. A value equal to the best counts
# as no worse, so the warm-up has to rise to count as not improving.
RISING = [5., 5.1, 5.2, 5.3, 5.4, 5.5, 6., 5.5, 5.8, 5.9, 6.0]


def test_previous_behaviour_can_stop_before_pruning_ever_starts(monkeypatch):
    model, seen = _train(monkeypatch, RISING, n_epochs_all_gps=6,
                         prune_aware=False, patience=2, lr_patience=1)
    assert model.trainer.epoch == 2
    assert not any(s["pruning"] for s in seen.values())
    assert seen[2]["lr"] < seen[0]["lr"]           # cut during the warm-up


def test_the_warm_up_neither_stops_nor_cuts_the_learning_rate(monkeypatch):
    model, seen = _train(monkeypatch, RISING, n_epochs_all_gps=6,
                         prune_aware=True, patience=2, lr_patience=1)
    trainer = model.trainer
    assert trainer.epoch >= 6                      # pruning was reached
    assert {seen[e]["lr"] for e in range(7)} == {seen[0]["lr"]}
    assert trainer.best_epoch >= 6


def test_without_scheduled_pruning_early_stopping_is_unchanged(monkeypatch):
    """With ´n_epochs_all_gps >= n_epochs´ nothing is pruned, so there is
    no warm-up to protect and both settings stop at the same epoch."""
    epochs = {}
    for prune_aware in (False, True):
        model, _ = _train(monkeypatch, RISING, n_epochs_all_gps=len(RISING),
                          prune_aware=prune_aware, patience=2, lr_patience=1)
        epochs[prune_aware] = (model.trainer.epoch, model.trainer.best_epoch)
    assert epochs[True] == epochs[False] == (2, 0)


def test_restore_is_in_place_and_refuses_a_changed_model():
    model = _model()
    buffer = model.model.target_rna_dynamic_decoder_mask
    saved = {"target_rna_dynamic_decoder_mask": torch.zeros_like(buffer)}
    trainer_module._restore_buffers(model.model, saved)
    assert model.model.target_rna_dynamic_decoder_mask is buffer
    assert not buffer.any()
    with pytest.raises(ValueError, match="no longer has it"):
        trainer_module._restore_buffers(
            model.model, {"target_rna_dynamic_decoder_mask":
                          torch.zeros(buffer.shape[0] + 1, buffer.shape[1],
                                      dtype=buffer.dtype)})


def test_a_second_train_call_starts_afresh_at_pruning(monkeypatch):
    """The reset when pruning starts keeps a first call's best state and best
    value from carrying over into a second ´train´ call on the same
    trainer."""
    curves = {1: JUMP, 2: [30, 29, 28, 27, 22, 21, 21.5, 21.6, 21.7, 21.8,
                           21.9, 22.0]}
    call = {"n": 1}
    original = trainer_module.Trainer.is_early_stopping

    def scripted(self):
        metric = self.early_stopping.early_stopping_metric
        self.epoch_logs[metric][-1] = curves[call["n"]][self.epoch]
        return original(self)

    monkeypatch.setattr(trainer_module.Trainer, "is_early_stopping", scripted)
    model = _model()
    model.train(n_epochs=len(JUMP), n_epochs_all_gps=4, edge_val_ratio=0,
                node_val_ratio=0, node_batch_size=8, edge_batch_size=8,
                use_cuda_if_available=False, lambda_l1_addon=0,
                lambda_edge_recon=1, lambda_gene_expr_recon=1, monitor=False,
                early_stopping_kwargs={"patience": 3, "lr_patience": 2})
    trainer = model.trainer
    call["n"] = 2
    trainer.train(n_epochs=len(JUMP), n_epochs_all_gps=4)
    # Without the reset the first call's best value of 11 would stand, the
    # second call would stop one epoch after pruning began and reload the
    # first call's state.
    assert trainer.early_stopping.best_performance_state == 21
    assert trainer.best_epoch == 5
    assert trainer.epoch == 8


def test_a_missing_metric_still_fails_in_the_first_epoch():
    """The warm-up does not hide a misconfigured metric until pruning."""
    model = _model()
    with pytest.raises(IndexError):
        model.train(n_epochs=6, n_epochs_all_gps=4, edge_val_ratio=0,
                    node_val_ratio=0, node_batch_size=8, edge_batch_size=8,
                    use_cuda_if_available=False, lambda_l1_addon=0,
                    lambda_edge_recon=1, lambda_gene_expr_recon=1,
                    monitor=False, early_stopping_kwargs={
                        "early_stopping_metric": "val_global_loss"})
    assert model.trainer.epoch == 0


def test_reconstructed_feature_lists_follow_the_restored_masks(monkeypatch):
    """Which genes are reconstructed is derived from the masks. Without
    add-on programs, cutting the only program that reconstructs a gene drops
    that gene; restoring the best epoch's masks must bring it back, also when
    no forward pass follows the restore (no validation set)."""
    model = _model(n_addon_gp=0)
    names = list(model.adata.uns[model.gp_names_key_])
    row = names.index("c")                         # the only program for C
    original = trainer_module.Trainer.is_early_stopping

    def scripted(self):
        metric = self.early_stopping.early_stopping_metric
        self.epoch_logs[metric][-1] = JUMP[self.epoch]
        if self.epoch == 6:
            with torch.no_grad():
                for name, buffer in self.model.named_buffers():
                    if "dynamic_decoder_mask" in name:
                        buffer[row, :] = 0
        return original(self)

    monkeypatch.setattr(trainer_module.Trainer, "is_early_stopping", scripted)
    model.train(n_epochs=len(JUMP), n_epochs_all_gps=4, edge_val_ratio=0,
                node_val_ratio=0, node_batch_size=8, edge_batch_size=8,
                use_cuda_if_available=False, lambda_l1_addon=0,
                lambda_edge_recon=1, lambda_gene_expr_recon=1, monitor=False,
                early_stopping_kwargs={"patience": 3, "lr_patience": 2})
    module = model.model
    assert module.target_rna_dynamic_decoder_mask[row].any(), (
        "the test needs program c to be connected again after the restore")
    for side in ("target", "source"):
        implied = torch.nonzero(
            (getattr(module, f"{side}_rna_decoder_mask")
             * getattr(module, f"{side}_rna_dynamic_decoder_mask")).sum(0)
        ).flatten().tolist()
        assert module.features_idx_dict_[
            f"{side}_reconstructed_rna_idx"] == implied
