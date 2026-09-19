"""Versioned, non-mutating views of gene-program activity and loadings."""

import hashlib
import json
import uuid
from copy import deepcopy

import numpy as np
import pandas as pd
import scipy.sparse as sp


def _array(value):
    if getattr(value, "is_sparse", False):
        value = value.to_dense()
    return value.detach().cpu().numpy()


def _digest(*values):
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            digest.update(str((value.shape, value.dtype)).encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        else:
            digest.update(json.dumps(value, sort_keys=True).encode())
        digest.update(b"\0")
    return digest.hexdigest()


def _importance(weights):
    mass = np.abs(weights)
    return np.divide(mass, mass.sum(0), out=np.zeros_like(mass), where=mass.sum(0) != 0)


class GPAnalysisMixin:
    """Keep model coordinates raw and share one orientation across analyses."""

    def _gp_validate_features(self, adata, modality="rna"):
        expected = getattr(self, "gp_analysis_feature_names_", {}).get(modality)
        if expected is not None and list(map(str, adata.var_names)) != list(expected):
            raise ValueError(f"Analysis {modality.upper()} features must match the fitted model's "
                             "feature names and order. Reorder the data before analysis.")

    def _gp_input_fingerprint(self, adata, cat_key, adata_atac=None):
        """Fingerprint model and inference inputs, independent of GPU roundoff.

        Hash components sequentially instead of retaining copies of all model
        tensors or atlas matrices. Score columns are excluded from obs inputs.
        """
        def matrix_digest(value):
            if sp.issparse(value):
                value = value.tocsr()
                return _digest(value.shape, value.data, value.indices, value.indptr)
            return _digest(np.asarray(value))

        parts = [(name, _digest(_array(value)))
                 for name, value in self.model.named_parameters()]
        # ´frozen_gp_statistic_mask´ is a freeze control, not an inference
        # input: it is non-persistent, and ´load´ derives it from the
        # requested unfreeze configuration rather than from the checkpoint.
        # Hashing it made every reloaded model disagree with the
        # fingerprint stored beside its own results, so saved results were
        # always reported stale after a reload.
        parts.extend((name, _digest(_array(value)))
                     for name, value in self.model.named_buffers()
                     if name != "frozen_gp_statistic_mask")
        keys = list(dict.fromkeys([cat_key] + list(self.cat_covariates_keys_ or [])))
        labels = pd.util.hash_pandas_object(adata.obs[keys], index=True, categorize=False).to_numpy()
        parts.extend([("obs", _digest(keys, list(map(str, adata.obs_names)), labels)),
                      ("adjacency", matrix_digest(adata.obsp[self.adj_key_])),
                      ("configuration", _digest(self.counts_key_, self.cat_covariates_cats_,
                         self.model.log_variational_, self.model.training))])
        datasets = {"rna": adata}
        if "atac" in self.modalities_:
            datasets["atac"] = self.adata_atac[adata.obs_names] if adata_atac is None else adata_atac
        for modality, data in datasets.items():
            counts = data.X if self.counts_key_ is None else data.layers[self.counts_key_]
            parts.append((modality, _digest(list(map(str, data.var_names)),
                                            list(map(str, data.obs_names)), matrix_digest(counts))))
        return _digest(parts)

    def _gp_selection(self, selected_gps=None, active=False):
        names = list(map(str, self.adata.uns[self.gp_names_key_]))
        if len(set(names)) != len(names):
            raise ValueError("Gene program names must be unique.")
        if len(names) != self.n_prior_gp_ + self.n_addon_gp_:
            raise ValueError("GP names must match the model's latent columns. "
                             "Resolve legacy GP name aliases before analysis.")
        if selected_gps is None:
            selected_gps = list(self.get_active_gps()) if active else names
        elif isinstance(selected_gps, str):
            selected_gps = [selected_gps]
        selected_gps = list(selected_gps)
        if len(set(selected_gps)) != len(selected_gps):
            raise ValueError("Selected gene programs must be unique.")
        unknown = set(selected_gps) - set(names)
        if unknown:
            raise ValueError(f"Unknown gene programs: {sorted(unknown)}")
        return selected_gps, np.array([names.index(gp) for gp in selected_gps], dtype=int)

    def _gp_arrays(self):
        self._gp_validate_features(self.adata)
        if "atac" in self.modalities_:
            self._gp_validate_features(self.adata_atac, "atac")
        if not hasattr(self.model.target_rna_decoder.nb_means_normalized_decoder, "masked_l"):
            raise ValueError("GP loading analysis requires masked linear decoders.")
        weights, memberships = {}, {}
        for modality, raw in zip(self.modalities_, self.model.get_gp_weights()):
            blocks, effective = [], []
            for entity in ("target", "source"):
                decoder = getattr(self.model, f"{entity}_{modality}_decoder")
                layer = decoder.nb_means_normalized_decoder
                mask = _array(layer.masked_l.mask).astype(bool)
                if self.n_addon_gp_:
                    mask = np.concatenate([mask, _array(layer.addon_l.mask).astype(bool)], axis=1)
                blocks.append(mask)
                # Only ATAC decoders apply a dynamic feature mask in forward.
                # RNA pruning acts on latent activity, not these loadings.
                if modality == "atac":
                    dynamic = _array(getattr(self.model, f"{entity}_{modality}_dynamic_decoder_mask")).T
                    effective.append(mask & dynamic.astype(bool))
                else:
                    effective.append(mask)
            memberships[modality] = np.concatenate(blocks)
            weights[modality] = _array(raw) * np.concatenate(effective)
            if not np.isfinite(weights[modality]).all():
                raise ValueError(f"Non-finite {modality} decoder weights.")
        return weights, memberships

    def _gp_snapshot(self, weights, memberships):
        names, _ = self._gp_selection()
        features = {"rna": list(map(str, self.adata.var_names))}
        if "atac" in self.modalities_:
            features["atac"] = list(map(str, self.adata_atac.var_names))
        return _digest(names, features, *[x for m in self.modalities_
                       for x in (weights[m], memberships[m])])

    def prepare_gp_analysis(self, orientation="paper_2025", overwrite=False):
        """Initialize or explicitly refresh canonical GP activity analysis.

        The paper policy flips a prior GP for a negative source RNA sum,
        falling back to targets only when the measured source membership is
        empty. Add-ons use the sum of both RNA components. Zero sums stay +1.
        Scores, model parameters and existing result tables are not modified.
        Calling this on a legacy model opts its high-level APIs into canonical
        analysis. Changed decoder weights require ``overwrite=True``; frozen
        reference mappings inherit signs for matching GP definitions.

        Returns a copy of the orientation/quality table. Coherence is a loading
        balance diagnostic, not a probability of biological activation.
        """
        if orientation != "paper_2025":
            raise ValueError("Supported orientation policy: 'paper_2025'.")
        self._check_if_trained(warn=False)
        weights, memberships = self._gp_arrays()
        snapshot = self._gp_snapshot(weights, memberships)
        old = getattr(self, "gp_analysis_", None)
        if old is not None and old["decoder_fingerprint"] == snapshot:
            self.gp_analysis_default_orientation_ = "canonical"
            self.adata.uns["nichecompass_gp_analysis"] = deepcopy(old)
            return old["table"].copy()
        if old is not None and not overwrite:
            raise ValueError("GP analysis is stale after decoder/feature changes. "
                             "Call prepare_gp_analysis(overwrite=True), then rerun analyses.")
        if not hasattr(self, "gp_analysis_model_id_"):
            self.gp_analysis_model_id_ = uuid.uuid4().hex
        names, _ = self._gp_selection()
        rna = weights["rna"]
        n_genes = self.adata.n_vars
        genes = list(map(str, self.adata.var_names))
        rows = []
        old_rows = {} if old is None else old["table"].set_index("gp_id").to_dict("index")
        n_prior = len(names) - self.n_addon_gp_
        for k, name in enumerate(names):
            definition = [[entity, genes[g]] for block, entity in enumerate(("target", "source"))
                          for g in np.flatnonzero(memberships["rna"][block*n_genes:(block+1)*n_genes, k])]
            gp_id = _digest(name, sorted(definition),
                            "prior" if k < n_prior else self.gp_analysis_model_id_)
            target, source = rna[:n_genes, k], rna[n_genes:, k]
            if k >= n_prior:
                anchor, policy = rna[:, k], "all_rna"
            elif memberships["rna"][n_genes:, k].any():
                anchor, policy = source, "source"
            else:
                anchor, policy = target, "target_fallback"
            signed_mass = float(anchor.sum(dtype=np.float64))
            mass = float(np.abs(anchor).sum(dtype=np.float64))
            coherence = abs(signed_mass) / mass if mass else 0.0
            sign = -1 if signed_mass < 0 else 1
            # ´freeze_´ is a whole model flag and stays True under partial
            # unfreezing, so for the ADD-ON block it does not establish that
            # this program's loadings are unchanged: they move under
            # ´unfreeze_addon_gp_weights´, which leaves ´freeze_´ True. When
            # an inherited sign contradicts the sign those loadings now imply,
            # the loadings moved, so the recomputed sign wins rather than
            # returning activities flipped against their own weights.
            #
            # Scoped to the add-on block deliberately. A PRIOR program's
            # loadings can only move under ´unfreeze_all_weights´, which
            # clears ´freeze_´, so a disagreement there is unreachable through
            # the API and inheriting is the documented contract - a frozen
            # reference keeps its signs, which is what makes reference and
            # query scores comparable.
            inherited = bool(getattr(self, "freeze_", False) and gp_id in old_rows)
            # 0 rather than None: this table is written to ´adata.uns´ and on
            # to HDF5, and a column mixing None with ints is an object column
            # that h5py refuses, which broke every ´save´ of a model carrying
            # a gene program analysis table. The sign is only ever +/-1, so 0
            # is an unambiguous "not inherited".
            inherited_sign = (int(old_rows[gp_id]["orientation_sign"])
                              if inherited else 0)
            sign_disagrees = bool(inherited and k >= n_prior and mass > 0
                                  and inherited_sign != sign)
            if sign_disagrees:
                inherited = False
            elif inherited:
                sign = inherited_sign
            status = ("zero_support" if mass == 0 else
                      "mixed" if coherence < 0.1 else
                      "weak_support" if mass < 1e-8 else "oriented")
            rows.append({"gp_id": gp_id, "gp_name": name, "all_gp_idx": k,
                         "gp_type": "prior" if k < n_prior else "addon",
                         "orientation_sign": sign, "anchor": policy,
                         "signed_mass": signed_mass, "anchor_mass": mass,
                         "coherence": coherence, "orientation_status": status,
                         "inherited": inherited,
                         "inherited_sign": inherited_sign,
                         "inherited_sign_disagreed": sign_disagrees,
                         "source_target_disagree": bool(np.sign(source.sum(dtype=np.float64)) *
                                                         np.sign(target.sum(dtype=np.float64)) < 0),
                         "n_source_genes": int(memberships["rna"][n_genes:, k].sum()),
                         "n_target_genes": int(memberships["rna"][:n_genes, k].sum())})
        table = pd.DataFrame(rows)
        self.gp_analysis_ = {"schema_version": 1, "policy": orientation,
                             "decoder_fingerprint": snapshot,
                             "orientation_id": _digest(orientation, snapshot, table.to_dict("list")),
                             "table": table}
        self.gp_analysis_default_orientation_ = "canonical"
        self.adata.uns["nichecompass_gp_analysis"] = deepcopy(self.gp_analysis_)
        return table.copy()

    def _gp_orientation(self, orientation=None):
        orientation = orientation or getattr(self, "gp_analysis_default_orientation_", "raw")
        if orientation not in ("raw", "canonical"):
            raise ValueError("orientation must be 'raw' or 'canonical'.")
        return orientation

    def _gp_signs(self, selected_gps=None, orientation=None):
        names, indices = self._gp_selection(selected_gps)
        if self._gp_orientation(orientation) == "raw":
            return np.ones(len(names))
        table = self._gp_analysis_table()
        return table.set_index("gp_name").loc[names, "orientation_sign"].to_numpy()

    def _gp_analysis_table(self):
        """Ensure metadata without turning a per-call override into a migration."""
        default = self._gp_orientation()
        try:
            return self.prepare_gp_analysis()
        finally:
            self.gp_analysis_default_orientation_ = default

    def get_gp_quality_table(self):
        """Return orientation diagnostics plus current global pruning status."""
        table = self._gp_analysis_table()
        table["gp_active"] = table.gp_name.isin(self.get_active_gps())
        return table

    def get_gp_activities(self, selected_gps=None, adata=None, return_std=False,
                          orientation=None, use_cached=False, adata_atac=None):
        """Return GP activity columns in requested name order (active by default).

        Canonical means use the persisted RNA-derived sign; standard deviations
        are unchanged. Raw ``get_latent_representation`` is never redefined.
        Inference is fresh unless ``use_cached=True`` explicitly requests the
        active raw cache. Cached data cannot supply posterior deviations.
        """
        names, indices = self._gp_selection(selected_gps, active=True)
        adata = self.adata if adata is None else adata
        self._gp_validate_features(adata)
        if not adata.var_names.equals(self.adata.var_names):
            raise ValueError("Analysis features must match model feature names and order.")
        if "atac" in self.modalities_:
            if adata_atac is None:
                if not adata.obs_names.isin(self.adata_atac.obs_names).all():
                    raise ValueError("Provide aligned adata_atac for new multimodal observations.")
                adata_atac = self.adata_atac[adata.obs_names]
            if (not adata_atac.var_names.equals(self.adata_atac.var_names)
                    or not adata_atac.obs_names.equals(adata.obs_names)):
                raise ValueError("ATAC features and observation order must match the analysis data.")
            self._gp_validate_features(adata_atac, "atac")
        signs = self._gp_signs(names, orientation)
        if use_cached:
            if return_std:
                raise ValueError("Cached posterior standard deviations are unavailable.")
            active = list(adata.uns[self.active_gp_names_key_])
            raw = adata.obsm[self.latent_key_]
            if raw.shape != (adata.n_obs, len(active)) or len(active) != len(set(active)):
                raise ValueError("Invalid raw GP cache/name alignment.")
            if not set(names).issubset(active):
                raise ValueError("Requested GP is not in the active raw cache.")
            means = np.asarray(raw)[:, [active.index(name) for name in names]]
            return means * signs
        means, std = self.get_latent_representation(
            adata=adata, adata_atac=adata_atac, counts_key=self.counts_key_, adj_key=self.adj_key_,
            cat_covariates_keys=self.cat_covariates_keys_, only_active_gps=False,
            return_mu_std=True, node_batch_size=self.node_batch_size_, selected_gps=names)
        means = means * signs
        return (means, std) if return_std else means

    def _write_gp_scores(self, adata, names, scores, orientation):
        """Record provenance per column when callers write different subsets."""
        orientation = self._gp_orientation(orientation)
        orientation_id = self.gp_analysis_["orientation_id"] if orientation == "canonical" else "raw"
        metadata = adata.uns.get("nichecompass_gp_score_columns", {})
        columns = ["gp_name", "orientation", "orientation_id"]
        previous = metadata.get("table")
        if previous is None:
            previous = pd.DataFrame([
                {"gp_name": name, "orientation": metadata.get("orientation", "raw"),
                 "orientation_id": metadata.get("orientation_id", "raw")}
                for name in metadata.get("gp_names", [])], columns=columns)
        previous = previous[previous.gp_name.isin(adata.obs.columns) & ~previous.gp_name.isin(names)]
        current = pd.DataFrame({"gp_name": names, "orientation": orientation,
                                "orientation_id": orientation_id}, columns=columns)
        table = pd.concat([previous, current], ignore_index=True)
        adata.obs[names] = scores
        adata.uns["nichecompass_gp_score_columns"] = {
            "schema_version": 1, "table": table, "gp_names": table.gp_name.tolist(),
            "orientation": orientation if table.orientation.nunique() <= 1 else "mixed",
            "orientation_id": orientation_id if table.orientation_id.nunique() <= 1 else "mixed"}

    def get_gp_feature_table(self, selected_gps=None, orientation=None):
        """Full-precision member loadings and signed arms; no expression effects.

        Importance is absolute loading mass normalized jointly across source
        and target within each modality. It is not variance explained.
        Includes masked members with zero learned weight, for coverage auditing.
        """
        names, indices = self._gp_selection(selected_gps)
        orientation = self._gp_orientation(orientation)
        signs = self._gp_signs(names, orientation)
        weights, memberships = self._gp_arrays()
        rows = []
        for modality in self.modalities_:
            features = self.adata.var_names if modality == "rna" else self.adata_atac.var_names
            n_features = len(features)
            raw = weights[modality][:, indices]
            oriented = raw * signs
            importance = _importance(raw)
            for k, (name, index) in enumerate(zip(names, indices)):
                for f in np.flatnonzero(memberships[modality][:, index]):
                    rows.append({"gp_name": name, "feature": str(features[f % n_features]),
                                 "modality": modality, "entity": "target" if f < n_features else "source",
                                 "raw_loading": float(raw[f, k]), "loading": float(oriented[f, k]),
                                 "importance": float(importance[f, k]),
                                 "arm": "positive" if oriented[f, k] > 0 else "negative" if oriented[f, k] < 0 else "zero",
                                 "orientation": orientation, "orientation_sign": int(signs[k])})
        return pd.DataFrame(rows, columns=["gp_name", "feature", "modality", "entity", "raw_loading",
                                           "loading", "importance", "arm", "orientation", "orientation_sign"])

    def _gp_summary(self, orientation=None):
        """Build the established wide summary from the common feature table."""
        features = self.get_gp_feature_table(orientation=orientation)
        names, _ = self._gp_selection()
        active = list(self.get_active_gps())
        groups = {key: block.sort_values(["importance", "feature"],
                                         ascending=[False, True], kind="stable")
                  for key, block in features.groupby(["gp_name", "modality", "entity"], sort=False)}
        rows = []
        for k, name in enumerate(names):
            row = {"gp_name": name, "all_gp_idx": k, "gp_active": name in active,
                   "active_gp_idx": active.index(name) if name in active else -1}
            for modality, unit in (("rna", "genes"), ("atac", "peaks")):
                if modality not in self.modalities_:
                    continue
                for entity in ("source", "target"):
                    block = groups.get((name, modality, entity), features.iloc[:0])
                    row[f"n_{entity}_{unit}"] = len(block)
                    row[f"n_non_zero_{entity}_{unit}"] = int((block.loading != 0).sum())
                    row[f"gp_{entity}_{unit}"] = block.feature.tolist()
                    row[f"gp_{entity}_{unit}_weights"] = block.loading.tolist()
                    row[f"gp_{entity}_{unit}_importances"] = block.importance.tolist()
            rows.append(row)
        result = pd.DataFrame(rows)
        result["active_gp_idx"] = result.active_gp_idx.replace(-1, pd.NA).astype("Int64")
        if self._gp_orientation(orientation) == "canonical":
            quality = self._gp_analysis_table()
            result = result.merge(quality.drop(columns=["all_gp_idx", "n_source_genes", "n_target_genes"]),
                                  on="gp_name", validate="one_to_one", sort=False)
        return result
