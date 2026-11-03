"""
Benchmark of the communication network on a synthetic tissue with a known
answer.

A small sender niche A expresses the ligand L1 strongly; the larger receiver
niche B next to it expresses the receptor R1 and a little L1; a background
niche C expresses neither. Niche D, far away, expresses both L2 and R2, a
genuine signal within one niche. A NicheCompass model is trained on this
tissue, and ´compute_communication_gp_network´ should rank A -> B first for
the L1 program and D -> D first for the L2 program.

For comparison it also prints the ranking the 0.3.3 scoring gives on the same
graph: sender and receiver activity both entering, summed per pair of niches.

This is deliberately not a pytest test: it trains a model, which takes a
minute or two on a CPU. Run it manually after changing the communication
scoring:

    python tests/benchmark_communication.py

It ends with ´Benchmark passed.´ or names the ranking that failed.
"""

import contextlib
import io
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.spatial import cKDTree

from nichecompass.models import NicheCompass
from nichecompass.utils import (add_gps_from_gp_dict_to_adata,
                                compute_communication_gp_network)


def make_tissue(seed: int=0, n_cells: int=4000) -> ad.AnnData:
    """Build the synthetic tissue with a 6-nearest-neighbour symmetric graph."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 100, (n_cells, 2))
    niche = np.full(n_cells, "C", dtype=object)
    in_band = (xy[:, 0] > 30) & (xy[:, 0] < 70)
    niche[in_band & (xy[:, 1] < 50)] = "B"
    niche[in_band & (xy[:, 1] >= 50) & (xy[:, 1] < 58)] = "A"
    niche[(xy[:, 0] > 80) & (xy[:, 1] > 70)] = "D"
    genes = (["L1", "R1", "L2", "R2"]
             + [f"m{k}{i}" for k in "ABCD" for i in range(8)]
             + [f"bg{i}" for i in range(16)])
    means = {gene: np.full(n_cells, 0.3) for gene in genes}
    means["L1"][niche == "A"] = 20.
    means["L1"][niche == "B"] = 4.
    means["R1"][niche == "B"] = 15.
    means["L2"][niche == "D"] = 12.
    means["R2"][niche == "D"] = 12.
    for k in "ABCD":
        for i in range(8):
            means[f"m{k}{i}"][niche == k] = 8.
    for i in range(16):
        means[f"bg{i}"] = rng.uniform(1, 5) * np.ones(n_cells)
    counts = np.column_stack(
        [rng.poisson(means[gene]) for gene in genes]).astype(np.float32)
    adata = ad.AnnData(sp.csr_matrix(counts))
    adata.var_names = genes
    adata.obs["niche"] = pd.Categorical(niche)
    adata.obsm["spatial"] = xy
    adata.layers["counts"] = adata.X.copy()
    neighbors = cKDTree(xy).query(xy, k=7)[1][:, 1:]
    knn = sp.csr_matrix(
        (np.ones(neighbors.size, dtype=np.float32),
         (np.repeat(np.arange(n_cells), 6), neighbors.ravel())),
        shape=(n_cells, n_cells))
    adata.obsp["spatial_connectivities"] = knn.maximum(knn.T)
    programs = {"L1_R1_GP": (["L1"], ["R1"]), "L2_R2_GP": (["L2"], ["R2"])}
    for k in "ABCD":
        programs[f"marker_{k}_GP"] = ([], [f"m{k}{i}" for i in range(8)])
    gp_dict = {name: {"sources": sources, "targets": targets,
                      "sources_categories": ["ligand"] * len(sources),
                      "targets_categories": ["receptor"] * len(targets)}
               for name, (sources, targets) in programs.items()}
    add_gps_from_gp_dict_to_adata(gp_dict, adata, min_genes_per_gp=1,
                                  min_source_genes_per_gp=0,
                                  min_target_genes_per_gp=1)
    return adata


def old_ranking(model, gp: str) -> list:
    """Rank niche pairs as 0.3.3 scored them, on the model's own graph."""
    adata = model.adata
    sign = float(model._gp_signs([gp], "canonical")[0])
    activity = np.clip(sign * model.get_gp_activities(
        [gp], orientation="raw", use_cached=True)[:, 0], 0., None)
    sender = activity * adata.obs[f"{gp}_source_score"].to_numpy()
    receiver = adata.obs[f"{gp}_target_score"].to_numpy()
    graph = sp.coo_matrix(adata.obsp["spatial_connectivities"])
    keep = graph.row != graph.col
    rows, cols = graph.row[keep], graph.col[keep]
    pairs = pd.DataFrame({"source": adata.obs["niche"].to_numpy()[rows],
                          "target": adata.obs["niche"].to_numpy()[cols],
                          "value": sender[rows] * receiver[cols]})
    total = pairs.groupby(["source", "target"], observed=True)["value"].sum()
    return [f"{s}->{t}" for s, t in total.sort_values(ascending=False).index]


def main():
    warnings.filterwarnings("ignore")
    torch.manual_seed(0)
    adata = make_tissue()
    with contextlib.redirect_stdout(io.StringIO()):
        model = NicheCompass(adata, n_addon_gp=2, use_cuda_if_available=False)
        model.train(n_epochs=60, n_epochs_all_gps=10, lr=0.003,
                    use_cuda_if_available=False, verbose=False)
    within = (adata.obs["niche"].to_numpy()[
        sp.coo_matrix(adata.obsp["spatial_connectivities"]).row]
        == adata.obs["niche"].to_numpy()[
        sp.coo_matrix(adata.obsp["spatial_connectivities"]).col]).mean()
    print(f"Cells per niche: {adata.obs['niche'].value_counts().to_dict()}; "
          f"{within:.1%} of graph edges join two cells of one niche.")
    expected = {"L1_R1_GP": "A->B", "L2_R2_GP": "D->D"}
    failures = []
    for gp, top in expected.items():
        if gp not in model.get_active_gps():
            failures.append(f"{gp} was pruned")
            continue
        network = compute_communication_gp_network([gp], model, "niche")
        ranking = [f"{s}->{t}" for s, t in zip(network.source, network.target)]
        print(f"\n{gp}: expected first {top}")
        print(network[["source", "target", "strength", "mean_strength",
                       "n_edges", "enrichment"]].head(5).to_string(index=False))
        print(f"0.3.3 scoring ranks: {old_ranking(model, gp)[:5]}")
        if not ranking or ranking[0] != top:
            failures.append(f"{gp} ranks {ranking[:3]} first, not {top}")
    if failures:
        raise SystemExit("Benchmark failed: " + "; ".join(failures))
    print("\nBenchmark passed.")


if __name__ == "__main__":
    main()
