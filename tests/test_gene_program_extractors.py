"""Prior gene program extractors that read a downloaded table, run offline."""

import numpy as np
import pandas as pd

import nichecompass.utils.gene_programs as gene_programs
from nichecompass.utils import extract_gp_dict_from_omnipath_lr_interactions


def _intercell_table():
    """A small OmniPath intercell table with the cases the download can hold."""
    rows = [
        # source accession, target accession, ligand, receptor, curation effort
        ("P60568", "P01589", "IL2", "IL2RA", 3),
        ("P60568", "A0A2R8YE73", "IL2", np.nan, 3),       # receptor without a symbol
        ("P60568", "A0A2R8YE74", "IL2", None, 3),         # the same, as None
        ("P60568", "COMPLEX:P14784_P31785", "IL2", "COMPLEX:IL2RB_IL2RG", 4),
        ("E7END6", "Q9UNN8", None, "PROCR", 3),            # ligand without a symbol
        ("P01137", "P36897", "TGFB1", "TGFBR1", 1),        # below the curation cutoff
    ]
    table = pd.DataFrame(rows, columns=["source", "target", "genesymbol_intercell_source",
                                        "genesymbol_intercell_target", "curation_effort"])
    table["category_intercell_source"] = "ligand"
    table["category_intercell_target"] = "receptor"
    return table


def test_omnipath_programs_skip_proteins_without_a_gene_symbol(monkeypatch, tmp_path, capsys):
    """A receptor without a gene symbol used to raise TypeError when its missing
    symbol arrived as NaN, as it does from a table read back from disk."""
    monkeypatch.setattr(gene_programs.op.interactions, "import_intercell_network",
                        lambda include: _intercell_table())
    path = tmp_path / "omnipath_lr_network.csv"
    expected = {"IL2_ligand_receptor_GP"}
    downloaded = extract_gp_dict_from_omnipath_lr_interactions(
        species="human", load_from_disk=False, save_to_disk=True,
        lr_network_file_path=str(path), plot_gp_gene_count_distributions=False)
    assert "Dropped 3 OmniPath ligand-receptor interactions" in capsys.readouterr().out
    assert set(downloaded) == expected
    program = downloaded["IL2_ligand_receptor_GP"]
    assert program["sources"] == ["IL2"]
    assert sorted(program["targets"]) == ["IL2RA", "IL2RB", "IL2RG"]
    assert program["targets_categories"] == ["receptor"] * 3

    reloaded = extract_gp_dict_from_omnipath_lr_interactions(
        species="human", load_from_disk=True, lr_network_file_path=str(path),
        plot_gp_gene_count_distributions=False)
    assert set(reloaded) == expected
    assert sorted(reloaded["IL2_ligand_receptor_GP"]["targets"]) == ["IL2RA", "IL2RB", "IL2RG"]


def test_cell_type_aggregation_keeps_every_cell_type_in_every_group():
    """Labels in adata.obs are categorical, and pandas 3 groupby drops absent
    categories by default, so a group without some cell type lost its rows and
    the chord plot, which reads rows by position, paired the wrong cell types."""
    import anndata as ad
    import scipy.sparse as sp
    from nichecompass.utils import aggregate_obsp_matrix_per_cell_type

    adata = ad.AnnData(np.zeros((6, 1), dtype=np.float32))
    adata.obs["cell_type"] = pd.Categorical(["a", "a", "b", "b", "c", "c"],
                                            categories=["a", "b", "c", "unused"])
    adata.obs["niche"] = pd.Categorical(["n1", "n1", "n1", "n2", "n2", "n2"])
    # n1 holds a and b, n2 holds b and c: each niche lacks one cell type.
    adata.obsp["conn"] = sp.csr_matrix(np.ones((6, 6)) - np.eye(6))
    grouped = aggregate_obsp_matrix_per_cell_type(
        adata, obsp_key="conn", cell_type_key="cell_type", group_key="niche", agg_rows=True)
    assert list(grouped.columns) == ["a", "b", "c"]
    for niche in ("n1", "n2"):
        block = grouped.loc[niche]
        assert list(block.index) == ["a", "b", "c"]
    # n2 holds no cell of type a, so its row of sending a-cells is all zero.
    assert (grouped.loc["n2"].loc["a"] == 0).all()
    assert grouped.loc["n1"].loc["a", "b"] == 4.0   # two a-cells, each with two b-neighbours
    flat = aggregate_obsp_matrix_per_cell_type(
        adata, obsp_key="conn", cell_type_key="cell_type", agg_rows=True)
    assert list(flat.index) == ["a", "b", "c"] and list(flat.columns) == ["a", "b", "c"]
