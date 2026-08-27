"""
Tests for the classification of the human protein-protein interactions that
back the human PPI gene programs. All tests here are offline: they exercise the
classification helpers directly and never download anything.
"""

import pytest

from nichecompass.utils.gene_programs import (
    _classify_humanppi_interaction,
    _classify_humanppi_protein_location,
    _humanppi_cis_complex_gene_families,
    _is_humanppi_ig_tcr_segment,
    _is_humanppi_paralog_cross_pair)


LOCATION_CLASSES = ["cell_surface", "secreted", "intracellular", "ambiguous",
                    "unknown"]


def anchored(is_membrane_anchored=True, has_topological_domain=False,
             has_extracellular_domain=False):
    """Build a topology record as returned by the UniProt lookup."""
    return {"is_membrane_anchored": is_membrane_anchored,
            "has_topological_domain": has_topological_domain,
            "has_extracellular_domain": has_extracellular_domain}


###############################################################################
## Protein location classification ##
###############################################################################

@pytest.mark.parametrize("localization,expected", [
    # An extracellular face beats an intracellular location
    ("Cytoplasm,Nucleus,Secreted", "secreted"),
    ("Cell membrane,Membrane", "cell_surface"),
    # Membrane anchoring beats secretion, so a shed soluble isoform does not
    # turn a receptor into a ligand (programmed cell death 1 ligand 1)
    ("Cell membrane,Endosome,Membrane,Nucleus,Secreted", "cell_surface"),
    # Antibody chains are the exception, since the secreted form dominates
    ("Cell membrane,Immunoglobulin,Membrane,Secreted", "secreted"),
    # Compartment keywords with a large cytoplasmic component do not establish
    # an extracellular face (tight junction plaque proteins such as ZO-1)
    ("Cell junction,Cytoplasm,Tight junction", "intracellular"),
    ("Membrane,Mitochondrion,Mitochondrion inner membrane", "intracellular"),
    # The generic membrane keyword alone is only compatible with a surface
    ("Membrane", "ambiguous"),
    ("Cell junction", "ambiguous"),
    # Surface complexes are surface
    ("MHC I", "cell_surface"),
    ("MHC II", "cell_surface"),
    ("T cell receptor", "cell_surface"),
    # The UniProt 'Exosome' keyword is the exoribonuclease complex, not vesicles
    ("Exosome", "intracellular"),
    # Missing annotation is not evidence of an intracellular location
    ("none", "unknown"),
    (None, "unknown"),
])
def test_protein_location_from_keywords(localization, expected):
    assert _classify_humanppi_protein_location(localization) == expected


def test_topology_rejects_a_cell_surface_keyword_without_a_membrane_anchor():
    # SNAP25 is a soluble SNARE docked onto the cytoplasmic leaflet, yet it
    # carries the 'Cell membrane' keyword
    snap25 = "Cell membrane,Cytoplasm,Membrane,Synapse"
    assert _classify_humanppi_protein_location(snap25) == "cell_surface"
    assert _classify_humanppi_protein_location(
        snap25, topology=anchored(is_membrane_anchored=False)) == (
            "intracellular")


def test_topology_rejects_a_cell_surface_keyword_without_an_ecto_domain():
    # A membrane protein whose annotated topology is exclusively cytoplasmic
    assert _classify_humanppi_protein_location(
        "Cell membrane,Membrane",
        topology=anchored(has_topological_domain=True,
                          has_extracellular_domain=False)) == "intracellular"
    assert _classify_humanppi_protein_location(
        "Cell membrane,Membrane",
        topology=anchored(has_topological_domain=True,
                          has_extracellular_domain=True)) == "cell_surface"


def test_topology_promotes_a_generic_membrane_keyword():
    # Interleukin 15 receptor subunit alpha is never annotated 'Cell membrane'
    il15ra = "Cytoplasmic vesicle,Endoplasmic reticulum,Membrane,Secreted"
    assert _classify_humanppi_protein_location(il15ra) == "secreted"
    # An annotated extracellular topological domain is decisive
    assert _classify_humanppi_protein_location(
        il15ra, topology=anchored(has_topological_domain=True,
                                  has_extracellular_domain=True)) == (
                                      "cell_surface")
    # A membrane anchor alone is not enough here, since the intracellular
    # keywords are equally compatible with an organelle membrane protein
    assert _classify_humanppi_protein_location(
        il15ra, topology=anchored()) == "secreted"


def test_secretion_is_trusted_without_a_membrane_anchor():
    # Non-classical secretion leaves no sequence signature
    assert _classify_humanppi_protein_location(
        "Cytoplasm,Secreted",
        topology=anchored(is_membrane_anchored=False)) == "secreted"


def test_process_keywords_are_only_a_fallback_for_missing_localization():
    assert _classify_humanppi_protein_location(
        "none", process="Keratinization") == "intracellular"
    assert _classify_humanppi_protein_location(
        "none", process="Immunity") == "unknown"
    # Never override an existing cellular component keyword
    assert _classify_humanppi_protein_location(
        "Secreted", process="Transcription") == "secreted"


###############################################################################
## Interaction classification ##
###############################################################################

@pytest.mark.parametrize("location_1,location_2,localization_filter,expected", [
    ("cell_surface", "cell_surface", "strict", "juxtacrine"),
    ("secreted", "cell_surface", "strict", "paracrine"),
    ("secreted", "secreted", "strict", "paracrine"),
    ("cell_surface", "intracellular", "strict", "intracellular"),
    ("cell_surface", "ambiguous", "strict", "intracellular"),
    ("cell_surface", "ambiguous", "include_ambiguous", "juxtacrine"),
    ("secreted", "ambiguous", "include_ambiguous", "paracrine"),
    ("ambiguous", "ambiguous", "include_ambiguous", "juxtacrine"),
    ("ambiguous", "intracellular", "include_ambiguous", "intracellular"),
    ("unknown", "cell_surface", "include_ambiguous", "unknown"),
    ("unknown", "unknown", "strict", "unknown"),
])
def test_interaction_classification(location_1, location_2,
                                    localization_filter, expected):
    assert _classify_humanppi_interaction(
        location_1, location_2, localization_filter) == expected


def test_interaction_classification_is_symmetric():
    for location_1 in LOCATION_CLASSES:
        for location_2 in LOCATION_CLASSES:
            for localization_filter in ["strict", "include_ambiguous"]:
                assert (_classify_humanppi_interaction(
                            location_1, location_2, localization_filter) ==
                        _classify_humanppi_interaction(
                            location_2, location_1, localization_filter))


def test_interaction_classification_rejects_raw_localization_strings():
    with pytest.raises(ValueError, match="not a location class"):
        _classify_humanppi_interaction("Cell membrane,Membrane",
                                       "cell_surface",
                                       "strict")


@pytest.mark.parametrize("localization_filter",
                         ["surface_secreted", "membrane_strict", "membrane",
                          "all", ""])
def test_interaction_classification_rejects_unknown_filters(
        localization_filter):
    with pytest.raises(ValueError, match="localization_filter"):
        _classify_humanppi_interaction("cell_surface", "cell_surface",
                                       localization_filter)


###############################################################################
## Immunoglobulin and T cell receptor gene segments ##
###############################################################################

@pytest.mark.parametrize("gene", [
    "IGHV4-30-2", "IGLV10-54", "IGKV2-30", "IGHJ4", "IGHD3-3", "TRAV8-7",
    "TRBV6-1", "TRDV1", "TRBD1"])
def test_ig_tcr_segments_are_detected(gene):
    assert _is_humanppi_ig_tcr_segment(gene)


@pytest.mark.parametrize("gene", [
    "IGHG1", "IGHM", "IGHD", "IGLC2", "IGKC", "TRAC", "TRBC1", "TRBC2",
    "TRDC", "CD3E", "PDCD1", "CD274"])
def test_complete_proteins_are_not_mistaken_for_segments(gene):
    assert not _is_humanppi_ig_tcr_segment(gene)


###############################################################################
## Paralogue combinations that do not form ##
###############################################################################

@pytest.mark.parametrize("gene_1,gene_2", [
    ("ITGB4", "ITGAL"), ("ITGB6", "ITGAL"), ("ITGA5", "ITGB5"),
    ("HLA-DRA", "HLA-DQB1"), ("HLA-DQB2", "HLA-DPA1")])
def test_paralog_cross_pairs_are_detected(gene_1, gene_2):
    assert _is_humanppi_paralog_cross_pair(gene_1, gene_2)


@pytest.mark.parametrize("gene_1,gene_2", [
    ("ITGAL", "ITGB2"), ("ITGA4", "ITGB1"), ("ITGAV", "ITGB3"),
    ("HLA-DRA", "HLA-DRB1"), ("HLA-DQA1", "HLA-DQB1"),
    ("PDCD1", "CD274"), ("CD3D", "CD3E")])
def test_real_heterodimers_are_kept(gene_1, gene_2):
    assert not _is_humanppi_paralog_cross_pair(gene_1, gene_2)


###############################################################################
## Curated cis complex families ##
###############################################################################

@pytest.mark.parametrize("gene_1,gene_2", [
    ("CD3D", "CD3E"), ("CD3E", "CD247"), ("CD8A", "CD8B"),
    ("CD79A", "CD79B"), ("HLA-DRA", "HLA-DRB1"), ("HLA-A", "B2M"),
    ("KLRC1", "KLRD1"), ("ITGAL", "ITGB2"), ("FCER1A", "FCER1G")])
def test_curated_cis_families_share_a_family(gene_1, gene_2):
    assert (_humanppi_cis_complex_gene_families(gene_1) &
            _humanppi_cis_complex_gene_families(gene_2))


@pytest.mark.parametrize("gene_1,gene_2", [
    ("PDCD1", "CD274"), ("SIRPA", "CD47"), ("TNF", "TNFRSF1A"),
    ("IL15", "IL15RA"), ("EPHB1", "EFNB3")])
def test_ligand_receptor_pairs_do_not_share_a_family(gene_1, gene_2):
    assert not (_humanppi_cis_complex_gene_families(gene_1) &
                _humanppi_cis_complex_gene_families(gene_2))
