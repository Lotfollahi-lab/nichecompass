"""
Tests for the classification of the human protein-protein interactions that
back the human PPI gene programs. All tests here are offline: they exercise the
classification helpers directly and never download anything.
"""

import json

import pytest

from nichecompass.utils.gene_programs import (
    _classify_humanppi_interaction,
    _classify_humanppi_go_cellular_components,
    _check_humanppi_cached_precision,
    _humanppi_gene_family_root,
    _humanppi_gene_symbol_role,
    _is_humanppi_ligand_receptor_symbol_pair,
    _is_humanppi_same_family_assembly,
    _humanppi_gene_symbol_stem_role,
    _humanppi_go_function_role,
    _is_humanppi_trans_capable,
    _orient_humanppi_intercellular_gp,
    _parse_humanppi_subcellular_location,
    _classify_humanppi_protein_location,
    _humanppi_cis_complex_gene_families,
    _is_humanppi_ig_tcr_segment,
    _is_humanppi_paralog_cross_pair)


LOCATION_CLASSES = ["cell_surface", "secreted", "intracellular", "ambiguous",
                    "unknown"]


def anchored(is_membrane_anchored=True, has_topological_domain=False,
             has_extracellular_domain=False, subcellular_location=None,
             has_signal_peptide=False, cellular_component_keywords=None,
             go_cellular_components=None):
    """Build a topology record as returned by the UniProt lookup."""
    return {"is_membrane_anchored": is_membrane_anchored,
            "has_topological_domain": has_topological_domain,
            "has_extracellular_domain": has_extracellular_domain,
            "has_signal_peptide": has_signal_peptide,
            "subcellular_location": subcellular_location or [],
            "cellular_component_keywords": cellular_component_keywords or [],
            "go_cellular_components": go_cellular_components or []}


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


def test_ambiguous_locality_governs_unresolved_evidence():
    # Only a keyword compatible with an extracellular face, and no topology
    assert _classify_humanppi_protein_location("Membrane") == "ambiguous"
    assert _classify_humanppi_protein_location(
        "Membrane", ambiguous_locality="intracellular") == "intracellular"
    # It must not override established evidence
    assert _classify_humanppi_protein_location(
        "Cell membrane", ambiguous_locality="intracellular") == "cell_surface"
    assert _classify_humanppi_protein_location(
        "Secreted", ambiguous_locality="intracellular") == "secreted"
    # Nor contradict topology, which is stronger evidence either way
    assert _classify_humanppi_protein_location(
        "Membrane", topology=anchored(has_topological_domain=True,
                                      has_extracellular_domain=True),
        ambiguous_locality="intracellular") == "cell_surface"
    assert _classify_humanppi_protein_location(
        "Membrane", topology=anchored(is_membrane_anchored=False),
        ambiguous_locality="extracellular") == "intracellular"


@pytest.mark.parametrize("annotation,expected", [
    ("SUBCELLULAR LOCATION: Cytoplasm {ECO:0000269|PubMed:17289661}. "
     "Nucleus, nucleolus {ECO:0000269|PubMed:34516797}. Note=Localized in "
     "granules. {ECO:0000269|PubMed:17289661}.",
     ["Cytoplasm", "Nucleus"]),
    # Topology qualifiers carry no location and are dropped
    ("SUBCELLULAR LOCATION: Cell membrane; Peripheral membrane protein "
     "{ECO:0000269|PubMed:20682791}. Endosome {ECO:0000269}.",
     ["Cell membrane", "Endosome"]),
    ("", []),
    (None, []),
])
def test_subcellular_location_parsing(annotation, expected):
    assert _parse_humanppi_subcellular_location(annotation) == expected


def test_subcellular_location_is_a_fallback_for_a_missing_keyword():
    # No cellular component keyword, but a curated subcellular location
    assert _classify_humanppi_protein_location("none") == "unknown"
    assert _classify_humanppi_protein_location(
        "none",
        topology=anchored(is_membrane_anchored=False,
                          subcellular_location=["Cytoplasm", "Nucleus"])) == (
                              "intracellular")
    # It must never override an existing keyword
    assert _classify_humanppi_protein_location(
        "Secreted",
        topology=anchored(subcellular_location=["Nucleus"])) == "secreted"
    # And it is still subject to the topology gate
    assert _classify_humanppi_protein_location(
        "none",
        topology=anchored(is_membrane_anchored=False,
                          subcellular_location=["Cell membrane"])) == (
                              "intracellular")


def test_refreshed_keywords_are_the_first_fallback():
    # The localization shipped with the predictions is a stale snapshot
    assert _classify_humanppi_protein_location(
        "none",
        topology=anchored(cellular_component_keywords=["Secreted"])) == (
            "secreted")
    # It must never override a localization that is present
    assert _classify_humanppi_protein_location(
        "Nucleus",
        topology=anchored(cellular_component_keywords=["Secreted"])) == (
            "intracellular")


@pytest.mark.parametrize("go_terms,expected", [
    (["nucleus", "cytosol"], "intracellular"),
    (["mitochondrial inner membrane"], "intracellular"),
    # Extracellular terms veto the conclusion rather than establishing one
    (["cytosol", "extracellular space"], "unknown"),
    (["cell surface"], "unknown"),
    (["external side of plasma membrane"], "unknown"),
    # Preparation-derived and generic terms establish nothing either way
    (["extracellular exosome"], "unknown"),
    (["blood microparticle"], "unknown"),
    (["plasma membrane"], "unknown"),
    (["extracellular exosome", "nucleus"], "intracellular"),
    (["plasma membrane", "cytosol"], "intracellular"),
    ([], "unknown"),
])
def test_go_cellular_components(go_terms, expected):
    assert _classify_humanppi_go_cellular_components(go_terms) == expected


def test_go_is_consulted_before_the_signal_peptide():
    # An endoplasmic reticulum chaperone has a signal peptide but is resident
    assert _classify_humanppi_protein_location(
        "none",
        topology=anchored(is_membrane_anchored=False, has_signal_peptide=True,
                          go_cellular_components=[
                              "endoplasmic reticulum lumen"])) == (
                                  "intracellular")
    # A secreted protease has a signal peptide and no intracellular GO term
    assert _classify_humanppi_protein_location(
        "none",
        topology=anchored(is_membrane_anchored=False,
                          has_signal_peptide=True)) == "secreted"
    # With a membrane anchor it ends up at a membrane instead
    assert _classify_humanppi_protein_location(
        "none",
        topology=anchored(is_membrane_anchored=True,
                          has_signal_peptide=True)) == "cell_surface"


def test_weak_evidence_is_only_used_when_no_keyword_is_available():
    # Established keywords always win over the weak fallbacks
    assert _classify_humanppi_protein_location(
        "Secreted",
        topology=anchored(go_cellular_components=["nucleus"])) == "secreted"
    assert _classify_humanppi_protein_location(
        "Nucleus",
        topology=anchored(has_signal_peptide=True,
                          is_membrane_anchored=False)) == "intracellular"


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

@pytest.mark.parametrize("location_1,location_2,expected", [
    ("cell_surface", "cell_surface", "juxtacrine"),
    ("secreted", "cell_surface", "paracrine"),
    ("secreted", "secreted", "paracrine"),
    ("cell_surface", "intracellular", "intracellular"),
    ("cell_surface", "ambiguous", "juxtacrine"),
    ("secreted", "ambiguous", "paracrine"),
    ("ambiguous", "ambiguous", "juxtacrine"),
    ("ambiguous", "intracellular", "intracellular"),
    ("unknown", "cell_surface", "unknown"),
    ("unknown", "unknown", "unknown"),
])
def test_interaction_classification(location_1, location_2, expected):
    assert _classify_humanppi_interaction(location_1, location_2) == expected


def test_interaction_classification_is_symmetric():
    for location_1 in LOCATION_CLASSES:
        for location_2 in LOCATION_CLASSES:
            assert (_classify_humanppi_interaction(location_1, location_2) ==
                    _classify_humanppi_interaction(location_2, location_1))


def test_interaction_classification_rejects_raw_localization_strings():
    with pytest.raises(ValueError, match="not a location class"):
        _classify_humanppi_interaction("Cell membrane,Membrane",
                                       "cell_surface")


###############################################################################
## Cache provenance ##
###############################################################################

def write_provenance(tmp_path, precision):
    network_file_path = tmp_path / "humanppi_network.csv"
    network_file_path.write_text("Name1\tName2\n")
    provenance = {"source_url": "https://example.invalid/predictions.tar.gz",
                  "retrieved_utc": "2026-08-27T00:00:00+00:00"}
    if precision is not None:
        provenance["precision"] = precision
    (tmp_path / "humanppi_network.csv.provenance.json").write_text(
        json.dumps(provenance))
    return str(network_file_path)


def test_cached_precision_accepts_a_matching_cache(tmp_path):
    # No exception: the cache holds the table that was asked for
    _check_humanppi_cached_precision(write_provenance(tmp_path, "90"), "90")


def test_cached_precision_rejects_a_mismatched_cache(tmp_path):
    # The two precision levels are different tables, so reading one while
    # asking for the other would return the wrong interaction set
    with pytest.raises(ValueError, match="retrieved at precision '90'"):
        _check_humanppi_cached_precision(write_provenance(tmp_path, "90"), "80")


@pytest.mark.parametrize("precision", [None])
def test_cached_precision_tolerates_a_cache_without_the_stamp(tmp_path,
                                                              precision):
    # Caches written before the precision was recorded carry no such entry
    _check_humanppi_cached_precision(write_provenance(tmp_path, precision),
                                     "80")


def test_cached_precision_tolerates_a_missing_provenance_file(tmp_path):
    network_file_path = tmp_path / "humanppi_network.csv"
    network_file_path.write_text("Name1\tName2\n")
    _check_humanppi_cached_precision(str(network_file_path), "80")


###############################################################################
## Reaching across the intercellular cleft ##
###############################################################################

def topology_record(n_transmem=1, max_extracellular_domain_length=200):
    return {"n_transmem": n_transmem,
            "max_extracellular_domain_length":
                max_extracellular_domain_length}


@pytest.mark.parametrize("gene,n_transmem,ectodomain,expected", [
    # Large ectodomains reach across: PD-1 146 aa, PD-L1 220 aa, NRP1 835 aa
    ("PDCD1", 1, 146, True),
    ("CD274", 1, 220, True),
    ("EFNB3", 1, 199, True),
    # Channel and transporter subunits do not: KCNJ6 24 aa, CNGA1 23 aa
    ("KCNJ6", 2, 24, False),
    ("CNGA1", 7, 23, False),
    ("HCN1", 6, 26, False),
    # Signalling adaptors have almost no ectodomain: CD247 9 aa, FCER1G 5 aa
    ("CD247", 1, 9, False),
    ("FCER1G", 1, 5, False),
    ("TYROBP", 1, 19, False),
    # No annotated ectodomain: polytopic proteins cannot reach, others are not
    # penalised, since only positive evidence demotes
    ("ORAI2", 4, 0, False),
    ("SOMEGENE", 1, 0, True),
    # Connexins, claudins and occludin dock through short loops
    ("GJA1", 4, 12, True),
    ("CLDN1", 4, 10, True),
    ("OCLN", 4, 8, True),
])
def test_trans_capability(gene, n_transmem, ectodomain, expected):
    assert _is_humanppi_trans_capable(
        gene, topology_record(n_transmem, ectodomain), 30) is expected


def test_trans_capability_without_topology_is_not_penalised():
    assert _is_humanppi_trans_capable("SOMEGENE", None, 30) is True


###############################################################################
## Assemblies within one gene family ##
###############################################################################

@pytest.mark.parametrize("gene,expected_root", [
    ("P2RX2", "P2RX"), ("P2RX3", "P2RX"), ("KCNA6", "KCNA"),
    ("ABCG1", "ABCG"), ("EPHA1", "EPHA"), ("EFNA1", "EFNA"),
    ("BTN3A1", "BTN3A"), ("OCLN", "OCLN"), ("BMP4", "BMP"),
])
def test_gene_family_root(gene, expected_root):
    assert _humanppi_gene_family_root(gene) == expected_root


@pytest.mark.parametrize("gene_1,gene_2", [
    # Channel, transporter and receptor subunits that oligomerize within one
    # membrane rather than binding a partner on a neighboring cell
    ("P2RX2", "P2RX3"), ("P2RX6", "P2RX7"), ("KCNA6", "KCNA1"),
    ("ABCG1", "ABCG4"), ("EPHA1", "EPHA2"), ("BTN3A1", "BTN3A2"),
    ("NOMO3", "NOMO1"),
    # Secreted multimers, assembled by the cell that produces them
    ("FCN1", "FCN2"), ("BMP4", "BMP7"), ("APOC2", "APOC3"), ("C1QL2", "C1QL4"),
])
def test_same_family_partners_assemble_within_one_cell(gene_1, gene_2):
    assert _is_humanppi_same_family_assembly(gene_1, gene_2)


@pytest.mark.parametrize("gene_1,gene_2", [
    # These families are built to engage a partner on the neighboring cell, so
    # two members of one of them is not evidence of a within-cell assembly
    ("CDH1", "CDH2"), ("PCDHB5", "PCDHB16"), ("CLDN3", "CLDN1"),
    ("CLDN16", "CLDN19"), ("GJA1", "GJA8"), ("NECTIN3", "NECTIN2"),
    ("CADM1", "CADM3"), ("DSG1", "DSG3"), ("JAM2", "JAM3"),
    ("NRXN1", "NRXN3"), ("CNTN1", "CNTN4"), ("SIGLEC7", "SIGLEC9"),
])
def test_trans_homophilic_families_are_exempt(gene_1, gene_2):
    assert not _is_humanppi_same_family_assembly(gene_1, gene_2)


@pytest.mark.parametrize("gene_1,gene_2", [
    # A ligand and its receptor never share a family root, so the rule must
    # not fire on them
    ("EFNA1", "EPHA2"), ("TNFSF4", "TNFRSF4"), ("CD274", "PDCD1"),
    ("KITLG", "KIT"), ("SEMA4D", "PLXNB1"), ("CCL19", "CCR7"),
])
def test_ligand_receptor_pairs_are_not_same_family_assemblies(gene_1, gene_2):
    assert not _is_humanppi_same_family_assembly(gene_1, gene_2)


@pytest.mark.parametrize("gene_1,gene_2", [
    # Stripping the trailing member number collapses a ligand and its own
    # receptor onto one family root, so they have to be excluded explicitly
    ("CSF1", "CSF1R"), ("CSF3", "CSF3R"), ("CSF2", "CSF2RA"),
    ("MST1", "MST1R"), ("KIT", "KITLG"), ("IL2", "IL2RA"),
])
def test_a_ligand_and_its_own_receptor_are_not_a_family_assembly(gene_1,
                                                                gene_2):
    assert _is_humanppi_ligand_receptor_symbol_pair(gene_1, gene_2)
    assert not _is_humanppi_same_family_assembly(gene_1, gene_2)


@pytest.mark.parametrize("gene_1,gene_2", [
    ("P2RX2", "P2RX3"), ("FCN1", "FCN2"), ("EPHA1", "EPHA2"),
])
def test_family_members_are_not_a_ligand_receptor_symbol_pair(gene_1, gene_2):
    assert not _is_humanppi_ligand_receptor_symbol_pair(gene_1, gene_2)


def test_a_short_family_root_does_not_decide():
    # A two-character root is too generic to be evidence of anything
    assert not _is_humanppi_same_family_assembly("CD4", "CD8")


###############################################################################
## Orientation of intercellular gene programs ##
###############################################################################

def annotated(go_functions=(), is_gpi=False, cytoplasmic=0, n_transmem=1,
              has_topo=True):
    return {"go_molecular_functions": list(go_functions),
            "is_gpi_anchored": is_gpi,
            "max_cytoplasmic_domain_length": cytoplasmic,
            "has_receptor_kinase_keyword": False,
            "has_topological_domain": has_topo,
            "n_transmem": n_transmem,
            "max_extracellular_domain_length": 200}


LIGAND_GO = ["receptor ligand activity"]
RECEPTOR_GO = ["transmembrane signaling receptor activity"]


def orient(gene_1, gene_2, location_1="cell_surface",
           location_2="cell_surface", interaction_class="juxtacrine",
           topology_1=None, topology_2=None, omnipath_roles=None):
    return _orient_humanppi_intercellular_gp(
        gene_1, gene_2, location_1, location_2,
        interaction_class=interaction_class, topology_1=topology_1,
        topology_2=topology_2, omnipath_roles=omnipath_roles)


@pytest.mark.parametrize("gene,expected", [
    ("EFNB3", "ligand"), ("EFNA1", "ligand"), ("TNFSF4", "ligand"),
    ("SEMA4D", "ligand"), ("JAG1", "ligand"), ("MICA", "ligand"),
    ("CD274", "ligand"), ("CD86", "ligand"), ("HLA-A", "ligand"),
    ("EPHB1", "receptor"), ("EPHA2", "receptor"), ("TNFRSF4", "receptor"),
    ("PLXNB1", "receptor"), ("NOTCH1", "receptor"), ("PDCD1", "receptor"),
    ("CTLA4", "receptor"), ("SIRPA", "receptor"), ("ITGB1", "receptor"),
    # No curated family, so no role is claimed
    ("TMEM154", None), ("SMIM5", None), ("GYPA", None),
])
def test_gene_symbol_role(gene, expected):
    assert _humanppi_gene_symbol_role(gene) == expected


@pytest.mark.parametrize("gene_1,gene_2,expected", [
    ("KITLG", "KIT", "KITLG"),
    ("KIT", "KITLG", "KITLG"),
    ("FASLG", "FAS", "FASLG"),
    ("EGF", "EGFR", "EGF"),
    ("EGFR", "EGF", "EGF"),
    # A bare 'L' suffix means "like" far more often than "ligand", so it must
    # not fire: PXDNL is a peroxidasin paralogue, not the ligand of PXDN
    ("PXDNL", "PXDN", None),
    ("DLL1", "DLK1", None),
    ("EFNB3", "EPHB1", None),
])
def test_gene_symbol_stem_role(gene_1, gene_2, expected):
    assert _humanppi_gene_symbol_stem_role(gene_1, gene_2) == expected


@pytest.mark.parametrize("functions,expected", [
    (LIGAND_GO, 1),
    (["cytokine activity"], 1),
    (RECEPTOR_GO, -1),
    (["protein tyrosine kinase activity"], -1),
    # Both or neither is no evidence either way
    (LIGAND_GO + RECEPTOR_GO, 0),
    (["calcium ion binding"], 0),
    ([], 0),
])
def test_go_function_role(functions, expected):
    assert _humanppi_go_function_role(annotated(functions)) == expected


def test_go_function_role_is_none_without_annotation():
    assert _humanppi_go_function_role(None) is None
    assert _humanppi_go_function_role({"go_molecular_functions": None}) is None


def test_a_soluble_partner_is_placed_in_the_source_component():
    # The ligand is released by the neighboring cell
    source, target, source_location, _, rule = orient(
        "CCR7", "CCL19", "cell_surface", "secreted",
        interaction_class="paracrine")
    assert (source, target) == ("CCL19", "CCR7")
    assert source_location == "secreted"
    assert rule == "secreted_partner"


def test_the_secreted_rule_beats_every_other_rule():
    # The symbols would put EFNB3 in the source, but a secreted partner is
    # decided on physical grounds and takes precedence
    source, _, _, _, rule = orient(
        "EFNB3", "EPHB1", "cell_surface", "secreted",
        interaction_class="paracrine")
    assert (source, rule) == ("EPHB1", "secreted_partner")


def test_two_soluble_partners_still_reach_the_remaining_rules():
    # 327 of the 856 paracrine interactions have two secreted partners, which
    # the secreted rule cannot separate
    source, _, _, _, rule = orient(
        "SOMEGENE", "OTHERGENE", "secreted", "secreted",
        interaction_class="paracrine",
        topology_1=annotated(LIGAND_GO), topology_2=annotated(RECEPTOR_GO))
    assert (source, rule) == ("SOMEGENE", "go_molecular_function")


@pytest.mark.parametrize("gene_1,gene_2,expected_source", [
    ("EFNB3", "EPHB1", "EFNB3"),
    ("EPHB1", "EFNB3", "EFNB3"),
    ("TNFSF4", "TNFRSF4", "TNFSF4"),
    ("JAG1", "NOTCH1", "JAG1"),
    ("CD274", "PDCD1", "CD274"),
    ("MICA", "KLRK1", "MICA"),
    ("SEMA4D", "PLXNB1", "SEMA4D"),
])
def test_curated_families_put_the_ligand_in_the_source(gene_1, gene_2,
                                                       expected_source):
    source, target, _, _, rule = orient(gene_1, gene_2)
    assert source == expected_source
    assert target == (gene_2 if expected_source == gene_1 else gene_1)
    assert rule == "gene_symbol_family"


@pytest.mark.parametrize("gene_1,gene_2", [
    # Both sides match the same lexicon, so no orientation is claimed
    ("EPHA4", "EPHB2"), ("EFNA1", "EFNA5"), ("ITGA4", "ITGB7"),
])
def test_curated_families_abstain_when_both_sides_match(gene_1, gene_2):
    assert orient(gene_1, gene_2)[4] != "gene_symbol_family"


def test_omnipath_exclusive_role():
    roles = {"AAA": {"ligand": 3, "cell_surface_ligand": 0, "receptor": 0},
             "BBB": {"ligand": 0, "cell_surface_ligand": 0, "receptor": 4}}
    source, _, _, _, rule = orient("BBB", "AAA", omnipath_roles=roles)
    assert (source, rule) == ("AAA", "omnipath_exclusive_role")


def test_omnipath_surface_ligand():
    # A ligand that stays on the surface is what a contact-dependent sender is
    roles = {"AAA": {"ligand": 1, "cell_surface_ligand": 2, "receptor": 1},
             "BBB": {"ligand": 0, "cell_surface_ligand": 0, "receptor": 4}}
    source, _, _, _, rule = orient("BBB", "AAA", omnipath_roles=roles)
    assert (source, rule) == ("AAA", "omnipath_surface_ligand")


def test_omnipath_abstains_when_both_are_annotated_the_same_way():
    roles = {"AAA": {"ligand": 2, "cell_surface_ligand": 1, "receptor": 2},
             "BBB": {"ligand": 2, "cell_surface_ligand": 1, "receptor": 2}}
    assert orient("AAA", "BBB", omnipath_roles=roles)[4] == "table_order"


def test_go_molecular_function_requires_evidence_on_both_sides():
    # A receptor function against no annotation at all must not decide: that
    # would let absence of evidence choose the ligand
    assert orient("AAA", "BBB", topology_1=annotated([]),
                  topology_2=annotated(RECEPTOR_GO))[4] == "table_order"
    # Positive evidence on both sides does decide
    source, _, _, _, rule = orient("AAA", "BBB",
                                   topology_1=annotated(RECEPTOR_GO),
                                   topology_2=annotated(LIGAND_GO))
    assert (source, rule) == ("BBB", "go_molecular_function")


def test_a_gpi_anchored_partner_is_the_ligand():
    # A GPI anchored protein has no cytoplasmic domain and cannot transduce a
    # signal into its own cell, as for the ephrin-A proteins
    source, _, _, _, rule = orient(
        "AAA", "BBB", topology_1=annotated(is_gpi=False),
        topology_2=annotated(is_gpi=True))
    assert (source, rule) == ("BBB", "gpi_anchor")


def test_the_curated_family_beats_the_uniprot_evidence():
    # Contradictory annotation must not override the curated family
    source, _, _, _, rule = orient(
        "EFNB3", "EPHB1", topology_1=annotated(RECEPTOR_GO),
        topology_2=annotated(LIGAND_GO))
    assert (source, rule) == ("EFNB3", "gene_symbol_family")


def test_omnipath_beats_the_uniprot_evidence():
    roles = {"AAA": {"ligand": 0, "cell_surface_ligand": 0, "receptor": 3},
             "BBB": {"ligand": 2, "cell_surface_ligand": 0, "receptor": 0}}
    source, _, _, _, rule = orient(
        "AAA", "BBB", omnipath_roles=roles,
        topology_1=annotated(LIGAND_GO), topology_2=annotated(RECEPTOR_GO))
    assert (source, rule) == ("BBB", "omnipath_exclusive_role")


def test_no_evidence_keeps_the_released_column_order():
    source, target, source_location, target_location, rule = orient(
        "TMEM154", "GYPA", topology_1=annotated([]), topology_2=annotated([]))
    assert (source, target) == ("TMEM154", "GYPA")
    assert (source_location, target_location) == ("cell_surface",
                                                 "cell_surface")
    assert rule == "table_order"


def test_locations_travel_with_their_genes_when_the_orientation_flips():
    _, _, source_location, target_location, _ = orient(
        "EPHB1", "EFNB3", location_1="ambiguous", location_2="cell_surface")
    assert (source_location, target_location) == ("cell_surface", "ambiguous")


###############################################################################
## Gene program orientation ##
###############################################################################

@pytest.mark.parametrize("gene_1,gene_2,location_1,location_2,expected,rule", [
    # The diffusible ligand belongs in the source (neighbour) component
    ("CCR7", "CCL19", "cell_surface", "secreted", ("CCL19", "CCR7"),
     "secreted_partner"),
    ("AXL", "GAS6", "cell_surface", "secreted", ("GAS6", "AXL"),
     "secreted_partner"),
    # Already correctly oriented, so left alone
    ("CCL19", "CCR7", "secreted", "cell_surface", ("CCL19", "CCR7"),
     "secreted_partner"),
    # Contact dependent between two membrane anchored partners: the curated
    # families identify PD-L1 as the ligand of PD-1, so the released order is
    # corrected rather than preserved
    ("PDCD1", "CD274", "cell_surface", "cell_surface", ("CD274", "PDCD1"),
     "gene_symbol_family"),
    # Two soluble partners, neither of which any rule can separate, so the
    # released order is preserved
    ("COL1A1", "COL1A2", "secreted", "secreted", ("COL1A1", "COL1A2"),
     "table_order"),
])
def test_the_soluble_partner_is_placed_in_the_source_component(
        gene_1, gene_2, location_1, location_2, expected, rule):
    """
    NicheCompass reconstructs the source component from the aggregated
    expression of a cell's neighbours and the target component from the cell
    itself, so a diffusible ligand belongs in the source component and its
    receptor in the target component.
    """
    interaction_class = ("paracrine" if "secreted" in (location_1, location_2)
                         else "juxtacrine")
    (source_gene, target_gene, source_location, target_location,
     orientation_rule) = _orient_humanppi_intercellular_gp(
         gene_1, gene_2, location_1, location_2,
         interaction_class=interaction_class)
    assert (source_gene, target_gene) == expected
    assert orientation_rule == rule
    # The locations must travel with their genes
    original = {gene_1: location_1, gene_2: location_2}
    assert source_location == original[source_gene]
    assert target_location == original[target_gene]


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
    # Co-receptor complexes that assemble on a single cell and are large enough
    # to reach across, so they have to be curated
    ("ERBB2", "ERBB3"), ("EGFR", "ERBB2"), ("INSR", "IGF1R"),
    ("BMPR1A", "BMPR2"), ("ENG", "TGFBR2"), ("TLR1", "TLR2"),
    ("GABBR1", "GABBR2"), ("PTCH1", "SMO"), ("NRP1", "KDR"),
    ("CALCR", "RAMP1"), ("ABCG5", "ABCG8"), ("SLC51A", "SLC51B"),
    ("ASIC4", "ASIC3"), ("AQP1", "AQP6")])
def test_curated_coreceptor_families_share_a_family(gene_1, gene_2):
    assert (_humanppi_cis_complex_gene_families(gene_1) &
            _humanppi_cis_complex_gene_families(gene_2))


@pytest.mark.parametrize("gene_1,gene_2", [
    ("PDCD1", "CD274"), ("SIRPA", "CD47"), ("TNF", "TNFRSF1A"),
    ("IL15", "IL15RA"), ("EPHB1", "EFNB3"),
    # A secreted ligand and its receptor must never share a family
    ("TGFB1", "TGFBR2"), ("VEGFA", "KDR"), ("CCL19", "CCR7")])
def test_ligand_receptor_pairs_do_not_share_a_family(gene_1, gene_2):
    assert not (_humanppi_cis_complex_gene_families(gene_1) &
                _humanppi_cis_complex_gene_families(gene_2))
