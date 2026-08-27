"""
This module contains utilities to retrieve interpretable prior knowledge gene
programs for use by the NicheCompass model.
"""

import copy
import os
import re
import shutil
import ssl
import tarfile
import tempfile
import urllib.error
import urllib.request
import warnings
from typing import Literal, Optional

import decoupler as dc
import numpy as np
import omnipath as op
import pandas as pd
from anndata import AnnData

from .utils import load_R_file_as_df, create_gp_gene_count_distribution_plots


# Classification of UniProt cellular-component keywords, used to decide whether
# a predicted protein-protein interaction can act between neighboring cells
# (intercellular) or only within a cell (intracellular).
#
# Each keyword is assigned to exactly one of four groups. The guiding principle
# is that a keyword only counts as evidence for an extracellular face if every
# protein carrying it necessarily presents one. Compartment keywords that cover
# both a membrane-embedded core and a cytoplasmic plaque or interior (for
# example ´Cell junction´, which is carried by cadherins as well as by
# cytosolic catenins, vinculin and ZO-1) are therefore NOT treated as evidence
# of an extracellular face; the genuinely surface-exposed proteins in those
# compartments virtually always also carry ´Cell membrane´.

# Soluble proteins that are released into the extracellular space. An
# interaction involving such a protein is paracrine rather than contact
# dependent.
HUMANPPI_SECRETED_KEYWORDS = {
    "Secreted", "Extracellular matrix", "Basement membrane",
    "Membrane attack complex", "Surface film",
    "HDL", "LDL", "VLDL", "Chylomicron"}

# Membrane-anchored proteins that necessarily present an extracellular face,
# i.e. specific plasma-membrane keywords and cell-surface complexes. An
# interaction between two such proteins is contact dependent (juxtacrine).
HUMANPPI_CELL_SURFACE_KEYWORDS = {
    "Cell membrane", "Cell surface", "Apical cell membrane",
    "Basolateral cell membrane", "Apicolateral cell membrane",
    "Lateral cell membrane", "Presynaptic cell membrane",
    "Postsynaptic cell membrane", "Sarcolemma", "Membrane raft",
    "Cilium membrane", "Gap junction", "MHC I", "MHC II", "T cell receptor",
    "Target cell membrane"}

# Keywords that establish an intracellular location. A protein carrying one of
# these and no surface or secreted keyword cannot act between cells.
HUMANPPI_INTRACELLULAR_KEYWORDS = {
    "Cytoplasm", "Nucleus", "Nucleosome core", "Chromosome", "Centromere",
    "Kinetochore", "Telomere", "Mitochondrion", "Mitochondrion inner membrane",
    "Mitochondrion outer membrane", "Mitochondrion nucleoid",
    "Endoplasmic reticulum", "Sarcoplasmic reticulum", "Microsome",
    "Golgi apparatus", "Lysosome", "Endosome", "Peroxisome", "Vacuole",
    "Autophagosome", "Melanosome", "Phagosome", "Cytoplasmic vesicle",
    "Coated pit", "Lipid droplet", "Cytoskeleton", "Microtubule",
    "Intermediate filament", "Keratin", "Thick filament", "Dynein",
    "Proteasome", "Ribosome", "Spliceosome", "Exosome", "Signalosome",
    "Inflammasome", "Primosome", "Signal recognition particle",
    "DNA-directed RNA polymerase", "Nuclear pore complex", "Synaptosome",
    "Flagellum", "CF(0)", "CF(1)"}

# Keywords that are compatible with an extracellular face but do not establish
# one, either because they are generic parents of the whole membrane branch
# (´Membrane´), because they denote a compartment with a large cytoplasmic
# component (junctions, projections, synapses), or because they describe a
# property rather than a location (´Amyloid´). These count as evidence of an
# extracellular face, but ´use_topology´ resolves them with sequence-level
# evidence where UniProt provides it.
HUMANPPI_AMBIGUOUS_LOCATION_KEYWORDS = {
    "Membrane", "Cell junction", "Tight junction", "Adherens junction",
    "Desmosome", "Hemidesmosome", "Focal adhesion", "Cell projection",
    "Cilium", "Synapse", "Microvillus", "Filopodium", "Lamellipodium",
    "Dendritic spine", "Stereocilium", "Growth cone", "Axon", "Dendrite",
    "Amyloid", "Immunoglobulin", "Virion", "Viral envelope protein",
    "Target membrane"}


def _classify_humanppi_protein_location(
        localization,
        topology: Optional[dict]=None,
        process: Optional[str]=None) -> str:
    """
    Classify a protein into ´secreted´, ´cell_surface´, ´intracellular´,
    ´ambiguous´ or ´unknown´ based on its UniProt cellular-component keywords.

    Evidence for an extracellular face takes precedence over evidence for an
    intracellular location, because many genuinely secreted or surface proteins
    are additionally annotated with the compartments they traverse (interleukin
    15, for example, is annotated ´Cytoplasm,Nucleus,Secreted´).

    Membrane anchoring in turn takes precedence over secretion, because a
    substantial number of cell-surface receptors additionally carry the
    ´Secreted´ keyword on account of a shed soluble isoform (programmed
    cell death 1 ligand 1, for example, is annotated
    ´Cell membrane,Endosome,Membrane,Nucleus,Secreted´). Classifying those as
    secreted would turn contact-dependent interactions such as PD-1 / PD-L1
    into paracrine ones.

    Antibody chains are an exception to that second rule, since they carry
    ´Cell membrane´ for the B cell receptor form and ´Secreted´ for the
    antibody form, and it is the secreted form that dominates their
    interactions with Fc receptors.
    """
    # A protein presents an extracellular face only if it is membrane anchored,
    # and is contradicted if UniProt annotates its topological domains without
    # any extracellular one
    can_be_at_cell_surface = None
    if topology is not None:
        can_be_at_cell_surface = (
            topology["is_membrane_anchored"] and
            not (topology["has_topological_domain"] and
                 not topology["has_extracellular_domain"]))

    if pd.isna(localization):
        keywords = []
    else:
        keywords = [keyword.strip() for keyword in str(localization).split(",")]
        keywords = [keyword for keyword in keywords
                    if keyword and keyword.lower() != "none"]
    if not keywords:
        # Fall back on biological process keywords, which cannot establish an
        # extracellular face but can establish an intracellular location
        if process is not None and not pd.isna(process):
            processes = [entry.strip() for entry in str(process).split(",")]
            if any(entry in HUMANPPI_INTRACELLULAR_PROCESS_KEYWORDS
                   for entry in processes):
                return "intracellular"
        return "unknown"
    # Antibody chains are an exception to the precedence of membrane anchoring
    # over secretion: they carry ´Cell membrane´ for the B cell receptor form
    # and ´Secreted´ for the antibody form, and it is the secreted form that
    # dominates their interactions with Fc receptors. Without this exception,
    # canonical secreted-antibody to surface-receptor interactions such as
    # IgG1 / Fc-gamma-receptor-IIb would be classified as contact dependent.
    if ("Immunoglobulin" in keywords
            and any(keyword in HUMANPPI_SECRETED_KEYWORDS
                    for keyword in keywords)):
        return "secreted"
    # A cell-surface keyword is only trusted if the protein is not known to
    # lack a membrane anchor, since a protein without a transmembrane segment
    # or GPI anchor cannot present a face at the cell surface. Secreted
    # keywords are always trusted, because proteins can also be released
    # through non-classical routes that leave no sequence signature.
    if (can_be_at_cell_surface is not False
            and any(keyword in HUMANPPI_CELL_SURFACE_KEYWORDS
                    for keyword in keywords)):
        return "cell_surface"
    # An annotated extracellular topological domain is decisive positive
    # evidence and outweighs the cellular-component keywords. UniProt uses
    # ´Extracellular´ only for the outside of the cell and ´Lumenal´ for the
    # inside of organelles, so this does not admit organelle membranes. Many
    # cytokine receptors need this rule, since they are annotated with the
    # generic ´Membrane´ keyword, with ´Secreted´ for a shed soluble form and
    # with the compartments they traverse, but never with ´Cell membrane´
    # (interleukin 15 receptor subunit alpha and interleukin 10 receptor
    # subunit beta among them).
    if topology is not None and topology["has_extracellular_domain"]:
        return "cell_surface"
    # A protein that UniProt annotates with a membrane anchor and that carries
    # no intracellular keyword is at the cell surface even when its
    # cellular-component keywords are only compatible with, rather than
    # establishing, an extracellular face. Many cytokine receptors are
    # annotated with the generic ´Membrane´ keyword and ´Secreted´ for a shed
    # soluble form but never with ´Cell membrane´ (interleukin 15 receptor
    # subunit alpha and interleukin 10 receptor subunit beta among them), and
    # would otherwise be classified as secreted ligands.
    if (can_be_at_cell_surface is True
            and not any(keyword in HUMANPPI_INTRACELLULAR_KEYWORDS
                        for keyword in keywords)
            and any(keyword in HUMANPPI_AMBIGUOUS_LOCATION_KEYWORDS
                    for keyword in keywords)):
        return "cell_surface"
    if any(keyword in HUMANPPI_SECRETED_KEYWORDS for keyword in keywords):
        return "secreted"
    if any(keyword in HUMANPPI_INTRACELLULAR_KEYWORDS for keyword in keywords):
        return "intracellular"
    if any(keyword in HUMANPPI_AMBIGUOUS_LOCATION_KEYWORDS
           for keyword in keywords):
        # An ambiguous keyword that is contradicted by the absence of a
        # membrane anchor is evidence of an intracellular location
        return "ambiguous" if can_be_at_cell_surface is not False else (
            "intracellular")
    return "unknown"


# Gene families whose members assemble into a common complex within one cell,
# used to complement the EBI Complex Portal. The Complex Portal covers only
# around a fifth of the proteins of the human interactome and is missing
# several prominent cell-surface complexes entirely, among them the MHC class
# II alpha-beta heterodimer, the B cell receptor CD79 heterodimer and the
# high-affinity IgE receptor. Two proteins matching the same pattern are
# treated as subunits of a common complex.
HUMANPPI_CIS_COMPLEX_GENE_FAMILY_PATTERNS = {
    "mhc_class_i": r"^(HLA-[ABCEFG]|B2M)$",
    "mhc_class_ii": r"^HLA-D[RQPMO]",
    "t_cell_receptor": r"^(CD3[DEG]|CD247|TRAC|TRBC\d*|TRDC|TRGC\d*)$",
    "cd8_coreceptor": r"^CD8[AB]$",
    "b_cell_receptor": r"^CD79[AB]$",
    "fc_epsilon_receptor": r"^(FCER1[AG]|MS4A2)$",
    "cd94_nkg2_receptor": r"^KLR[DC]\d?$",
    "integrin": r"^ITG[AB]",
    "collagen": r"^COL\d",
    "laminin": r"^LAM[ABC]\d",
    "sarcoglycan": r"^SGC[ABDEG]$",
    "bbsome": r"^(BBS\d+|TTC8|ARL6)$",
    "ap2_adaptor": r"^AP2[ABMS]\d",
    "complement_c1q": r"^C1Q[ABC]$",
    "fibrinogen": r"^FG[ABG]$",
    "gaba_a_receptor": r"^GABR[ABGDEPQR]",
    "ionotropic_glutamate_receptor": r"^GRI[ANKD]\d",
    "glycine_receptor": r"^GLR[AB]\d?$",
    "nicotinic_acetylcholine_receptor": r"^CHRN[ABDEG]\d?$",
    "serotonin_receptor_3": r"^HTR3[A-E]$",
    "epithelial_sodium_channel": r"^SCNN1[ABGD]$",
    "heteromeric_amino_acid_transporter": r"^SLC[37]A\d+$",
    "catsper_channel": r"^CATSPER",
    "shared_gamma_chain_cytokine_receptor":
        r"^(IL2R[ABG]|IL4R|IL7R|IL9R|IL21R)$"}


# Immunoglobulin and T cell receptor V, D and J gene segments. These are
# somatically recombined into a single antibody or receptor chain, so an
# interaction between two of them, or between a segment and the constant region
# of the same chain, is intramolecular rather than an interaction between two
# proteins. The structural predictor produces many such pairs because all
# variable domains share the immunoglobulin fold. Constant region genes
# (´IGHG1´, ´IGHM´, ´IGLC2´, ´TRAC´, ´TRBC1´, ...) encode real proteins and are
# deliberately not matched.
# UniProt biological-process keywords that establish an intracellular location.
# They are used only as a fallback for proteins without any cellular-component
# keyword, and only to establish an intracellular location, never an
# extracellular face.
HUMANPPI_INTRACELLULAR_PROCESS_KEYWORDS = {
    "Transcription", "Transcription regulation", "Transcription termination",
    "Transcription antitermination", "DNA damage", "DNA repair",
    "DNA replication", "DNA recombination", "DNA condensation",
    "mRNA processing", "mRNA splicing", "mRNA transport", "rRNA processing",
    "tRNA processing", "Translation regulation", "Protein biosynthesis",
    "Protein transport", "Protein folding", "Cell cycle", "Cell division",
    "Mitosis", "Meiosis", "Chromosome partition", "Ubl conjugation pathway",
    "Autophagy", "Proteasome", "Nonsense-mediated mRNA decay",
    "Keratinization", "Glycolysis", "Gluconeogenesis",
    "Tricarboxylic acid cycle", "Fatty acid biosynthesis",
    "Fatty acid metabolism", "Respiratory chain",
    "Electron transport", "Nucleotide biosynthesis", "Purine biosynthesis",
    "Pyrimidine biosynthesis", "Cholesterol biosynthesis",
    "Steroid biosynthesis", "Ribosome biogenesis", "Spermatogenesis"}

HUMANPPI_IG_TCR_SEGMENT_PATTERN = (
    r"^(IG[HKL][VJ]\d|IGHD\d|TR[ABDG][VJ]\d|TR[BD]D\d)")


# Heterodimers that actually form, for gene families in which the structural
# predictor generalizes across close paralogues and produces combinations that
# do not exist. Integrin alpha-beta heterodimers are restricted to the 24 known
# pairs, and MHC class II alpha-beta heterodimers to matching isotypes.
HUMANPPI_VALID_INTEGRIN_HETERODIMERS = {
    frozenset(pair) for pair in [
        ("ITGA1", "ITGB1"), ("ITGA2", "ITGB1"), ("ITGA3", "ITGB1"),
        ("ITGA4", "ITGB1"), ("ITGA5", "ITGB1"), ("ITGA6", "ITGB1"),
        ("ITGA7", "ITGB1"), ("ITGA8", "ITGB1"), ("ITGA9", "ITGB1"),
        ("ITGA10", "ITGB1"), ("ITGA11", "ITGB1"), ("ITGAV", "ITGB1"),
        ("ITGAL", "ITGB2"), ("ITGAM", "ITGB2"), ("ITGAX", "ITGB2"),
        ("ITGAD", "ITGB2"), ("ITGA2B", "ITGB3"), ("ITGAV", "ITGB3"),
        ("ITGA6", "ITGB4"), ("ITGAV", "ITGB5"), ("ITGAV", "ITGB6"),
        ("ITGA4", "ITGB7"), ("ITGAE", "ITGB7"), ("ITGAV", "ITGB8")]}


def _is_humanppi_paralog_cross_pair(gene_1: str, gene_2: str) -> bool:
    """
    Return whether two genes are paralogues that the structural predictor pairs
    although the corresponding heterodimer does not form.
    """
    gene_1, gene_2 = gene_1.upper(), gene_2.upper()
    if re.match(r"^ITG[AB]", gene_1) and re.match(r"^ITG[AB]", gene_2):
        return frozenset((gene_1, gene_2)) not in (
            HUMANPPI_VALID_INTEGRIN_HETERODIMERS)
    mhc_2_isotype = r"^HLA-D([RQP])"
    match_1 = re.match(mhc_2_isotype, gene_1)
    match_2 = re.match(mhc_2_isotype, gene_2)
    if match_1 and match_2:
        # Only matching isotypes pair, e.g. HLA-DRA with HLA-DRB1
        return match_1.group(1) != match_2.group(1)
    return False


def _is_humanppi_ig_tcr_segment(gene: str) -> bool:
    """
    Return whether a gene is an immunoglobulin or T cell receptor V, D or J
    gene segment rather than a gene encoding a complete protein.
    """
    return re.match(HUMANPPI_IG_TCR_SEGMENT_PATTERN, gene.upper()) is not None


def _load_humanppi_protein_topology(accessions: list,
                                    topology_file_path: str) -> dict:
    """
    Retrieve membrane topology for a list of UniProt accessions and return a
    mapping from accession to whether the protein is membrane anchored, i.e.
    whether it has a transmembrane segment or a GPI anchor.

    Cellular-component keywords describe whole proteins and do not indicate
    which side of a membrane a protein faces, so proteins that are docked onto
    the cytoplasmic leaflet of the plasma membrane (SNAP25, the protein kinase A
    holoenzyme, adducin, calpains) carry the ´Cell membrane´ keyword even though
    they cannot participate in intercellular interactions. Membrane anchoring is
    a sequence-level property and resolves those cases.

    Results are cached at ´topology_file_path´ and only accessions that are
    missing from the cache are requested.

    Parameters
    ----------
    accessions:
        UniProt accessions for which topology is retrieved.
    topology_file_path:
        Path of the file where the retrieved topology is cached.

    Returns
    ----------
    topology:
        Mapping from UniProt accession to a bool indicating membrane anchoring.
        Accessions for which UniProt returned no entry are absent.
    """
    topology = {}
    required_columns = ["accession", "is_membrane_anchored",
                        "has_topological_domain", "has_extracellular_domain"]
    if os.path.exists(topology_file_path):
        topology_df = pd.read_csv(topology_file_path, sep="\t")
        if not set(required_columns).issubset(topology_df.columns):
            # An outdated cache is ignored and retrieved again
            topology_df = topology_df.iloc[0:0]
        for _, topology_row in topology_df.iterrows():
            topology[str(topology_row["accession"])] = {
                "is_membrane_anchored": bool(
                    topology_row["is_membrane_anchored"]),
                "has_topological_domain": bool(
                    topology_row["has_topological_domain"]),
                "has_extracellular_domain": bool(
                    topology_row["has_extracellular_domain"])}

    missing = sorted({accession for accession in accessions
                      if accession not in topology})
    if missing:
        print(f"Retrieving membrane topology for {len(missing)} proteins from "
              "UniProt...")
        batch_size = 100
        fields = "accession,ft_transmem,ft_lipid,ft_topo_dom"
        for batch_start in range(0, len(missing), batch_size):
            batch = missing[batch_start:batch_start + batch_size]
            query = "+OR+".join(f"accession:{accession}"
                                for accession in batch)
            url = (f"https://rest.uniprot.org/uniprotkb/search?query={query}"
                   f"&fields={fields}&format=tsv&size={batch_size}")
            with urllib.request.urlopen(url) as response:
                lines = response.read().decode("utf-8").splitlines()
            for line in lines[1:]:
                columns = line.split("\t")
                if not columns or not columns[0]:
                    continue
                accession = columns[0]
                transmem = columns[1] if len(columns) > 1 else ""
                lipid = columns[2] if len(columns) > 2 else ""
                topo_dom = columns[3] if len(columns) > 3 else ""
                topology[accession] = {
                    "is_membrane_anchored": bool("TRANSMEM" in transmem
                                                 or "GPI-anchor" in lipid),
                    "has_topological_domain": "TOPO_DOM" in topo_dom,
                    "has_extracellular_domain": "Extracellular" in topo_dom}

        cache_dir = os.path.dirname(topology_file_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        pd.DataFrame([{"accession": accession, **values}
                      for accession, values in topology.items()]).to_csv(
                          topology_file_path, sep="\t", index=False)
    return topology


def _humanppi_cis_complex_gene_families(gene: str) -> set:
    """
    Return the names of the curated gene families that a gene belongs to.
    """
    return {family for family, pattern in
            HUMANPPI_CIS_COMPLEX_GENE_FAMILY_PATTERNS.items()
            if re.match(pattern, gene.upper())}


def _load_complex_portal_cis_pairs(complex_portal_file_path: str,
                                   complex_portal_url: str) -> set:
    """
    Load the human protein complexes of the EBI Complex Portal and return the
    set of unordered UniProt accession pairs whose two proteins are subunits of
    a common complex. Such pairs assemble within a single cell and are
    therefore not intercellular, even when both subunits are located at the
    cell surface.

    The file is downloaded on first use and cached at
    ´complex_portal_file_path´.

    Parameters
    ----------
    complex_portal_file_path:
        Path of the file where the Complex Portal table is cached.
    complex_portal_url:
        URL of the Complex Portal table for human, used if the cached file does
        not exist.

    Returns
    ----------
    cis_pairs:
        Set of ´frozenset´s of two UniProt accessions.
    """
    if os.path.exists(complex_portal_file_path):
        complex_df = pd.read_csv(complex_portal_file_path, sep="\t")
    else:
        print("Downloading human protein complexes from the EBI Complex "
              "Portal...")
        complex_df = pd.read_csv(complex_portal_url, sep="\t")
        cache_dir = os.path.dirname(complex_portal_file_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        complex_df.to_csv(complex_portal_file_path, sep="\t", index=False)

    subunit_cols = [col for col in
                    ["Expanded participant list",
                     "Identifiers (and stoichiometry) of molecules in complex"]
                    if col in complex_df.columns]
    if not subunit_cols:
        raise ValueError(
            "Could not find a subunit column in the Complex Portal table at "
            f"'{complex_portal_file_path}'. Expected 'Expanded participant "
            "list' or 'Identifiers (and stoichiometry) of molecules in "
            "complex'.")

    def parse_subunits(subunit_string) -> list:
        # Entries look like 'P84022(1)|Q13485(1)'. Stoichiometry is stripped,
        # as are chain ('-PRO_') and isoform ('-2') suffixes. Non-protein
        # participants (small molecules, RNAs, nested complexes) are skipped.
        if pd.isna(subunit_string):
            return []
        accessions = []
        for entry in str(subunit_string).split("|"):
            accession = entry.split("(")[0].strip()
            if not accession or accession == "-":
                continue
            if (":" in accession) or accession.startswith("CPX-"):
                continue
            accession = accession.split("-PRO_")[0].split("-")[0]
            if accession:
                accessions.append(accession)
        return accessions

    cis_pairs = set()
    for _, row in complex_df.iterrows():
        subunits = []
        for col in subunit_cols:
            subunits = parse_subunits(row[col])
            if len(subunits) > 1:
                break
        unique_subunits = sorted(set(subunits))
        # Complexes of exactly two subunits are skipped, because the Complex
        # Portal also registers ligand-receptor pairs as complexes and those
        # are predominantly binary (tumor necrosis factor with its receptors,
        # lymphotoxin beta with its receptor, colony stimulating factor 1 with
        # its receptor). Genuine binary cis assemblies such as the CD8, CD79
        # and integrin heterodimers are covered by the curated gene families
        # instead.
        if len(unique_subunits) < 3:
            continue
        for i, accession_i in enumerate(unique_subunits):
            for accession_j in unique_subunits[i + 1:]:
                cis_pairs.add(frozenset((accession_i, accession_j)))
    return cis_pairs


def _classify_humanppi_interaction(location_1: str,
                                   location_2: str) -> str:
    """
    Classify an interaction into ´paracrine´, ´juxtacrine´, ´intracellular´ or
    ´unknown´ based on the location classes of its two partners.

    Note that the two location arguments are location classes as returned by
    ´_classify_humanppi_protein_location´, not raw localization strings.
    """
    valid_locations = {"secreted", "cell_surface", "intracellular",
                       "ambiguous", "unknown"}
    for location in (location_1, location_2):
        if location not in valid_locations:
            raise ValueError(
                f"'{location}' is not a location class. Expected one of "
                f"{sorted(valid_locations)}, as returned by "
                "´_classify_humanppi_protein_location´. Note that raw UniProt "
                "localization strings have to be classified first.")
    if "unknown" in (location_1, location_2):
        return "unknown"
    extracellular_facing = {"secreted", "cell_surface", "ambiguous"}
    if location_1 in extracellular_facing and location_2 in extracellular_facing:
        if "secreted" in (location_1, location_2):
            return "paracrine"
        return "juxtacrine"
    return "intracellular"


def add_gps_from_gp_dict_to_adata(
        gp_dict: dict,
        adata: AnnData,
        genes_uppercase: bool=True,
        gp_targets_mask_key: str="nichecompass_gp_targets",
        gp_targets_categories_mask_key: str="nichecompass_gp_targets_categories",
        targets_categories_label_encoder_key: str="nichecompass_targets_categories_label_encoder",
        gp_sources_mask_key: str="nichecompass_gp_sources",
        gp_sources_categories_mask_key: str="nichecompass_gp_sources_categories",
        sources_categories_label_encoder_key: str="nichecompass_sources_categories_label_encoder",
        gp_names_key: str="nichecompass_gp_names",
        source_genes_idx_key: str="nichecompass_source_genes_idx",
        target_genes_idx_key: str="nichecompass_target_genes_idx",
        genes_idx_key: str="nichecompass_genes_idx",
        min_genes_per_gp: int=1,
        min_source_genes_per_gp: int=0,
        min_target_genes_per_gp: int=0,
        max_genes_per_gp: Optional[int]=None,
        max_source_genes_per_gp: Optional[int]=None,
        max_target_genes_per_gp: Optional[int]=None,
        filter_genes_not_in_masks: bool=False,
        add_fc_gps_instead_of_gp_dict_gps: bool=False,
        plot_gp_gene_count_distributions: bool=False):
    """
    Add gene programs defined in a gene program dictionary to an AnnData object.
    This is done by converting the gene program lists of gene program target and
    source genes to binary masks and aligning the masks with genes for which
    gene expression is available in the AnnData object.

    Parts of the implementation are inspired by
    https://github.com/theislab/scarches/blob/master/scarches/utils/annotations.py#L5
    (01.10.2022).

    Parameters
    ----------
    gp_dict:
        Nested dictionary containing the gene programs with keys being gene 
        program names and values being dictionaries with keys ´targets´ and 
        ´sources´, where ´targets´ contains a list of the names of genes in the
        gene program for the reconstruction of the gene expression of the node
        itself (receiving node) and ´sources´ contains a list of the names of
        genes in the gene program for the reconstruction of the gene expression
        of the node's neighbors (transmitting nodes).
    adata:
        AnnData object to which the gene programs will be added.
    genes_uppercase:
        If `True`, convert the gene names in the adata and in the gene program
        dictionary to uppercase for comparison.
    gp_targets_mask_key:
        Key in ´adata.varm´ where the binary gene program mask for target genes
        of a gene program will be stored (target genes are used for the 
        reconstruction of the gene expression of the node itself (receiving node
        )).
    gp_sources_mask_key:
        Key in ´adata.varm´ where the binary gene program mask for source genes
        of a gene program will be stored (source genes are used for the 
        reconstruction of the gene expression of the node's neighbors 
        (transmitting nodes)).
    gp_names_key:
        Key in ´adata.uns´ where the gene program names will be stored.
    source_genes_idx_key:
        Key in ´adata.uns´ where the index of the source genes that are in the
        gene program mask will be stored.
    target_genes_idx_key:
        Key in ´adata.uns´ where the index of the target genes that are in the
        gene program mask will be stored.
    genes_idx_key:
        Key in ´adata.uns´ where the index of a concatenated vector of target
        and source genes that are in the gene program masks will be stored.
    min_genes_per_gp:
        Minimum number of genes in a gene program inluding both target and 
        source genes that need to be available in the adata (gene expression has
        been probed) for a gene program not to be discarded.
    min_source_genes_per_gp:
        Minimum number of source genes in a gene program that need to be 
        available in the adata (gene expression has been probed) for a gene 
        program not to be discarded.
    min_target_genes_per_gp:
        Minimum number of target genes in a gene program that need to be 
        available in the adata (gene expression has been probed) for a gene 
        program not to be discarded.
    max_genes_per_gp:
        Maximum number of genes in a gene program inluding both target and 
        source genes that can be available in the adata (gene expression has 
        been probed) for a gene program not to be discarded.
    max_source_genes_per_gp:
        Maximum number of source genes in a gene program that can be available 
        in the adata (gene expression has been probed) for a gene program not to
        be discarded.
    max_target_genes_per_gp:
        Maximum number of target genes in a gene program that can be available 
        in the adata (gene expression has been probed) for a gene program not to
        be discarded.
    filter_genes_not_in_masks:
        If ´True´, remove the genes that are not in the gp masks from the adata
        object.
    add_fc_gps_instead_of_gp_dict_gps:
        Note: this parameter is just used for ablation studies. If ´True´,
        ignores the gene programs from the gp dict and instead creates a mask
        of fully-connected gene programs (same amount as gps in the gp dict).
    plot_gp_gene_count_distributions:
        If ´True´, display the distribution of gene programs per number of
        source and target genes.
    """
    # Retrieve probed genes from adata
    adata_genes = (adata.var_names.str.upper() if genes_uppercase
                   else adata.var_names)
    
    # Just for ablation studies, create fully-connected mask
    if add_fc_gps_instead_of_gp_dict_gps:
        gp_targets_mask = [[1 for _, _ in gp_dict.items()]
                           for gene in adata_genes]
        gp_targets_mask = np.asarray(gp_targets_mask, dtype="int32")
        gp_sources_mask = [[1 for _, _ in gp_dict.items()]
                           for gene in adata_genes]
        gp_sources_mask = np.asarray(gp_sources_mask, dtype="int32")
        
        gp_targets_categories_mask = [[0 for _, _ in gp_dict.items()]
                                      for gene in adata_genes]
        gp_targets_categories_mask = np.asarray(gp_targets_categories_mask,
                                                dtype="int32")
        gp_sources_categories_mask = [[0 for _, _ in gp_dict.items()]
                                      for gene in adata_genes]
        gp_sources_categories_mask = np.asarray(gp_sources_categories_mask,
                                                dtype="int32")
        
        categories_label_encoder = {"fc": 0}

        adata.varm[gp_sources_mask_key] = gp_sources_mask
        adata.varm[gp_targets_mask_key] = gp_targets_mask
        adata.varm[gp_sources_categories_mask_key] = gp_sources_categories_mask
        adata.varm[gp_targets_categories_mask_key] = gp_targets_categories_mask
        adata.uns[sources_categories_label_encoder_key] = (
            categories_label_encoder)
        adata.uns[targets_categories_label_encoder_key] = (
            categories_label_encoder)
        
        # Get index of genes present in the sources and targets mask respectively
        adata.uns[source_genes_idx_key] = np.arange(
            len(adata.varm[gp_sources_mask_key]))
        adata.uns[target_genes_idx_key] = np.arange(
            len(adata.varm[gp_targets_mask_key]))
        adata.uns[genes_idx_key] = np.concatenate(
            (adata.uns[target_genes_idx_key],
             adata.uns[source_genes_idx_key] + adata.n_vars), axis=0)

        # Add gene program names
        adata.uns[gp_names_key] = np.array([f"FC_{i}_GP" for i, (_, _) in 
                                            enumerate(gp_dict.items())])
        return
    
    if genes_uppercase:
        # Convert gene program genes to uppercase
        for _, gp_genes_dict in gp_dict.items():
            gp_genes_dict["sources"] = [
                source.upper() for source in gp_genes_dict["sources"]]
            gp_genes_dict["targets"] = [
                target.upper() for target in gp_genes_dict["targets"]]

    # Create binary gene program gene masks considering only probed genes
    gp_targets_mask = [[int(gene in gp_genes_dict["targets"])
                        for _, gp_genes_dict in gp_dict.items()]
                       for gene in adata_genes]
    gp_targets_mask = np.asarray(gp_targets_mask, dtype="int32")
    gp_sources_mask = [[int(gene in gp_genes_dict["sources"])
                        for _, gp_genes_dict in gp_dict.items()]
                       for gene in adata_genes]
    gp_sources_mask = np.asarray(gp_sources_mask, dtype="int32")
    gp_mask = np.concatenate((gp_sources_mask, gp_targets_mask), axis=0)
    
    # Create gene program gene category masks considering only probed genes
    # First, get unique categories
    sources_categories = []
    targets_categories = []
    for _, gp_genes_dict in gp_dict.items():
        sources_categories.extend(gp_genes_dict["sources_categories"])
        targets_categories.extend(gp_genes_dict["targets_categories"])
    sources_categories = list(set(sources_categories))
    targets_categories = list(set(targets_categories))
    
    # Second, create and store categories label encoders
    sources_categories_label_encoder = {
        k: v for k, v in zip(sources_categories, range(1, len(sources_categories) + 1))}
    targets_categories_label_encoder = {
        k: v for k, v in zip(targets_categories, range(1, len(targets_categories) + 1))}
    adata.uns[sources_categories_label_encoder_key] = sources_categories_label_encoder
    adata.uns[targets_categories_label_encoder_key] = targets_categories_label_encoder

    # Third, create new gp dict with label encoded categories
    category_encoded_gp_dict = copy.deepcopy(gp_dict)
    for _, gp_genes_dict in category_encoded_gp_dict.items():
        gp_genes_dict["targets_categories"] = [
            targets_categories_label_encoder.get(target) for target in
            gp_genes_dict["targets_categories"]]
        gp_genes_dict["sources_categories"] = [
            sources_categories_label_encoder.get(source) for source in
            gp_genes_dict["sources_categories"]]

    # Fourth, use label encoded gp dict to create category masks
    # (encode with category 0 if gene is not in mask)
    gp_targets_categories_mask = [
        [gp_genes_dict["targets_categories"][gp_genes_dict["targets"].index(gene)]
         if gene in gp_genes_dict["targets"] else 0
         for _, gp_genes_dict in category_encoded_gp_dict.items()]
        for gene in adata_genes]
    gp_targets_categories_mask = np.asarray(gp_targets_categories_mask, dtype="int32")

    gp_sources_categories_mask = [
        [gp_genes_dict["sources_categories"][gp_genes_dict["sources"].index(gene)]
         if gene in gp_genes_dict["sources"] else 0
         for _, gp_genes_dict in category_encoded_gp_dict.items()]
        for gene in adata_genes]
    gp_sources_categories_mask = np.asarray(gp_sources_categories_mask,
                                            dtype="int32")

    # Filter gene programs for min genes and max genes
    gp_mask_filter = gp_mask.sum(0) >= min_genes_per_gp
    if max_genes_per_gp is not None:
        gp_mask_filter &= gp_mask.sum(0) <= max_genes_per_gp
    gp_targets_mask_filter = gp_targets_mask.sum(0) >= min_target_genes_per_gp
    if max_target_genes_per_gp is not None:
        gp_targets_mask_filter &= (gp_targets_mask.sum(0)
                                   <= max_target_genes_per_gp)
    gp_sources_mask_filter = gp_sources_mask.sum(0) >= min_source_genes_per_gp
    if max_source_genes_per_gp is not None:
        gp_sources_mask_filter &= (gp_sources_mask.sum(0)
                                   <= max_source_genes_per_gp)
    gp_mask_filter &= gp_sources_mask_filter
    gp_mask_filter &= gp_targets_mask_filter
    gp_targets_mask = gp_targets_mask[:, gp_mask_filter]
    gp_sources_mask = gp_sources_mask[:, gp_mask_filter]
    gp_targets_categories_mask = gp_targets_categories_mask[:, gp_mask_filter]
    gp_sources_categories_mask = gp_sources_categories_mask[:, gp_mask_filter]

    # Add binary gene program gene masks to ´adata.varm´
    adata.varm[gp_sources_mask_key] = gp_sources_mask
    adata.varm[gp_targets_mask_key] = gp_targets_mask
    
    # Add gene program gene category masks to ´adata.varm´
    adata.varm[gp_sources_categories_mask_key] = gp_sources_categories_mask
    adata.varm[gp_targets_categories_mask_key] = gp_targets_categories_mask

    if filter_genes_not_in_masks:
        # Filter out genes not present in any of the masks
        combined_gp_mask = np.maximum(adata.varm["nichecompass_gp_sources"],
                                      adata.varm["nichecompass_gp_targets"])
        adata._inplace_subset_var(combined_gp_mask.sum(axis=1) > 0)

    # Get index of genes present in the sources and targets mask respectively
    adata.uns[source_genes_idx_key] = np.nonzero(
        adata.varm[gp_sources_mask_key].sum(axis=1))[0]
    adata.uns[target_genes_idx_key] = np.nonzero(
        adata.varm[gp_targets_mask_key].sum(axis=1))[0]
    adata.uns[genes_idx_key] = np.concatenate(
        (adata.uns[target_genes_idx_key],
         adata.uns[source_genes_idx_key] + adata.n_vars), axis=0)
         
    # Add gene program names of gene programs that passed filter to adata.uns
    removed_gp_idx = np.where(~gp_mask_filter)[0]
    adata.uns[gp_names_key] = np.array([gp_name for i, (gp_name, _) in 
                                        enumerate(gp_dict.items()) if i not in 
                                        removed_gp_idx])
    
    if plot_gp_gene_count_distributions:
        create_gp_gene_count_distribution_plots(adata=adata,
                                                gp_plot_label="AnnData")


def extract_gp_dict_from_collectri_tf_network(
        species: Literal["mouse", "human"],
        tf_network_file_path: Optional[str]="collectri_tf_network.csv",
        load_from_disk: bool=False,
        save_to_disk: bool=False,
        plot_gp_gene_count_distributions: bool=True,
        gp_gene_count_distributions_save_path: Optional[str]=None) -> dict:
    """
    Retrieve 1072 mouse or 1186 human transcription factor (TF) target gene gene
    programs from CollecTRI via decoupler. CollecTRI is a comprehensive resource
    containing a curated collection of TFs and their transcriptional targets
    compiled from 12 different resources. This collection provides an increased
    coverage of transcription factors and a superior performance in identifying
    perturbed TFs compared to the DoRothEA network and other literature based
    GRNs see
    https://decoupler-py.readthedocs.io/en/latest/notebooks/dorothea.html).

    Parameters
    ----------
    species:
        Species for which the gene programs will be extracted.
    load_from_disk:
        If ´True´, the CollecTRI TF network will be loaded from disk instead of
        from the decoupler library.
    save_to_disk:
        If ´True´, the CollecTRI TF network will additionally be stored on disk.
        Only applies if ´load_from_disk´ is ´False´.
    plot_gp_gene_count_distributions:
        If ´True´, display the distribution of gene programs per number of
        source and target genes.
    gp_gene_count_distributions_save_path:
        Path of the file where the gene program gene count distribution plot
        will be saved if ´plot_gp_gene_count_distributions´ is ´True´.

    Returns
    ----------
    gp_dict:
        Nested dictionary containing the CollecTRI TF target genes gene programs
        with keys being gene program names and values being dictionaries with
        keys ´sources´, ´targets´, ´sources_categories´, and
        ´targets_categories´, where ´sources´ and ´targets´ contain the
        CollecTRI TFs and target genes, and ´sources_categories´ and
        ´targets_categories´ contain the categories of all genes ('tf' or
        'target_gene').
    """
    if not load_from_disk:
        net = dc.op.collectri(organism=species, remove_complexes=False)
        if save_to_disk:
            net.to_csv(tf_network_file_path, index=False)
    else:
        net = pd.read_csv(tf_network_file_path)

    tf_target_genes_df = net[["source", "target"]].groupby(
        "source")["target"].agg(list).reset_index()
    
    gp_dict = {}
    for tf, target_genes in zip(tf_target_genes_df["source"],
                                tf_target_genes_df["target"]):
        gp_dict[tf + "_TF_target_genes_GP"] = {
            "sources": [],
            "targets": [tf] + target_genes,
            "sources_categories": [],
            "targets_categories": ["tf"] + ["target_gene"] * len(target_genes)}
        
    if plot_gp_gene_count_distributions:
        create_gp_gene_count_distribution_plots(
            gp_dict=gp_dict,
            gp_plot_label="CollecTRI",
            save_path=gp_gene_count_distributions_save_path)
        
    return gp_dict


def extract_gp_dict_from_nichenet_lrt_interactions(
        species: Literal["mouse", "human"],
        version: Literal["v1", "v2"]="v2",
        keep_target_genes_ratio: float=1.,
        max_n_target_genes_per_gp: int=250,
        load_from_disk: bool=False,
        save_to_disk: bool=False,
        lr_network_file_path: Optional[str]="nichenet_lr_network.csv",
        ligand_target_matrix_file_path: Optional[str]="../data/gene_programs/" \
                                                      "nichenet_ligand_target" \
                                                      "_matrix.csv",
        gene_orthologs_mapping_file_path: Optional[str]="../data/gene_" \
                                                        "annotations/human_" \
                                                        "mouse_gene_orthologs.csv",
        plot_gp_gene_count_distributions: bool=True,
        gp_gene_count_distributions_save_path: Optional[str]=None) -> dict:
    """
    Retrieve the NicheNet ligand receptor network and ligand target gene
    regulatory potential matrix as described in Browaeys, R., Saelens, W. &
    Saeys, Y. NicheNet: modeling intercellular communication by linking ligands
    to target genes. Nat. Methods 17, 159–162 (2020), and extract 1287 mouse or
    1226 human gene programs of ligands with their corresponding receptors and
    top target genes based on NicheNet regulatory potential scores.

    Parameters
    ----------
    species:
        Species for which the gps will be extracted. The default is human and, 
        if version is 'v1', human genes are mapped to mouse orthologs using a
        mapping file. NicheCompass contains a default mapping file stored under
        "<root>/data/gene_annotations/human_mouse_gene_orthologs.csv", which was
        created with Ensembl BioMart
        (http://www.ensembl.org/info/data/biomart/index.html).
    version:
        Version of the NicheNet ligand receptor network and ligand target gene
        regulatory potential matrix. ´v2´ is an improved version of ´v1´, and
        has separate files for mouse and human.
    keep_target_genes_ratio:
        Ratio of target genes that are kept compared to total target genes. This
        ratio is applied over the entire matrix (not on gene program level), and
        determines the ´all_gps_score_keep_threshold´, which will be used to
        filter target genes according to their regulatory potential scores.
    max_n_target_genes_per_gp:
        Maximum number of target genes per gene program. If a gene program has
        more target genes than ´max_n_target_genes_per_gp´, only the
        ´max_n_target_genes_per_gp´ gene programs with the highest regulatory
        potential scores will be kept. Default value is chosen based on
        MultiNicheNet specification (s. Browaeys, R. et al. MultiNicheNet: a
        flexible framework for differential cell-cell communication analysis
        from multi-sample multi-condition single-cell transcriptomics data.
        bioRxiv (2023) doi:10.1101/2023.06.13.544751).
    load_from_disk:
        If ´True´, the NicheNet files will be loaded from disk instead of the
        web.
    save_to_disk:
        If ´True´, the NicheNet files will additionally be stored on disk.
    lr_network_file_path:
        Path of the file where the NicheNet ligand receptor network will be
        stored (if ´save_to_disk´ is ´True´) or loaded from (if ´load_from_disk´
        is ´True´).
    ligand_target_matrix_file_path:
        Path of the file where the NicheNet ligand target matrix will be stored
        (if ´save_to_disk´ is ´True´) or loaded from (if ´load_from_disk´ is
        ´True´).
    gene_orthologs_mapping_file_path:
        Path of the file where the gene orthologs mapping is stored. Relevant if
        version is ´v1´ and species is ´mouse´.
    plot_gp_gene_count_distributions:
        If ´True´, display the distribution of gene programs per number of
        sources and targets.
    gp_gene_count_distributions_save_path:
        Path of the file where the gene program gene count distribution plot
        will be saved if ´plot_gp_gene_count_distributions´ is ´True´.

    Returns
    ----------
    gp_dict:
        Nested dictionary containing the NicheNet ligand receptor target gene 
        programs with keys being gene program names and values being 
        dictionaries with keys ´sources´, ´targets´, ´sources_categories´, and
        ´target_categories´, where ´sources´ contains the NicheNet ligands,
        ´targets´ contains the NicheNet receptors and target genes,
        ´sources_categories´ contains the categories of the sources, and
        ´target_categories´ contains the categories of the targets.
    """
    # Download (or load) NicheNet ligand receptor network and ligand target
    # matrix and store in df (optionally also on disk)
    if not load_from_disk:
        if version == "v1":
            lr_network_url = "https://zenodo.org/record/3260758/files/" \
                             "lr_network.rds"
            ligand_target_matrix_url = "https://zenodo.org/record/3260758/" \
                                       "files/ligand_target_matrix.rds"
        elif version == "v2" and species == "human":
            lr_network_url = "https://zenodo.org/record/7074291/files/" \
                             "lr_network_human_21122021.rds"
            ligand_target_matrix_url = "https://zenodo.org/record/7074291/" \
                                       "files/ligand_target_matrix_nsga2r_" \
                                       "final.rds"
        elif version == "v2" and species == "mouse":
            lr_network_url = "https://zenodo.org/record/7074291/files/" \
                             "lr_network_mouse_21122021.rds"
            ligand_target_matrix_url = "https://zenodo.org/record/7074291/" \
                                       "files/ligand_target_matrix_nsga2r_" \
                                       "final_mouse.rds"    
        print(f"Downloading NicheNet ligand receptor network '{version}' from "
              "the web...")
        lr_network_df = load_R_file_as_df(
            R_file_path="lr_network.rds",
            url=lr_network_url,
            save_df_to_disk=save_to_disk,
            df_save_path=lr_network_file_path) # multiple rows per ligand
        print(f"Downloading NicheNet ligand target matrix '{version}' from the "
              "web. This might take a while...")
        ligand_target_matrix_df = load_R_file_as_df(
            R_file_path="ligand_target_matrix.rds",
            url=ligand_target_matrix_url,
            save_df_to_disk=save_to_disk,
            df_save_path=ligand_target_matrix_file_path) # one column per ligand
    else:
        lr_network_df = pd.read_csv(lr_network_file_path,
                                    index_col=0) # multiple rows per ligand
        ligand_target_matrix_df = pd.read_csv(ligand_target_matrix_file_path,
                                              index_col=0) # one column per
                                                           # ligand
        
    # Group receptors by ligands to have one row per ligand
    grouped_lr_network_df = lr_network_df.groupby("from")["to"].agg(
        list).reset_index()

    # Filter ligand target matrix based on sorted potential / scores, using
    # ´keep_target_genes_ratio´ (over all gps) and ´max_n_target_genes_per_gp´
    # (over each gp separately). Each ligand (column) will make up one gp.
    # Store filter outputs as a mask dict where for each gp it is indicated
    # which genes are part of it
    per_gp_target_gene_scores = ligand_target_matrix_df.values.copy()
    all_target_gene_scores = np.squeeze(per_gp_target_gene_scores).flatten()
    per_gp_target_gene_scores_sorted = np.flip(
        np.sort(per_gp_target_gene_scores, axis=0), axis=0)
    per_gp_score_keep_threshold = pd.Series(
        per_gp_target_gene_scores_sorted[
            min(max_n_target_genes_per_gp, len(per_gp_target_gene_scores_sorted) - 1), :],
        index=ligand_target_matrix_df.columns)
    all_target_gene_scores.sort()
    all_target_gene_scores_sorted = np.flip(all_target_gene_scores)
    all_gps_score_keep_threshold = all_target_gene_scores_sorted[int(
        (len(all_target_gene_scores_sorted) - 1) * keep_target_genes_ratio)]
    ligand_target_all_gps_score_keep_threshold_mask_df = (
        ligand_target_matrix_df.applymap(
        lambda x: x > all_gps_score_keep_threshold))
    ligand_target_per_gp_score_keep_threshold_mask_df = (
        ligand_target_matrix_df.apply(
        lambda col: col > per_gp_score_keep_threshold[col.name], axis=0))
    ligand_target_combined_keep_threshold_mask_df = (
        ligand_target_all_gps_score_keep_threshold_mask_df &
        ligand_target_per_gp_score_keep_threshold_mask_df)
    
    # Extract ligands to build gene programs, add receptors and target genes,
    # and store in nested dict
    gp_dict = {}
    for ligand, gene_mask in ligand_target_combined_keep_threshold_mask_df.items():
        # Retrieve ligand receptors
        receptors = grouped_lr_network_df[
            grouped_lr_network_df["from"] == ligand]["to"].values[0]
        
        # Build gp dict using ligand in source node and receptors and target
        # genes in target node
        gp_dict[ligand + "_ligand_receptor_target_gene_GP"] = {
            "sources": [ligand],
            "targets": receptors +
                       [target for target, include in
                        gene_mask.items() if include & 
                        (target not in receptors)]} # don't duplicate receptors
        
        # Add source and target categories
        gp_dict[ligand + "_ligand_receptor_target_gene_GP"][
            "sources_categories"] = ["ligand"]
        gp_dict[ligand + "_ligand_receptor_target_gene_GP"][
            "targets_categories"] = (["receptor"] * len(receptors) +
                                     ["target_gene"] * (
            len(gp_dict[ligand + "_ligand_receptor_target_gene_GP"]["targets"]) -
            len(receptors)))
        
    if version == "v1" and species == "mouse":
        # Create mapping df to map from human genes to mouse orthologs
        mapping_df = pd.read_csv(gene_orthologs_mapping_file_path)
        grouped_mapping_df = mapping_df.groupby(
            "Gene name")["Mouse gene name"].agg(list).reset_index()
        
        # Map all genes in the gene programs to their orthologs from the mapping
        # df or capitalize them if no orthologs are found (one human gene can
        # have multiple mouse orthologs)
        for _, gp in gp_dict.items():
            gp["sources"] = [element for nested_list_l1 in [
                list_element for nested_list_l2 in [
                    grouped_mapping_df[
                        grouped_mapping_df["Gene name"] == source][
                            "Mouse gene name"].values.tolist() if
                            source in grouped_mapping_df["Gene name"].values else
                            [[source.capitalize()]] for source in gp["sources"]]
                            for list_element in nested_list_l2]
                            for element in nested_list_l1]
            gp["targets"] = [element for nested_list_l1 in [
                list_element for nested_list_l2 in [
                    grouped_mapping_df[
                        grouped_mapping_df["Gene name"] == target][
                            "Mouse gene name"].values.tolist() if
                            target in grouped_mapping_df["Gene name"].values else
                            [[target.capitalize()]] for target in gp["targets"]]
                            for list_element in nested_list_l2]
                            for element in nested_list_l1]
            gp["sources_categories"] = [element for nested_list_l1 in [
                list_element for nested_list_l2 in [
                    [source_category] * len(grouped_mapping_df[
                        grouped_mapping_df["Gene name"] == source][
                            "Mouse gene name"].values[0])
                            if source in grouped_mapping_df["Gene name"].values else
                            [source_category] for source, source_category in zip(
                                gp["sources"], gp["sources_categories"])]
                                for list_element in nested_list_l2]
                                for element in nested_list_l1]
            gp["targets_categories"] = [element for nested_list_l1 in [
                list_element for nested_list_l2 in [
                    [target_category] * len(grouped_mapping_df[
                        grouped_mapping_df["Gene name"] == target][
                            "Mouse gene name"].values[0])
                            if target in grouped_mapping_df["Gene name"].values else
                            [target_category] for target, target_category in zip(
                                gp["targets"], gp["targets_categories"])]
                                for list_element in nested_list_l2]
                                for element in nested_list_l1]
        
    if plot_gp_gene_count_distributions:
        create_gp_gene_count_distribution_plots(
            gp_dict=gp_dict,
            gp_plot_label=f"NicheNet {version.replace('_', ' ').title()}",
            save_path=gp_gene_count_distributions_save_path)
        
    return gp_dict


def extract_gp_dict_from_omnipath_lr_interactions(
        species: Literal["mouse", "human"],
        min_curation_effort: int=2,
        load_from_disk: bool=False,
        save_to_disk: bool=False,
        lr_network_file_path: Optional[str]="../data/gene_programs/" \
                                            "omnipath_lr_network.csv",
        gene_orthologs_mapping_file_path: Optional[str]="../data/gene_" \
                                                        "annotations/human_" \
                                                        "mouse_gene_orthologs.csv",
        plot_gp_gene_count_distributions: bool=True,
        gp_gene_count_distributions_save_path: Optional[str]=None) -> dict:
    """
    Retrieve 724 human ligand-receptor interactions from OmniPath and extract
    them into a gene program dictionary. OmniPath is a database of molecular
    biology prior knowledge that combines intercellular communication data from
    many different resources (all resources for intercellular communication
    included in OmniPath can be queried via
    ´op.requests.Intercell.resources()´). If ´species´ is ´mouse´, orthologs
    from human interactions are returned.

    Parts of the implementation are inspired by 
    https://workflows.omnipathdb.org/intercell-networks-py.html (01.10.2022).

    Parameters
    ----------
    species:
        Species for which the gene programs will be extracted. The default is
        human. Human genes are mapped to mouse orthologs using a mapping file.
        NicheCompass contains a default mapping file stored under
        "<root>/data/gene_annotations/human_mouse_gene_orthologs.csv", which was
        created with Ensembl BioMart
        (http://www.ensembl.org/info/data/biomart/index.html).
    min_curation_effort: 
        Indicates how many times an interaction has to be described in a 
        paper and mentioned in a database to be included in the retrieval.
    load_from_disk:
        If ´True´, the OmniPath ligand receptor interactions will be loaded from
        disk instead of from the OmniPath library.
    save_to_disk:
        If ´True´, the OmniPath ligand receptor interactions will additionally 
        be stored on disk. Only applies if ´load_from_disk´ is ´False´.
    lr_network_file_path:
        Path of the file where the OmniPath ligand receptor interactions will be
        stored (if ´save_to_disk´ is ´True´) or loaded from (if ´load_from_disk´
        is ´True´).
    gene_orthologs_mapping_file_path:
        Path of the file where the gene orthologs mapping is stored if species
        is ´mouse´.
    plot_gp_gene_count_distributions:
        If ´True´, display the distribution of gene programs per number of
        source and target genes.
    gp_gene_count_distributions_save_path:
        Path of the file where the gene program gene count distribution plot
        will be saved if ´plot_gp_gene_count_distributions´ is ´True´.

    Returns
    ----------
    gp_dict:
        Nested dictionary containing the OmniPath ligand-receptor interaction
        gene programs with keys being gene program names and values being
        dictionaries with keys ´sources´, ´targets´, ´sources_categories´, and
        ´targets_categories´, where ´sources´ contains the OmniPath ligands,
        ´targets´ contains the OmniPath receptors, ´sources_categories´ contains
        the categories of the sources, and ´targets_categories´ contains
        the categories of the targets.
    """
    if not load_from_disk:
        # Define intercell_network categories to be retrieved (see
        # https://workflows.omnipathdb.org/intercell-networks-py.html,
        # https://omnipath.readthedocs.io/en/latest/api/omnipath.interactions.import_intercell_network.html#omnipath.interactions.import_intercell_network)
        intercell_df = op.interactions.import_intercell_network(
            include=["omnipath", "pathwayextra", "ligrecextra"])
        lr_interaction_df = intercell_df[
            (intercell_df["category_intercell_source"] == "ligand")
            & (intercell_df["category_intercell_target"] == "receptor")]
        if save_to_disk:
            lr_interaction_df.to_csv(lr_network_file_path, index=False)
    else:
        lr_interaction_df = pd.read_csv(lr_network_file_path, index_col=0)

    # Only keep curated interactions (see
    # https://r.omnipathdb.org/reference/filter_intercell_network.html)
    lr_interaction_df = lr_interaction_df[
        lr_interaction_df["curation_effort"] >= min_curation_effort]

    # Group receptors by ligands
    grouped_lr_interaction_df = lr_interaction_df.groupby(
        "genesymbol_intercell_source")["genesymbol_intercell_target"].agg(
            list).reset_index()
    
    # Resolve protein complexes into individual genes
    def compute_elementwise_func(lst, func):
        return [func(item) for item in lst]

    def resolve_protein_complexes(x):
        if x:
            if "COMPLEX:" not in x:
                return [x]
            else:
                return x.removeprefix("COMPLEX:").split("_")
        else:
            return []
        
    grouped_lr_interaction_df["sources"] = grouped_lr_interaction_df[
        "genesymbol_intercell_source"].apply(
            lambda x: list(set(resolve_protein_complexes(x))))
    grouped_lr_interaction_df["sources_categories"] = grouped_lr_interaction_df[
        "sources"].apply(lambda x: ["ligand"] * len(x))
    grouped_lr_interaction_df["targets"] = grouped_lr_interaction_df[
        "genesymbol_intercell_target"].apply(
            lambda x: list(set([element for sublist in compute_elementwise_func(x, resolve_protein_complexes) for element in sublist])))
    grouped_lr_interaction_df["targets_categories"] = grouped_lr_interaction_df[
        "targets"].apply(lambda x: ["receptor"] * len(x))

    # Extract gene programs and store in nested dict
    gp_dict = {}
    for _, row in grouped_lr_interaction_df.iterrows():
        gp_dict[row["genesymbol_intercell_source"] +
                "_ligand_receptor_GP"] = {
                    "sources": row["sources"],
                    "targets": row["targets"],
                    "sources_categories": row["sources_categories"],
                    "targets_categories": row["targets_categories"]}
        
    if species == "mouse":
        # Create mapping df to map from human genes to mouse orthologs
        mapping_df = pd.read_csv(gene_orthologs_mapping_file_path)
        grouped_mapping_df = mapping_df.groupby(
            "Gene name")["Mouse gene name"].agg(list).reset_index()
        
        # Map all genes in the gene programs to their orthologs from the mapping
        # df or capitalize them if no orthologs are found (one human gene can
        # have multiple mouse orthologs)
        for _, gp in gp_dict.items():
            gp["sources"] = [element for nested_list_l1 in [
                list_element for nested_list_l2 in [
                    grouped_mapping_df[
                        grouped_mapping_df["Gene name"] == source][
                            "Mouse gene name"].values.tolist() if
                            source in grouped_mapping_df["Gene name"].values else
                            [[source.capitalize()]] for source in gp["sources"]]
                            for list_element in nested_list_l2]
                            for element in nested_list_l1]
            gp["targets"] = [element for nested_list_l1 in [
                list_element for nested_list_l2 in [
                    grouped_mapping_df[
                        grouped_mapping_df["Gene name"] == target][
                            "Mouse gene name"].values.tolist() if
                            target in grouped_mapping_df["Gene name"].values else
                            [[target.capitalize()]] for target in gp["targets"]]
                            for list_element in nested_list_l2]
                            for element in nested_list_l1]
            gp["sources_categories"] = ["ligand"] * len(gp["sources"])
            gp["targets_categories"] = ["receptor"] * len(gp["targets"])
    
    if plot_gp_gene_count_distributions:
        create_gp_gene_count_distribution_plots(
            gp_dict=gp_dict,
            gp_plot_label="OmniPath",
            save_path=gp_gene_count_distributions_save_path)
        
    return gp_dict


def extract_gp_dict_from_mebocost_ms_interactions(
        species: Literal["mouse", "human"],
        dir_path: str="../data/gene_programs/metabolite_enzyme_sensor_gps",
        plot_gp_gene_count_distributions: bool=True,
        gp_gene_count_distributions_save_path: Optional[str]=None) -> dict:
    """
    Retrieve 115 mouse or 116 human metabolite-sensor interactions based on the
    Human Metabolome Database (HMDB) data curated in Chen, K. et al. MEBOCOST:
    Metabolite-mediated cell communication modeling by single cell
    transcriptome. Research Square (2022) doi:10.21203/rs.3.rs-2092898/v1.
    Gene expression of enzymes involved in reactions with metabolite products is
    used as proxy for metabolite presence.
    
    This data is available in the NicheCompass package under 
    ´..data/gene_programs/metabolite_enzyme_sensor_gps´.

    Parameters
    ----------
    species:
        Species for which to retrieve metabolite-sensor interactions.
    dir_path:
        Path of the directory where the metabolite gene programs are stored.
    plot_gp_gene_count_distributions:
        If ´True´, display the distribution of gene programs per number of
        target and source genes.
    gp_gene_count_distributions_save_path:
        Path of the file where the gene program gene count distribution plot
        will be saved if ´plot_gp_gene_count_distributions´ is ´True´.

    Returns
    ----------
    gp_dict:
        Nested dictionary containing the MEBOCOST metabolite-sensor interaction
        gene programs with keys being gene program names and values being 
        dictionaries with keys ´sources´, ´targets´, ´sources_categories´, and
        ´targets_categories´, where ´sources´ contains the MEBOCOST enzymes,
        ´targets´ contains the MEBOCOST sensors, ´sources_categories´ contains
        the categories of the sources, and ´targets_categories´ contains
        the categories of the targets.
    """
    # Read data from directory
    if species == "human":
        metabolite_enzymes_df = pd.read_csv(
            dir_path + "/human_metabolite_enzymes.tsv", sep="\t")
        metabolite_sensors_df = pd.read_csv(
            dir_path + "/human_metabolite_sensors.tsv", sep="\t")
    elif species == "mouse":
        metabolite_enzymes_df = pd.read_csv(
            dir_path + "/mouse_metabolite_enzymes.tsv", sep="\t")
        metabolite_sensors_df = pd.read_csv(
            dir_path + "/mouse_metabolite_sensors.tsv", sep="\t")
    else:
        raise ValueError("Species should be either human or mouse.")

    # Retrieve metabolite names
    metabolite_names_df = (metabolite_sensors_df[["HMDB_ID",
                                                  "standard_metName"]]
                           .drop_duplicates()
                           .set_index("HMDB_ID"))

    # Keep only enzymes for which the metabolite is the product (filter enzymes
    # for which the metabolite is the substrate)
    metabolite_enzymes_df = metabolite_enzymes_df[
        metabolite_enzymes_df["direction"] == "product"]

    # Retrieve metabolite enzyme and sensor genes
    metabolite_enzymes_unrolled = []
    for _, row in metabolite_enzymes_df.iterrows():
        genes = row["gene"].split("; ")
        for gene in genes:
            tmp = row.copy()
            tmp["gene"] = gene
            metabolite_enzymes_unrolled.append(tmp)
    metabolite_enzymes_df = pd.DataFrame(metabolite_enzymes_unrolled)
    metabolite_enzymes_df["gene_name"] = metabolite_enzymes_df["gene"].apply(
        lambda x: x.split("[")[0])
    metabolite_enzymes_df = (metabolite_enzymes_df.groupby(["HMDB_ID"])
                             .agg({"gene_name": lambda x: sorted(
                                x.unique().tolist())})
                             .rename({"gene_name": "enzyme_genes"}, axis=1)
                             .reset_index()).set_index("HMDB_ID")
    metabolite_sensors_df = (metabolite_sensors_df.groupby(["HMDB_ID"])
                             .agg({"Gene_name": lambda x: sorted(
                                x.unique().tolist())})
                             .rename({"Gene_name": "sensor_genes"}, axis=1)
                             .reset_index()).set_index("HMDB_ID")

    # Combine enzyme and sensor genes based on metabolite names (sensor genes
    # are not available for most metabolites)
    metabolite_df = metabolite_enzymes_df.join(
        other=metabolite_sensors_df,
        how="inner").join(metabolite_names_df).set_index("standard_metName")

    # Convert to gene program dictionary format
    met_interaction_dict = metabolite_df.to_dict()
    gp_dict = {}
    for metabolite, enzyme_genes in met_interaction_dict["enzyme_genes"].items():
        gp_dict[metabolite + "_metabolite_enzyme_sensor_GP"] = {
            "sources": enzyme_genes,
            "sources_categories": ["enzyme"] * len(enzyme_genes)}
    for metabolite, sensor_genes in met_interaction_dict["sensor_genes"].items():
        gp_dict[metabolite + "_metabolite_enzyme_sensor_GP"][
            "targets"] = sensor_genes
        gp_dict[metabolite + "_metabolite_enzyme_sensor_GP"][
            "targets_categories"] = ["sensor"] * len(sensor_genes)

    if plot_gp_gene_count_distributions:
        create_gp_gene_count_distribution_plots(
            gp_dict=gp_dict,
            gp_plot_label="MEBOCOST",
            save_path=gp_gene_count_distributions_save_path)

    return gp_dict


def _download_and_load_humanppi_predictions(
        precision: Literal["90", "80"],
        url: str) -> pd.DataFrame:
    """
    Helper to download the human interactome predictions tarball from the web
    and load the requested precision table into a pandas DataFrame.

    The download server may present an incomplete TLS certificate chain; in
    that case the download is retried with certificate verification disabled
    (with a warning), as the underlying data is a public research resource.

    Parameters
    ----------
    precision:
        Expected precision of the predicted interactions to load ('90' or
        '80'), corresponding to the ´final_predictions_90.tsv´ and
        ´final_predictions_80.tsv´ files respectively.
    url:
        URL of the ´final_predictions.tar.gz´ archive.

    Returns
    ----------
    ppi_df:
        Predicted protein-protein interactions loaded into a pandas DataFrame.
    """
    target_file = f"final_predictions_{precision}.tsv"
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_tar_path = os.path.join(tmp_dir, "final_predictions.tar.gz")
        try:
            with urllib.request.urlopen(url) as response, \
                    open(tmp_tar_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)
        except (ssl.SSLError, urllib.error.URLError):
            warnings.warn(
                "Could not verify the TLS certificate of the human "
                f"interactome download server ('{url}'). Retrying the "
                "download with certificate verification disabled.")
            unverified_context = ssl._create_unverified_context()
            with urllib.request.urlopen(
                    url, context=unverified_context) as response, \
                    open(tmp_tar_path, "wb") as out_file:
                shutil.copyfileobj(response, out_file)

        with tarfile.open(tmp_tar_path, "r:gz") as tar:
            member = next((m for m in tar.getmembers()
                           if m.name.endswith(target_file)), None)
            if member is None:
                raise ValueError(
                    f"Could not find '{target_file}' in the downloaded human "
                    "interactome archive.")
            extracted_file = tar.extractfile(member)
            # The tsv is preceded by comment lines (starting with '#') that
            # describe each column; the first non-comment line is the header.
            ppi_df = pd.read_csv(extracted_file, sep="\t", comment="#")
    return ppi_df


def extract_gp_dict_from_humanppi_interactions(
        species: Literal["mouse", "human"],
        precision: Literal["90", "80"]="90",
        program_type: Literal[
            "intercellular", "intracellular", "both"]="intercellular",
        unknown_locality: Literal["exclude", "intracellular"]="exclude",
        filter_ig_tcr_segments: bool=True,
        filter_paralog_cross_pairs: bool=True,
        use_topology: bool=True,
        topology_file_path: Optional[str]="../data/gene_programs/" \
                                          "humanppi_protein_topology.tsv",
        detect_cis_complexes: bool=True,
        complex_portal_file_path: Optional[str]="../data/gene_programs/" \
                                                "complex_portal_human.tsv",
        complex_portal_url: str="https://ftp.ebi.ac.uk/pub/databases/intact/" \
                                "complex/current/complextab/9606.tsv",
        min_rf_prob: Optional[float]=None,
        min_af_prob: Optional[float]=None,
        load_from_disk: bool=False,
        save_to_disk: bool=False,
        ppi_network_file_path: Optional[str]="../data/gene_programs/" \
                                             "humanppi_network.csv",
        humanppi_predictions_url: str="https://conglab.swmed.edu/humanPPI/" \
                                      "downloads/final_predictions.tar.gz",
        gene_orthologs_mapping_file_path: Optional[str]="../data/gene_" \
                                                        "annotations/human_" \
                                                        "mouse_gene_orthologs.csv",
        plot_gp_gene_count_distributions: bool=True,
        gp_gene_count_distributions_save_path: Optional[str]=None) -> dict:
    """
    Retrieve predicted human protein-protein interactions from the human
    interactome resource described in Zhang, J., Humphreys, I. R. et al.
    Predicting protein-protein interactions in the human proteome. Science
    (2025) doi:10.1126/science.adt1630, and extract them into a gene program
    dictionary. The predictions were generated with RoseTTAFold2-PPI and
    AlphaFold2 by screening ~190 million human protein pairs, and are
    distributed at two expected precision levels (17,849 interactions at 90%
    precision and 29,257 at 80% precision). The data is archived on Dryad
    (doi:10.5061/dryad.15dv41p84) and additionally available for direct
    download from https://conglab.swmed.edu/humanPPI/.

    Each interaction is classified before it is turned into a gene program.
    First, both partners are assigned a location class from their UniProt
    cellular-component keywords:

    - ´cell_surface´: membrane anchored with a necessarily extracellular face
      (´Cell membrane´, ´MHC I´, ´MHC II´, ´T cell receptor´, ...).
    - ´secreted´: released into the extracellular space and not membrane
      anchored (´Secreted´, ´Extracellular matrix´, lipoprotein particles,
      ...).
    - ´intracellular´: an intracellular location and no surface or secreted
      keyword (´Nucleus´, ´Mitochondrion´, ´Cytoskeleton´, ...).
    - ´ambiguous´: compatible with an extracellular face but not establishing
      one, either because the keyword is the generic parent of the whole
      membrane branch (´Membrane´), or because the compartment has a large
      cytoplasmic component (´Cell junction´ is carried by cadherins but also
      by cytosolic catenins, vinculin and ZO-1), or because it describes a
      property rather than a location (´Amyloid´). Such proteins count as
      extracellular facing, since excluding them was measured to discard a
      large amount of genuine biology; ´use_topology´ resolves them properly.
    - ´unknown´: no usable keyword at all.

    Evidence for an extracellular face takes precedence over evidence for an
    intracellular location, since secreted and surface proteins are frequently
    also annotated with the compartments they traverse.

    Second, the interaction is classified from the two location classes:

    - ´paracrine´: both partners are extracellular facing and at least one is
      secreted, i.e. signaling through a diffusible partner.
    - ´juxtacrine´: both partners are membrane anchored and extracellular
      facing, i.e. contact dependent signaling.
    - ´intracellular´: anything else, i.e. at least one partner has no
      extracellular face.
    - ´unknown´: at least one partner could not be classified.

    Interactions between two secreted subunits of a common complex are
    classified as ´extracellular_assembly´ rather than paracrine, since
    secreted multimers such as collagen and laminin trimers, fibrinogen and
    complement C1q are assembled by the cell that produces them.

    If ´use_topology´ is ´True´, a cell-surface keyword is only trusted for
    proteins that UniProt annotates with a transmembrane segment or a GPI
    anchor. Cellular-component keywords describe whole proteins and say nothing
    about which side of a membrane a protein faces, so peripheral proteins
    docked onto the cytoplasmic leaflet of the plasma membrane carry the
    ´Cell membrane´ keyword as well; the neuronal SNARE complex, the protein
    kinase A holoenzyme, adducin, the calpains and the cytoplasmic adherens
    plaque are all of this kind. Membrane anchoring is a sequence-level
    property and resolves those cases. Secreted keywords are always trusted,
    since proteins can also be released through non-classical routes that leave
    no sequence signature.

    Third, if ´detect_cis_complexes´ is ´True´, interactions that were called
    paracrine or juxtacrine are checked against the human protein complexes of
    the EBI Complex Portal. Two subunits of a common complex assemble within a
    single cell, so such an interaction is not intercellular even when both
    partners are located at the cell surface (the T cell receptor chains, the
    CD8 heterodimer, integrin alpha-beta heterodimers and the MHC class II
    alpha-beta heterodimer are all of this kind). These interactions are
    reclassified as ´cis_complex´ and placed in the target component like
    intracellular ones. Note that the Complex Portal also registers genuine
    ligand-receptor assemblies as complexes; those are recognized by pairing a
    secreted with a membrane-anchored partner and are retained as paracrine.
    The Complex Portal is used only to reclassify interactions of the human
    interactome, never to add interactions.

    NicheCompass gene programs have a source component (genes reconstructed in
    a node's neighbors, i.e. the transmitting cells) and a target/self
    component (genes reconstructed in the node itself, i.e. the receiving
    cell). Physical interactions can therefore be represented in two ways,
    controlled by ´program_type´:
    - As intercellular gene programs, where the two partners are placed in the
      source and target component respectively, modeling contact-dependent or
      paracrine signaling between neighboring cells. This is only meaningful
      for partners that can act between cells, so such programs are restricted
      to cell-surface / secreted / extracellular proteins (based on the UniProt
      cellular-component keywords provided with the predictions). Physical
      interactions are undirected, so the source/target assignment is arbitrary
      but consistent.
    - As intracellular gene programs, where both partners are placed in the
      target/self component (with an empty source component), modeling a within
      -cell protein complex / co-expression module. This mirrors the structure
      of the CollecTRI transcription-factor programs (see
      ´extract_gp_dict_from_collectri_tf_network´) and is consistent with the
      self component representing the target of intercellular OR intracellular
      interactions.

    Note that intracellular (target-only) gene programs have an empty source
    component and are therefore discarded by ´add_gps_from_gp_dict_to_adata´
    unless it is called with ´min_source_genes_per_gp=0´ (as is also required
    for the CollecTRI transcription-factor programs). A warning is emitted when
    such programs are produced.

    Parameters
    ----------
    species:
        Species for which the gene programs will be extracted. The predictions
        are human; if ´mouse´, human genes are mapped to their mouse orthologs
        using a mapping file. NicheCompass contains a default mapping file
        stored under
        "<root>/data/gene_annotations/human_mouse_gene_orthologs.csv", which
        was created with Ensembl BioMart
        (http://www.ensembl.org/info/data/biomart/index.html).
    precision:
        Expected precision of the predicted interactions to use ('90' or '80').
        '90' (default) uses the high-confidence set of 17,849 interactions,
        '80' uses the broader set of 29,257 interactions. Note that a small
        number of interactions are listed twice, and that interactions
        involving a protein without a UniProt gene name are dropped, so the
        number of gene programs is lower than the number of table rows (17,809
        and 29,191 unique gene pairs respectively).
    program_type:
        Determines which interactions are retained and how they are placed into
        gene program components. If ´intercellular´ (default), only paracrine
        and juxtacrine interactions are kept, and each is turned into a
        source-to-target gene program. If ´intracellular´, only the remaining
        interactions are kept (both intracellular and ´cis_complex´ ones), and
        each is turned into a target-only gene program (empty source
        component). If ´both´, all classified
        interactions are kept and placed into the appropriate component.
        Intracellular (target-only) programs require ´min_source_genes_per_gp=0´
        downstream (see above).
    unknown_locality:
        Determines how interactions are handled in which at least one partner
        has no usable localization annotation, which is the case for around a
        tenth of the proteins in the resource. If ´exclude´ (default), these
        interactions are dropped, since they cannot be classified either way.
        If ´intracellular´, they are treated as intracellular, which retains
        them at the risk of mislabeling genuine intercellular interactions.
    filter_paralog_cross_pairs:
        If ´True´ (default), interactions between paralogues that do not form a
        heterodimer are dropped. Integrin alpha-beta pairs are restricted to
        the 24 heterodimers that exist and MHC class II alpha-beta pairs to
        matching isotypes. The structural predictor produces such combinations
        because close paralogues have interchangeable interfaces.
    filter_ig_tcr_segments:
        If ´True´ (default), interactions involving an immunoglobulin or T cell
        receptor V, D or J gene segment are dropped. These segments are
        somatically recombined into a single antibody or receptor chain, so such
        an interaction is intramolecular rather than an interaction between two
        proteins. They are abundant in the predictions because all variable
        domains share the immunoglobulin fold. Constant region genes such as
        ´IGHG1´ and ´TRAC´ encode complete proteins and are retained.
    use_topology:
        If ´True´ (default), membrane topology is retrieved from UniProt and a
        cell-surface keyword is only trusted for proteins that have a
        transmembrane segment or a GPI anchor.
    topology_file_path:
        Path of the file where the retrieved membrane topology is cached. Only
        accessions missing from the cache are requested from UniProt.
    detect_cis_complexes:
        If ´True´ (default), interactions whose two partners are subunits of a
        common protein complex in the EBI Complex Portal, or that belong to a
        common curated gene family (see
        ´HUMANPPI_CIS_COMPLEX_GENE_FAMILY_PATTERNS´), are reclassified from
        paracrine or juxtacrine to ´cis_complex´, since they assemble within a
        single cell. The curated families are needed because the Complex Portal
        covers only around a fifth of the proteins of the human interactome and
        is missing several prominent cell-surface complexes entirely. Measured on the predictions at 90% precision, this affects
        around a tenth of the interactions that would otherwise be called
        intercellular.
    complex_portal_file_path:
        Path of the file where the Complex Portal table for human is cached.
        The file is downloaded on first use.
    complex_portal_url:
        URL of the Complex Portal table for human, used if the cached file does
        not exist.
    min_rf_prob:
        If not ´None´, only interactions with a RoseTTAFold2-PPI interaction
        probability (´RFprob´) greater than or equal to this value are kept.
        The predictions are already precision-filtered, so this is an optional
        additional filter.
    min_af_prob:
        If not ´None´, only interactions with an AlphaFold2 interaction
        probability (´AFprob´) greater than or equal to this value are kept.
    load_from_disk:
        If ´True´, the human PPI network will be loaded from disk instead of
        from the web.
    save_to_disk:
        If ´True´, the human PPI network will additionally be stored on disk.
        Only applies if ´load_from_disk´ is ´False´.
    ppi_network_file_path:
        Path of the file where the human PPI network will be stored (if
        ´save_to_disk´ is ´True´) or loaded from (if ´load_from_disk´ is
        ´True´).
    humanppi_predictions_url:
        URL of the ´final_predictions.tar.gz´ archive to download if
        ´load_from_disk´ is ´False´.
    gene_orthologs_mapping_file_path:
        Path of the file where the gene orthologs mapping is stored if species
        is ´mouse´.
    plot_gp_gene_count_distributions:
        If ´True´, display the distribution of gene programs per number of
        source and target genes.
    gp_gene_count_distributions_save_path:
        Path of the file where the gene program gene count distribution plot
        will be saved if ´plot_gp_gene_count_distributions´ is ´True´.

    Returns
    ----------
    gp_dict:
        Nested dictionary containing the human PPI gene programs with keys being
        gene program names and values being dictionaries with keys ´sources´,
        ´targets´, ´sources_categories´, and ´targets_categories´. For
        intercellular programs, ´sources´ contains the first interacting protein
        and ´targets´ the second; for intracellular programs, ´sources´ is empty
        and ´targets´ contains both interacting proteins.

        The interaction class is part of the gene program name, which is
        ´<gene 1>_<gene 2>_<interaction class>_ppi_GP´ with the interaction
        class being ´paracrine´, ´juxtacrine´, ´cis_complex´,
        ´extracellular_assembly´ or ´intracellular´. It therefore
        remains visible in gene program summaries, differential gene program
        test results and plots. The gene categories are the location classes of
        the corresponding proteins (´secreted´, ´cell_surface´, ´ambiguous´ or
        ´intracellular´), so they can additionally be used to regularize or
        select genes by location via ´l1_targets_categories´ and
        ´l1_sources_categories´.
    """
    if precision not in ("90", "80"):
        raise ValueError("´precision´ should be either '90' or '80'.")
    if program_type not in ("intercellular", "intracellular", "both"):
        raise ValueError("´program_type´ should be one of 'intercellular', "
                         "'intracellular', or 'both'.")
    if unknown_locality not in ("exclude", "intracellular"):
        raise ValueError("´unknown_locality´ should be either 'exclude' or "
                         "'intracellular'.")

    # Download (or load) the human interactome predictions and store in df
    # (optionally also on disk)
    if not load_from_disk:
        print("Downloading human interactome predictions "
              f"(precision '{precision}') from the web...")
        ppi_df = _download_and_load_humanppi_predictions(
            precision=precision,
            url=humanppi_predictions_url)
        if save_to_disk:
            ppi_df.to_csv(ppi_network_file_path, sep="\t", index=False)
    else:
        ppi_df = pd.read_csv(ppi_network_file_path, sep="\t")

    # Drop interactions without a gene symbol for either partner. Note that the
    # predictions use the literal string 'none' as a placeholder for missing
    # values, so interactions involving a protein whose UniProt entry has no
    # gene name are removed here as well. Such interactions can never be
    # matched to measured genes, and because all unnamed proteins share the
    # same placeholder they would additionally be conflated into a single
    # spurious gene.
    ppi_df = ppi_df.dropna(subset=["Name1", "Name2"])
    gene_name_missing = pd.Series(False, index=ppi_df.index)
    for gene_name_col in ["Name1", "Name2"]:
        gene_names = ppi_df[gene_name_col].astype(str).str.strip()
        gene_name_missing |= ((gene_names.str.len() == 0) |
                              (gene_names.str.lower() == "none"))
    ppi_df = ppi_df[~gene_name_missing]

    # Optionally drop immunoglobulin and T cell receptor gene segments
    if filter_ig_tcr_segments:
        is_segment = (ppi_df["Name1"].astype(str).apply(
                          _is_humanppi_ig_tcr_segment) |
                      ppi_df["Name2"].astype(str).apply(
                          _is_humanppi_ig_tcr_segment))
        n_segment = int(is_segment.sum())
        if n_segment > 0:
            print(f"Dropped {n_segment} interactions involving an "
                  "immunoglobulin or T cell receptor gene segment, which are "
                  "recombined into a single chain and therefore do not "
                  "represent interactions between two proteins.")
        ppi_df = ppi_df[~is_segment]

    # Optionally drop paralogue combinations that do not form
    if filter_paralog_cross_pairs:
        is_cross_pair = ppi_df.apply(
            lambda row: _is_humanppi_paralog_cross_pair(str(row["Name1"]),
                                                        str(row["Name2"])),
            axis=1)
        n_cross_pair = int(is_cross_pair.sum())
        if n_cross_pair > 0:
            print(f"Dropped {n_cross_pair} interactions between paralogues "
                  "that do not form a heterodimer, which the structural "
                  "predictor produces because close paralogues have "
                  "interchangeable interfaces.")
        ppi_df = ppi_df[~is_cross_pair]

    # Optionally filter by interaction probabilities
    if min_rf_prob is not None:
        ppi_df = ppi_df[pd.to_numeric(ppi_df["RFprob"], errors="coerce")
                        >= min_rf_prob]
    if min_af_prob is not None:
        ppi_df = ppi_df[pd.to_numeric(ppi_df["AFprob"], errors="coerce")
                        >= min_af_prob]

    # Optionally retrieve membrane topology, used to reject cell-surface
    # keywords for proteins that have no membrane anchor
    topology = {}
    if use_topology:
        accessions = sorted(set(ppi_df["Protein1"].astype(str)) |
                            set(ppi_df["Protein2"].astype(str)))
        topology = _load_humanppi_protein_topology(
            accessions=accessions,
            topology_file_path=topology_file_path)

    # Optionally load the protein complexes used to detect cis interactions
    cis_pairs = None
    if detect_cis_complexes:
        cis_pairs = _load_complex_portal_cis_pairs(
            complex_portal_file_path=complex_portal_file_path,
            complex_portal_url=complex_portal_url)

    # Extract gene programs and store in nested dict (deduplicate symmetric and
    # repeated interactions based on the unordered gene pair). Each protein is
    # first assigned a location class, from which the interaction is classified
    # as paracrine, juxtacrine, intracellular or unknown. The interaction class
    # determines both the gene program components and the gene program name, so
    # that it remains visible in downstream gene program summaries, and the
    # location classes are used as the gene categories.
    gp_dict = {}
    seen_pairs = set()
    produced_source_empty_gp = False
    n_unknown_locality = 0
    n_cis_complex = 0
    n_extracellular_assembly = 0
    for _, row in ppi_df.iterrows():
        gene_1 = str(row["Name1"])
        gene_2 = str(row["Name2"])
        pair_key = frozenset((gene_1.upper(), gene_2.upper()))
        if pair_key in seen_pairs:
            continue

        location_1 = _classify_humanppi_protein_location(
            row["Locality1"],
            topology=topology.get(str(row["Protein1"])),
            process=row["Process1"] if "Process1" in row else None)
        location_2 = _classify_humanppi_protein_location(
            row["Locality2"],
            topology=topology.get(str(row["Protein2"])),
            process=row["Process2"] if "Process2" in row else None)
        interaction_class = _classify_humanppi_interaction(
            location_1, location_2)

        if interaction_class == "unknown":
            # At least one partner has no usable localization annotation, so
            # the interaction cannot be classified either way
            n_unknown_locality += 1
            if unknown_locality == "exclude":
                continue
            interaction_class = "intracellular"

        if (cis_pairs is not None and
                interaction_class in ("paracrine", "juxtacrine")):
            # Two subunits of a common complex assemble within one cell, so
            # they are not intercellular even when both are at the cell
            # surface. The Complex Portal also registers genuine ligand
            # receptor assemblies as complexes, which are recognized by pairing
            # a secreted with a membrane anchored partner and are kept.
            accession_pair = frozenset((str(row["Protein1"]),
                                        str(row["Protein2"])))
            shares_curated_family = bool(
                _humanppi_cis_complex_gene_families(gene_1) &
                _humanppi_cis_complex_gene_families(gene_2))
            # A shared complex in which exactly one partner is soluble is a
            # ligand-receptor assembly rather than a cis complex: the ligand is
            # released by one cell and engages the receptor of another. Both
            # partners being soluble is not sufficient, since secreted
            # multimers such as collagen and laminin trimers assemble within
            # the cell that produces them.
            is_ligand_receptor_assembly = (
                (location_1 == "secreted") != (location_2 == "secreted"))
            if ((accession_pair in cis_pairs or shares_curated_family)
                    and not is_ligand_receptor_assembly):
                if location_1 == "secreted" and location_2 == "secreted":
                    # Secreted multimers such as collagen and laminin trimers,
                    # fibrinogen and complement C1q are assembled by the cell
                    # that produces them, so they are not signaling between two
                    # cells even though they act outside the cell
                    interaction_class = "extracellular_assembly"
                    n_extracellular_assembly += 1
                else:
                    interaction_class = "cis_complex"
                    n_cis_complex += 1

        is_intercellular = interaction_class in ("paracrine", "juxtacrine")
        if is_intercellular and program_type == "intracellular":
            continue
        if not is_intercellular and program_type == "intercellular":
            continue
        seen_pairs.add(pair_key)

        if is_intercellular:
            # Intercellular program: partners split across neighbor (source)
            # and self (target) components
            gp_dict[f"{gene_1}_{gene_2}_{interaction_class}_ppi_GP"] = {
                "sources": [gene_1],
                "targets": [gene_2],
                "sources_categories": [location_1],
                "targets_categories": [location_2]}
        else:
            # Intracellular program: both partners in the self (target)
            # component, empty source component (as for CollecTRI TF programs)
            produced_source_empty_gp = True
            gp_dict[f"{gene_1}_{gene_2}_{interaction_class}_ppi_GP"] = {
                "sources": [],
                "targets": [gene_1, gene_2],
                "sources_categories": [],
                "targets_categories": [location_1, location_2]}

    if species == "mouse":
        # Create mapping to map from human genes to mouse orthologs
        mapping_df = pd.read_csv(gene_orthologs_mapping_file_path)
        grouped_mapping_df = mapping_df.groupby(
            "Gene name")["Mouse gene name"].agg(list).reset_index()
        ortholog_map = dict(zip(grouped_mapping_df["Gene name"],
                                grouped_mapping_df["Mouse gene name"]))

        def map_to_mouse_orthologs(gene: str) -> list:
            # One human gene can have multiple mouse orthologs; capitalize if no
            # (valid) ortholog is found (consistent with the other GP resources).
            # Some human genes are present in the mapping file but only with
            # missing (NaN) mouse gene names, which are filtered out here.
            orthologs = [o for o in ortholog_map.get(gene, [])
                         if isinstance(o, str) and o]
            return orthologs if orthologs else [gene.capitalize()]

        # Map every gene in each program (handles empty source components and
        # multi-gene target components of intracellular programs). One human
        # gene can have several mouse orthologs, so the gene category is
        # repeated for each ortholog.
        for _, gp in gp_dict.items():
            for entity in ("sources", "targets"):
                mapped_genes = []
                mapped_categories = []
                for gene, category in zip(gp[entity],
                                          gp[f"{entity}_categories"]):
                    orthologs = map_to_mouse_orthologs(gene)
                    mapped_genes.extend(orthologs)
                    mapped_categories.extend([category] * len(orthologs))
                gp[entity] = mapped_genes
                gp[f"{entity}_categories"] = mapped_categories

    if n_extracellular_assembly > 0:
        print(f"Reclassified {n_extracellular_assembly} interactions as "
              "'extracellular_assembly' because both partners are secreted "
              "subunits of a common complex, which is assembled by the cell "
              "that produces it.")

    if n_cis_complex > 0:
        print(f"Reclassified {n_cis_complex} interactions as 'cis_complex' "
              "because both partners are subunits of a common protein complex "
              "and the interaction therefore takes place within a single "
              "cell.")

    if n_unknown_locality > 0:
        print(f"Encountered {n_unknown_locality} interactions in which at "
              "least one partner has no usable localization annotation. These "
              + ("were excluded." if unknown_locality == "exclude" else
                 "were treated as intracellular."))

    if produced_source_empty_gp:
        warnings.warn(
            "Produced intracellular (target-only) human PPI gene programs with "
            "an empty source component. Call ´add_gps_from_gp_dict_to_adata´ "
            "with ´min_source_genes_per_gp=0´ so that these programs are not "
            "discarded (as is also required for the CollecTRI transcription-"
            "factor programs).")

    if plot_gp_gene_count_distributions:
        create_gp_gene_count_distribution_plots(
            gp_dict=gp_dict,
            gp_plot_label="HumanPPI",
            save_path=gp_gene_count_distributions_save_path)

    return gp_dict


def filter_and_combine_gp_dict_gps(
        gp_dict: dict,
        gp_filter_mode: Optional[Literal["subset", "superset"]]=None,
        combine_overlap_gps: bool=True,
        overlap_thresh_source_genes: float=1.,
        overlap_thresh_target_genes: float=1.,
        overlap_thresh_genes: float=1.,
        verbose: bool=False) -> dict:
    """
    Filter and combine the gene programs in a gene program dictionary based on
    overlapping genes.

    Parameters
    ----------
    gp_dict:
        Nested dictionary containing the gene programs with keys being gene 
        program names and values being dictionaries with keys ´targets´ and 
        ´sources´, where ´targets´ contains a list of the names of genes in the
        gene program for the reconstruction of the gene expression of the node
        itself (receiving node) and ´sources´ contains a list of the names of
        genes in the gene program for the reconstruction of the gene expression
        of the node's neighbors (transmitting nodes).
    gp_filter_mode:
        If `None` (default), do not filter any gene programs. If `subset`, 
        remove gene programs that are subsets of other gene programs from the 
        gene program dictionary. If `superset`, remove gene programs that are 
        supersets of other gene programs instead.
    combine_overlap_gps:
        If `True`, combine gene programs that overlap according to the defined
        thresholds.
    overlap_thresh_source_genes:
        If `combine_overlap_gps` is `True`, the minimum ratio of source 
        genes that need to overlap between two gene programs for them to be 
        combined.
    overlap_thresh_target_genes:
        If `combine_overlap_gps` is `True`, the minimum ratio of target 
        genes that need to overlap between two gene programs for them to be 
        combined.
    overlap_thresh_genes:
        If `combine_overlap_gps` is `True`, the minimum ratio of total genes
        (source genes & target genes) that need to overlap between two gene 
        programs for them to be combined.
    verbose:
        If `True`, print gene programs that are removed and combined.

    Returns
    ----------
    new_gp_dict:
        Modified gene program dictionary with gene programs filtered according 
        to ´gp_filter_mode´ and combined according to ´combine_overlap_gps´,
        ´overlap_thresh_source_genes´, ´overlap_thresh_target_genes´, and 
        ´overlap_thresh_genes´.
    """
    new_gp_dict = gp_dict.copy()

    # Remove gps that are subsets or supersets of other gps from the gp dict.
    # Note that ´issubset´ and ´issuperset´ are also ´True´ for identical gene
    # sets, and the loops below compare each pair of gene programs in both
    # directions. Gene programs are therefore only removed based on a gene
    # program that is itself still part of the gene program dictionary, so that
    # two gene programs with identical genes do not remove each other and one
    # of them is retained.
    if gp_filter_mode != None:
        for i, (gp_i, gp_genes_dict_i) in enumerate(gp_dict.items()):
            source_genes_i = set([gene.upper() for gene in 
                                  gp_genes_dict_i["sources"]])
            target_genes_i = set([gene.upper() for gene in 
                                  gp_genes_dict_i["targets"]])
            for j, (gp_j, gp_genes_dict_j) in enumerate(gp_dict.items()):
                if i != j:
                    source_genes_j = set([gene.upper() for gene in 
                                          gp_genes_dict_j["sources"]])
                    target_genes_j = set([gene.upper() for gene in
                                          gp_genes_dict_j["targets"]])
                    if gp_filter_mode == "subset":
                        if (gp_i in new_gp_dict and
                            source_genes_j.issubset(source_genes_i) and
                            target_genes_j.issubset(target_genes_i)):
                                new_gp_dict.pop(gp_j, None)
                                if verbose:
                                    print(f"Removing GP '{gp_j}' as it is a "
                                          f"subset of GP '{gp_i}'.")
                    elif gp_filter_mode == "superset":
                        if (gp_i in new_gp_dict and
                            source_genes_j.issuperset(source_genes_i) and
                            target_genes_j.issuperset(target_genes_i)):
                                new_gp_dict.pop(gp_j, None)
                                if verbose:
                                    print(f"Removing GP '{gp_j}' as it is a "
                                          f"superset of GP '{gp_i}'.")

    # Combine overlap gps in the gp dict (overlap ratios are calculated 
    # based on average gene numbers of the compared gene programs)
    if combine_overlap_gps:
        # First, get all overlap gps per gene program (this includes
        # duplicate overlaps and unresolved cross overlaps (i.e. GP A might 
        # overlap with GP B and GP B might overlap with GP C while GP A and GP C
        # do not overlap)
        all_overlap_gps = []
        for i, (gp_i, gp_genes_dict_i) in enumerate(new_gp_dict.items()):
            source_genes_i = set([gene.upper() for gene in 
                                  gp_genes_dict_i["sources"]])
            target_genes_i = set([gene.upper() for gene in 
                                  gp_genes_dict_i["targets"]])
            gp_overlap_gps = [gp_i]
            for j, (gp_j, gp_genes_dict_j) in enumerate(new_gp_dict.items()):
                if i != j:
                    source_genes_j = set([gene.upper() for gene in 
                                          gp_genes_dict_j["sources"]])
                    target_genes_j = set([gene.upper() for gene in
                                          gp_genes_dict_j["targets"]])
                    source_genes_overlap = list(source_genes_i & source_genes_j)
                    target_genes_overlap = list(target_genes_i & target_genes_j)
                    n_source_gene_overlap = len(source_genes_overlap)
                    n_target_gene_overlap = len(target_genes_overlap)
                    n_gene_overlap = (n_source_gene_overlap + 
                                      n_target_gene_overlap)
                    n_avg_source_genes = (len(source_genes_i) + 
                                          len(source_genes_j)) / 2
                    n_avg_target_genes = (len(target_genes_i) + 
                                          len(target_genes_j)) / 2
                    n_avg_genes = n_avg_source_genes + n_avg_target_genes
                    if n_avg_source_genes > 0:
                        ratio_shared_source_genes = (n_source_gene_overlap / 
                                                     n_avg_source_genes)
                    else: 
                        ratio_shared_source_genes = 1
                    if n_avg_target_genes > 0:
                        ratio_shared_target_genes = (n_target_gene_overlap /
                                                     n_avg_target_genes)
                    else:
                        ratio_shared_target_genes = 1
                    ratio_shared_genes = n_gene_overlap / n_avg_genes
                    if ((ratio_shared_source_genes >= 
                         overlap_thresh_source_genes) &
                        (ratio_shared_target_genes >= 
                         overlap_thresh_target_genes) &
                        (ratio_shared_genes >= overlap_thresh_genes)):
                            gp_overlap_gps.append(gp_j)
            if len(gp_overlap_gps) > 1:
                all_overlap_gps.append(set(gp_overlap_gps))

        # Second, clean up duplicate overlaps 
        all_unique_overlap_gps = []
        _ = [all_unique_overlap_gps.append(overlap_gp) for overlap_gp in 
             all_overlap_gps if overlap_gp not in all_unique_overlap_gps]

        # Third, split overlaps into no cross and cross overlaps
        no_cross_overlap_gps = []
        cross_overlap_gps = []
        for i, overlap_gp_i in enumerate(all_unique_overlap_gps):
            if all(overlap_gp_j.isdisjoint(overlap_gp_i) for 
            j, overlap_gp_j in enumerate(all_unique_overlap_gps) 
            if i != j):
                no_cross_overlap_gps.append(overlap_gp_i)
            else:
                cross_overlap_gps.append(overlap_gp_i)

        # Fourth, resolve cross overlaps by sequentally combining them (until
        # convergence)
        sequential_overlap_gps = list(cross_overlap_gps)
        while True:
            new_sequential_overlap_gps = []
            for i, overlap_gp_i in enumerate(sequential_overlap_gps):
                paired_overlap_gps = [overlap_gp_i.union(overlap_gp_j) for 
                                      j, overlap_gp_j in 
                                      enumerate(sequential_overlap_gps) 
                                      if (i != j) & 
                                      (overlap_gp_i.intersection(overlap_gp_j) 
                                       != set())]
                paired_overlap_gps_union = set().union(*paired_overlap_gps)
                if (paired_overlap_gps_union != set() and
                paired_overlap_gps_union not in new_sequential_overlap_gps):
                    new_sequential_overlap_gps.append(paired_overlap_gps_union)
            if (sorted([list(gp) for gp in new_sequential_overlap_gps]) == 
            sorted([list(gp) for gp in sequential_overlap_gps])):
                break
            else:
                sequential_overlap_gps = list(new_sequential_overlap_gps)

        # Fifth, add overlap gps to gp dict and remove component gps
        final_overlap_gps = [list(overlap_gp) for overlap_gp in 
                             no_cross_overlap_gps]
        _ = [final_overlap_gps.append(list(overlap_gp)) for overlap_gp in 
             sequential_overlap_gps if list(overlap_gp) not in 
             final_overlap_gps]

        for overlap_gp in final_overlap_gps:
            new_gp_name = "_".join([gp[:-3] for gp in overlap_gp]) + "_GP"
            new_gp_sources = []
            new_gp_targets = []
            new_gp_sources_categories = []
            new_gp_targets_categories = []
            for gp in overlap_gp:
                for new_gp_source, new_gp_source_category in zip(
                    gp_dict[gp]["sources"], gp_dict[gp]["sources_categories"]):
                    if new_gp_source not in new_gp_sources:
                        new_gp_sources.append(new_gp_source)
                        new_gp_sources_categories.append(new_gp_source_category)
                for new_gp_target, new_gp_target_category in zip(
                    gp_dict[gp]["targets"], gp_dict[gp]["targets_categories"]):
                    if new_gp_target not in new_gp_targets:
                        new_gp_targets.append(new_gp_target)
                        new_gp_targets_categories.append(new_gp_target_category)
                new_gp_dict.pop(gp, None)
                if verbose:
                    print(f"Removing GP '{gp}' as it is a component of the "
                          f"combined GP '{new_gp_name}'.")
            new_gp_dict[new_gp_name] = {"sources": new_gp_sources}
            new_gp_dict[new_gp_name]["targets"] = new_gp_targets
            new_gp_dict[new_gp_name]["sources_categories"] = (
                new_gp_sources_categories)
            new_gp_dict[new_gp_name]["targets_categories"] = (
                new_gp_targets_categories)
    return new_gp_dict


def filter_and_combine_gp_dict_gps_v2(
        gp_dicts: list,
        overlap_thresh_target_genes: float=1.,
        verbose: bool=False) -> dict:
    """
    Combine gene program dictionaries and filter them based on gene overlaps.

    Parameters
    ----------
    gp_dicts:
        List of gene program dictionaries with keys being gene program names and
        values being dictionaries with keys ´sources´, ´targets´,
        ´sources_categories´, and ´targets_categories´, where ´targets´ contains
        a list of the names of genes in the gene program for the reconstruction
        of the gene expression of the node itself (receiving node) and ´sources´
        contains a list of the names of genes in the gene program for the
        reconstruction of the gene expression of the node's neighbors
        (transmitting nodes).
    overlap_thresh_target_genes:
        The minimum ratio of target genes that need to overlap between a GP
        without source genes and another GP for the GP to be dropped.
        Gene programs with different source genes are never combined or dropped.
    verbose:
        If `True`, print gene programs that are dropped and combined.

    Returns
    ----------
    new_gp_dict:
        Combined gene program dictionary with filtered gene programs.
    """
    # Combine gene program dictionaries
    combined_gp_dict = {}
    for i, gp_dict in enumerate(gp_dicts):
        combined_gp_dict.update(gp_dict)

    new_gp_dict = combined_gp_dict.copy()

    # Combine gene programs with overlapping genes
    all_combined = False
    while not all_combined:
        all_combined = True
        combined_gp_dict = new_gp_dict.copy()
        for i, (gp_i, gp_genes_dict_i) in enumerate(combined_gp_dict.items()):
            source_genes_i = [
                gene for gene in gp_genes_dict_i["sources"]]
            target_genes_i = [
                gene for gene in gp_genes_dict_i["targets"]]
            target_genes_categories_i = [
                target_gene_category for target_gene_category in
                gp_genes_dict_i["targets_categories"]]
            for j, (gp_j, gp_genes_dict_j) in enumerate(
                combined_gp_dict.items()):
                if j != i:
                    source_genes_j = [
                        gene for gene in gp_genes_dict_j["sources"]]
                    target_genes_j = [
                        gene for gene in gp_genes_dict_j["targets"]]
                    target_genes_categories_j = [
                        target_gene_category for target_gene_category in
                        gp_genes_dict_j["targets_categories"]]

                    if ((source_genes_i == source_genes_j) and
                        len(source_genes_i) > 0):
                        # if source genes are exactly the same, combine gene
                        # programs
                        all_combined = False
                        if verbose:
                            print(f"Combining {gp_i} and {gp_j}.")
                        source_genes = source_genes_i
                        target_genes = target_genes_i
                        target_genes_categories = target_genes_categories_i
                        for target_gene, target_gene_category in zip(
                            target_genes_j, target_genes_categories_j):
                            if target_gene not in target_genes:
                                target_genes.extend([target_gene])
                                target_genes_categories.extend(
                                    [target_gene_category])
                        new_gp_dict.pop(gp_i, None)
                        new_gp_dict.pop(gp_j, None)
                        if (gp_j.split("_")[0] + 
                            "_combined_GP") not in new_gp_dict.keys():
                            new_gp_name = gp_i.split("_")[0] + "_combined_GP"
                            new_gp_dict[new_gp_name] = {"sources": source_genes}
                            new_gp_dict[new_gp_name]["targets"] = target_genes
                            new_gp_dict[new_gp_name][
                                "sources_categories"] = gp_genes_dict_i[
                                    "sources_categories"]
                            new_gp_dict[new_gp_name][
                                "targets_categories"] = target_genes_categories
                            
                    elif len(source_genes_i) == 0:
                        target_genes_overlap = list(
                            set(target_genes_i) & set(target_genes_j))
                        n_target_gene_overlap = len(target_genes_overlap)
                        n_target_genes = len(target_genes_i)
                        ratio_shared_target_genes = (n_target_gene_overlap /
                                                     n_target_genes)
                        if ratio_shared_target_genes >= overlap_thresh_target_genes:
                            # if source genes not existent and target genes
                            # overlap more than specified, drop gene program
                            if gp_j in new_gp_dict.keys():
                                if verbose:
                                    print(f"Dropping {gp_i}.")
                                new_gp_dict.pop(gp_i, None)
                    else:
                        # otherwise do not combine or drop gene programs
                        pass

    return new_gp_dict


def get_unique_genes_from_gp_dict(
        gp_dict: dict,
        retrieved_gene_entities: list=["sources", "targets"],
        retrieved_gene_categories: Optional[list]=None) -> list:
    """
    Return all unique genes of a gene program dictionary.

    Parameters
    ----------
    gp_dict:
        The gene program dictionary from which to retrieve the unique genes.
    retrieved_gene_entities:
        A list that contains all gene entities ("sources", "targets")
        for which unique genes of the gene program dictionary should be
        retrieved.
    retrieved_gene_categories:
        A list that contains all gene categories for which unique genes of the
        gene program dictionary should be retrieved. If `None`, all gene
        categories are included.

    Returns
    ----------
    unique_genes:
        A list of unique genes used in the gene program dictionary.
    """
    gene_list = []

    for _, gp in gp_dict.items():
        for gene_entity in retrieved_gene_entities:
            genes = gp[gene_entity]
            gene_categories = gp[f"{gene_entity}_categories"]
            if retrieved_gene_categories is not None:
                genes = [gene for gene, gene_category in zip(genes, gene_categories) if
                         gene_category in retrieved_gene_categories]
            gene_list.extend(genes)
    unique_genes = list(set(gene_list))
    unique_genes.sort()
    return unique_genes