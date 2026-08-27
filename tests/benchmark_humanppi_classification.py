"""
Benchmark of the human PPI interaction classification against curated
ligand-receptor pairs.

A curated ligand-receptor pair is by definition an interaction between two
cells, so every such pair that is also present in the predicted human
interactome should be classified as paracrine or juxtacrine. The fraction that
is recovered is the recall reported here.

This is deliberately not a pytest test: it downloads the interactome, the
Complex Portal table and a curated ligand-receptor reference, and it queries
UniProt. Run it manually after changing the classification logic:

    python tests/benchmark_humanppi_classification.py

A drop in recall means that a change to the keyword sets or to the
classification rules has started discarding genuine intercellular biology.
"""

import argparse
import io
import ssl
import urllib.request

import pandas as pd

from nichecompass.utils import extract_gp_dict_from_humanppi_interactions

OMNIPATH_URL = ("https://omnipathdb.org/interactions?datasets=ligrecextra"
                "&organisms=9606&genesymbols=yes&fields=sources")


def load_curated_ligand_receptor_pairs() -> set:
    """Retrieve curated ligand-receptor pairs from OmniPath."""
    context = ssl._create_unverified_context()
    with urllib.request.urlopen(OMNIPATH_URL, context=context) as response:
        table = response.read().decode("utf-8", "replace")
    interaction_df = pd.read_csv(io.StringIO(table), sep="\t")
    source_col = [col for col in interaction_df.columns
                  if "source_genesymbol" in col][0]
    target_col = [col for col in interaction_df.columns
                  if "target_genesymbol" in col][0]
    pairs = set()
    for source, target in zip(interaction_df[source_col],
                              interaction_df[target_col]):
        # Protein complexes are written as 'COMPLEX:A_B' and are expanded
        for source_gene in str(source).replace("COMPLEX:", "").split("_"):
            for target_gene in str(target).replace("COMPLEX:", "").split("_"):
                if source_gene and target_gene and source_gene != target_gene:
                    pairs.add(frozenset((source_gene.upper(),
                                         target_gene.upper())))
    return pairs


def classify_interactome(**kwargs) -> dict:
    """Classify the interactome and return a gene pair to class mapping."""
    gp_dict = extract_gp_dict_from_humanppi_interactions(
        species="human",
        program_type="both",
        plot_gp_gene_count_distributions=False,
        **kwargs)
    interaction_classes = {}
    for gp_name, gp in gp_dict.items():
        genes = [gene.upper() for gene in gp["sources"] + gp["targets"]]
        if len(genes) != 2:
            continue
        prefix = gp_name.rsplit("_ppi_GP", 1)[0]
        for interaction_class in ["paracrine", "juxtacrine", "cis_complex",
                                  "extracellular_assembly", "intracellular"]:
            if prefix.endswith("_" + interaction_class):
                interaction_classes[frozenset(genes)] = interaction_class
                break
    return interaction_classes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min_recall", type=float, default=0.9,
                        help="Recall below which the benchmark fails.")
    args = parser.parse_args()

    print("Retrieving curated ligand-receptor pairs...")
    pairs = load_curated_ligand_receptor_pairs()
    print(f"Curated ligand-receptor pairs: {len(pairs)}.")

    print("\nClassifying the predicted human interactome...")
    interaction_classes = classify_interactome()

    testable = [pair for pair in pairs if pair in interaction_classes]
    intercellular = [pair for pair in testable
                     if interaction_classes[pair] in ("paracrine",
                                                      "juxtacrine")]
    recall = len(intercellular) / len(testable) if testable else 0.
    print(f"\nCurated pairs present in the interactome: {len(testable)}.")
    print(f"Classified as paracrine or juxtacrine: {len(intercellular)}.")
    print(f"Recall: {100 * recall:.1f}%.")

    missed = [pair for pair in testable if pair not in intercellular]
    if missed:
        print(f"\nNot recovered ({len(missed)}):")
        for pair in sorted(missed, key=sorted):
            print(f"  {' / '.join(sorted(pair))}: "
                  f"{interaction_classes[pair]}")

    if recall < args.min_recall:
        raise SystemExit(
            f"Recall {100 * recall:.1f}% is below the required "
            f"{100 * args.min_recall:.1f}%.")
    print("\nBenchmark passed.")


if __name__ == "__main__":
    main()
