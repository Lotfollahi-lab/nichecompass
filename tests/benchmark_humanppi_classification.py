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
import csv
import os
from collections import Counter
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


GOLD_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "data", "humanppi_orientation_gold.tsv")


def load_orientation_gold() -> dict:
    """
    Load the curated sender to receiver direction of the interactions whose
    direction is unambiguous from established cell biology.

    Returns
    ----------
    gold:
        Mapping from unordered gene pair to a dict with the ligand and the
        confidence of the curation.
    """
    gold = {}
    with open(GOLD_FILE_PATH) as gold_file:
        for row in csv.DictReader(gold_file, delimiter="\t"):
            if not row.get("ligand"):
                continue
            pair = frozenset((row["gene_1"].upper(), row["gene_2"].upper()))
            gold[pair] = {"ligand": row["ligand"].upper(),
                          "confidence": row.get("confidence", ""),
                          "class": row.get("class", "")}
    return gold


def orient_interactome(**kwargs) -> dict:
    """
    Return, for every intercellular gene program, which gene was placed in the
    source component and which rule decided it.
    """
    gp_dict = extract_gp_dict_from_humanppi_interactions(
        species="human",
        program_type="intercellular",
        plot_gp_gene_count_distributions=False,
        **kwargs)
    orientations = {}
    for gp in gp_dict.values():
        if len(gp["sources"]) != 1 or len(gp["targets"]) != 1:
            continue
        source = gp["sources"][0].upper()
        target = gp["targets"][0].upper()
        orientations[frozenset((source, target))] = {
            "source": source, "rule": gp.get("orientation_rule",
                                             "table_order")}
    return orientations


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
    parser.add_argument("--min_orientation_accuracy", type=float, default=0.9,
                        help="Fraction of the curated interactions that the "
                             "orientation rules decide correctly, below which "
                             "the benchmark fails. Only the interactions that "
                             "a rule decided are counted, since the remainder "
                             "keep the arbitrary order of the released "
                             "columns.")
    parser.add_argument("--min_orientation_coverage", type=float, default=0.4,
                        help="Fraction of the intercellular interactions that "
                             "have to be oriented from evidence rather than "
                             "left on the released column order.")
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

    failures = []
    if recall < args.min_recall:
        failures.append(f"Recall {100 * recall:.1f}% is below the required "
                        f"{100 * args.min_recall:.1f}%.")

    print("\nChecking the orientation of the intercellular interactions...")
    gold = load_orientation_gold()
    orientations = orient_interactome()
    coverage_pairs = [pair for pair in orientations
                      if orientations[pair]["rule"] != "table_order"]
    coverage = (len(coverage_pairs) / len(orientations) if orientations
                else 0.)
    print(f"Intercellular interactions: {len(orientations)}, of which "
          f"{len(coverage_pairs)} oriented from evidence "
          f"({100 * coverage:.1f}%).")

    decided = [pair for pair in gold
               if pair in orientations
               and orientations[pair]["rule"] != "table_order"]
    correct = [pair for pair in decided
               if orientations[pair]["source"] == gold[pair]["ligand"]]
    accuracy = len(correct) / len(decided) if decided else 0.
    print(f"Curated directions the rules decided: {len(decided)}, correct: "
          f"{len(correct)} ({100 * accuracy:.1f}%).")

    # Reported separately, because the ephrins dominate the curated set and are
    # resolved by a single family rule
    non_ephrin = [pair for pair in decided
                  if gold[pair]["class"] != "ephrin_eph"]
    non_ephrin_correct = [pair for pair in non_ephrin
                          if orientations[pair]["source"]
                          == gold[pair]["ligand"]]
    if non_ephrin:
        print(f"Excluding the ephrins: {len(non_ephrin_correct)} of "
              f"{len(non_ephrin)} "
              f"({100 * len(non_ephrin_correct) / len(non_ephrin):.1f}%).")

    wrong = [pair for pair in decided if pair not in correct]
    if wrong:
        print(f"\nOriented against the curated direction ({len(wrong)}):")
        for pair in sorted(wrong, key=sorted):
            print(f"  {' / '.join(sorted(pair))}: source "
                  f"{orientations[pair]['source']}, curated ligand "
                  f"{gold[pair]['ligand']} "
                  f"[{orientations[pair]['rule']}]")

    rule_counts = Counter(orientation["rule"]
                          for orientation in orientations.values())
    print("\nInteractions per orientation rule:")
    for rule, count in rule_counts.most_common():
        print(f"  {rule}: {count}")

    if accuracy < args.min_orientation_accuracy:
        failures.append(
            f"Orientation accuracy {100 * accuracy:.1f}% is below the "
            f"required {100 * args.min_orientation_accuracy:.1f}%.")
    if coverage < args.min_orientation_coverage:
        failures.append(
            f"Orientation coverage {100 * coverage:.1f}% is below the "
            f"required {100 * args.min_orientation_coverage:.1f}%.")

    if failures:
        raise SystemExit(" ".join(failures))
    print("\nBenchmark passed.")


if __name__ == "__main__":
    main()
