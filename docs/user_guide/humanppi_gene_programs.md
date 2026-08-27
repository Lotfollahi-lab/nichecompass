# Human PPI gene programs

This page documents, step by step, how `extract_gp_dict_from_humanppi_interactions` turns the predicted
human interactome into prior gene programs for NicheCompass. It is written for a computational biologist or
bioinformatician who wants to understand exactly what the resulting gene program mask contains, what
evidence each decision rests on, and where the limits are.

## 1. What this resource is and why it needs a classification step

NicheCompass makes its latent space interpretable by masking the decoder with **prior gene programs**. Each
gene program has two components, and the distinction is the reason a raw interactome cannot be used directly:

- the **source component** holds genes whose expression is reconstructed from the **aggregated expression of
  a cell's spatial neighbours**, i.e. the transmitting cells;
- the **target component** holds genes whose expression is reconstructed from the **cell's own expression**,
  i.e. the receiving cell.

A gene program therefore encodes a statement of the form *"these genes in my neighbours, together with those
genes in me"*. A program with an empty source component models a purely within-cell process, exactly as the
CollecTRI transcription factor programs do.

The source resource is the predicted human interactome of
[Zhang, Humphreys et al., Science 390, eadt1630 (2025)](https://www.science.org/doi/10.1126/science.adt1630).
The authors screened roughly 190 million human protein pairs with RoseTTAFold2-PPI and AlphaFold2 and
released 17,849 predicted interactions at an expected precision of 90% and 29,257 at 80%, of which about
3,600 were previously unreported. Two properties of that resource drive everything below:

1. It is a **physical interaction** network, not a signalling network. It is dominated by intracellular
   complexes: the proteasome, the mitochondrial respiratory chain, spliceosome subunits, chaperone pairs.
   Feeding it in unfiltered would fill the latent space with within-cell complexes placed in a
   neighbour-to-self signalling role.
2. Interactions are **undirected**. Nothing in the table says which partner is the ligand and which the
   receptor, so the assignment to the source and target component has to be derived.

The classification below exists to solve both problems: it decides which interactions can plausibly act
*between* cells, and for those it decides which partner belongs in which component.

## 2. Retrieval, caching and provenance

Three external resources are used. Each is downloaded on first use, cached, and accompanied by a
`<file>.provenance.json` recording the source URL, the retrieval time in UTC and whatever version the server
reports. Reusing a cache prints its provenance; a cache without provenance is reported as such. This matters
because all three are updated independently of NicheCompass, so a gene program mask is only reproducible if
the versions that produced it are known.

| Resource | Source | Cached as | Version recorded |
| :-- | :-- | :-- | :-- |
| Interaction predictions | `conglab.swmed.edu/humanPPI` (archived on [Dryad](https://doi.org/10.5061/dryad.15dv41p84)) | `humanppi_network_<precision>.csv` | `Last-Modified`, size |
| Protein annotation | UniProt REST | `humanppi_protein_topology.tsv` | UniProt release and release date |
| Protein complexes | EBI Complex Portal, human | `complex_portal_human.tsv` | `Last-Modified`, size |

The download server for the predictions presents an incomplete TLS certificate chain, so the first attempt is
retried with certificate verification disabled and a warning. This is expected, not a failure.

### What the prediction table contains

28 columns per interaction. The classification uses only a few of them, so it is worth knowing what is
there. Taking PD-1 / PD-L1 as an example:

| Column | Value | Used? |
| :-- | :-- | :-- |
| `Protein1`, `Protein2` | `Q15116`, `Q9NZQ7` (UniProt accessions) | yes, to join UniProt and the Complex Portal |
| `Name1`, `Name2` | `PDCD1`, `CD274` (gene symbols) | yes, these become the gene program genes |
| `RFprob`, `AFprob` | 0.998, 0.998 | optional thresholds |
| `AFprob5` / `CFprob`, `AFMprob` | 0.9995, 0.9976 | no (these columns differ between the two precision levels) |
| `Source` | `S,P` (how the pair entered the screen) | no |
| `PDBtemp`, `Exact_/Ortho_/Homo_templates` | `exact`, `5ius,4zqk`, ... | no |
| `confDBs`, `allDBs`, `STRING` | `BIOGRID,STRING,UNIPROT`, 999 | no |
| `Known1/2`, `Count1/2` | 4.1, 17.0, 17, 13 | no |
| **`Locality1`, `Locality2`** | `Cell membrane,Membrane` / `Cell membrane,Endosome,Membrane,Nucleus,Secreted` | **yes, the primary evidence** |
| `Process1`, `Process2` | `Adaptive immunity,Apoptosis,Immunity` | yes, as a last-resort fallback |
| `Disease1/2`, `Function1/2` | ..., `Programmed cell death protein 1` | no |

`Locality` holds **UniProt cellular-component keywords**, and it is a snapshot taken when the screen was
run, which turns out to matter (see step 4.3).

## 3. Preprocessing: which interactions are removed before classification

Four filters run on the table, in this order. Each prints how many interactions it removed. Numbers below
are for `precision="90"`, `species="human"`.

### 3.1 Missing gene names (always applied)

The table uses the **literal string `none`** as its placeholder for missing values, in every column
including the gene name columns, so `dropna` does not catch it. 38 interactions involve a protein whose
UniProt entry has no gene name at all: putative and uncharacterised proteins such as
`Putative speedy protein-like protein 3`, and immunoglobulin variable segments. These are dropped for two
reasons: a protein without a gene symbol can never be matched to a measured gene, and because *all* unnamed
proteins share the single placeholder, keeping them would conflate 18 distinct proteins into one spurious
gene called `none` and would make distinct interactions look like duplicates of each other.

### 3.2 Immunoglobulin and T cell receptor gene segments (`filter_ig_tcr_segments`, default `True`)

Removes **358 interactions**. Antibodies and T cell receptors are encoded as separate V, D and J gene
segments that are somatically recombined into a single chain, so an interaction between two of them, or
between a segment and the constant region of the same chain, is **intramolecular** rather than an interaction
between two proteins. `IGLV10-54`-`IGLC2` is the variable and the constant region of one lambda light chain.

The structural predictor produces many such pairs because every variable domain shares the immunoglobulin
fold: `IGHV4-30-2`-`TRDV1`, an antibody heavy V region with a T cell receptor delta V region, scores
`RFprob` 0.998. The pattern matches V, D and J segments only
(`^(IG[HKL][VJ]\d|IGHD\d|TR[ABDG][VJ]\d|TR[BD]D\d)`); constant region genes such as `IGHG1`, `IGHM`,
`IGLC2`, `TRAC` and `TRBC2` encode complete proteins and are deliberately kept. Note that `IGHD` without a
digit is the IgD heavy constant region, while `IGHD3-3` is a D segment.

### 3.3 Paralogue combinations that do not form (`filter_paralog_cross_pairs`, default `True`)

Removes **80 interactions**. Close paralogues have interchangeable interfaces, so the predictor generalises
across them and produces heterodimers that do not exist. Two families are constrained explicitly:

- **Integrins**: only the 24 alpha-beta heterodimers that actually form are kept, so `ITGB4`-`ITGAL`,
  `ITGB6`-`ITGAL` and `ITGA5`-`ITGB5` are dropped while `ITGAL`-`ITGB2` and `ITGAV`-`ITGB3` are kept.
- **MHC class II**: only matching isotypes pair, so `HLA-DRA`-`HLA-DRB1` is kept while `HLA-DRA`-`HLA-DQB1`
  and `HLA-DQB2`-`HLA-DPA1` are dropped.

### 3.4 Prediction confidence (`min_rf_prob`, `min_af_prob`, default `None`)

Optional. `RFprob` is the RoseTTAFold2-PPI probability from the fast coevolution-driven screen of all
candidate pairs; `AFprob` is the AlphaFold2 probability from the slower structural rescoring of the pairs
that passed. **A low score in one column does not mean the interaction is unreliable**: pairs entered the
final set through several routes and the evidence-guided routes used relaxed cutoffs, so the released tables
deliberately contain interactions with a very low score from one predictor but strong support overall. In the
precision-90 table `RFprob` has a median of 0.678 with only 39.4% of entries at or above 0.9, while `AFprob`
has a median of 0.943. Thresholds are therefore aggressive and the default is to trust the authors'
own precision calibration.

## 4. Gathering evidence about each protein

### 4.1 UniProt annotation (`use_topology`, default `True`)

One batched UniProt REST request per 100 accessions retrieves seven fields, all cached together. For the
12,258 accessions across both precision levels:

| Evidence | Field | Coverage |
| :-- | :-- | --: |
| Membrane anchor (transmembrane segment or GPI anchor) | `ft_transmem`, `ft_lipid` | 3,157 |
| Topological domains annotated | `ft_topo_dom` | 2,431 |
| ... of which an **extracellular** domain | `ft_topo_dom` | 1,695 |
| Signal peptide | `ft_signal` | 2,248 |
| Current cellular-component keywords | `keyword` | 11,168 |
| Subcellular location comment | `cc_subcellular_location` | 11,104 |
| Gene Ontology cellular components | `go_c` | 11,954 |

Why topology matters so much: **cellular-component keywords describe whole proteins and say nothing about
which side of a membrane a protein faces.** Peripheral proteins docked onto the cytoplasmic leaflet of the
plasma membrane carry `Cell membrane` too. Without topology, the neuronal SNARE complex
(`SNAP25`-`VAMP2`), the protein kinase A holoenzyme (`PRKACA`-`PRKAR2B`), adducin, the calpains and the
cytoplasmic adherens plaque (`CTNNA1`-`CTNNB1`) are all classified as contact-dependent signalling.
Membrane anchoring is a sequence-level property and resolves them.

### 4.2 Protein complexes (`detect_cis_complexes`, default `True`)

The human Complex Portal table yields **18,284 unordered accession pairs** that are subunits of a common
complex. Two details:

- **Complexes with fewer than three subunits are skipped.** The Complex Portal also registers
  ligand-receptor pairs as complexes, and those are predominantly binary (tumour necrosis factor with its
  receptors, lymphotoxin beta with its receptor, colony stimulating factor 1 with its receptor). Genuine
  binary *cis* assemblies are covered by the curated families instead.
- **24 curated gene family patterns** complement the Portal, because it covers only about a fifth of the
  proteins in this network and is missing several prominent surface complexes outright, among them every MHC
  class II alpha-beta pair, the CD79 heterodimer of the B cell receptor and the high-affinity IgE receptor.
  The families cover MHC class I with beta-2 microglobulin, MHC class II, the T cell receptor chains, CD8,
  CD79, FcεRI, CD94/NKG2, integrins, collagens, laminins, sarcoglycans, the BBSome, the AP-2 adaptor,
  complement C1q, fibrinogen, GABA-A, ionotropic glutamate, glycine, nicotinic acetylcholine and serotonin-3
  receptors, the epithelial sodium channel, heteromeric amino acid transporters, CatSper and the shared
  gamma-chain cytokine receptors.

## 5. Step one of the classification: the location class of each protein

Every protein is assigned exactly one of five **location classes**.

| Class | Meaning | Keyword set size | Examples |
| :-- | :-- | --: | :-- |
| `cell_surface` | membrane anchored with a necessarily extracellular face | 16 | `Cell membrane`, `Cell surface`, `Apical cell membrane`, `Sarcolemma`, `Membrane raft`, `Gap junction`, `MHC I`, `MHC II`, `T cell receptor`, `Target cell membrane` |
| `secreted` | released extracellularly, not membrane anchored | 9 | `Secreted`, `Extracellular matrix`, `Basement membrane`, `Membrane attack complex`, `HDL`/`LDL`/`VLDL`/`Chylomicron`, `Surface film` |
| `intracellular` | an intracellular location and no surface or secreted keyword | 67 | `Nucleus`, `Cytoplasm`, `Mitochondrion`, `Endoplasmic reticulum`, `Golgi apparatus`, `Lysosome`, `Cytoskeleton`, `Proteasome`, `Coated pit`, `Synaptosome`, `Flagellum`, `Exosome` |
| `ambiguous` | compatible with an extracellular face without establishing one | 23 | `Membrane`, `Cell junction`, `Tight junction`, `Desmosome`, `Focal adhesion`, `Cell projection`, `Cilium`, `Synapse`, `Amyloid`, `Immunoglobulin`, `Virion` |
| `unknown` | no usable evidence from any source | — | — |

### 5.1 Why several plausible-sounding keywords are *not* treated as surface

These assignments were reviewed independently and are the least obvious part of the design:

- **`Membrane`** is the generic parent of the whole membrane branch and also covers the endoplasmic
  reticulum, mitochondrial, Golgi, nuclear and endolysosomal membranes. Among proteins whose only
  membrane evidence is this keyword, 23% are also annotated to the endoplasmic reticulum, 21% to
  mitochondria and 16% to the Golgi. UniProt applies the specific child `Cell membrane` when plasma-membrane
  localisation is known.
- **`Cell junction`, `Tight junction`, `Synapse`, `Cell projection`, `Cilium`** denote compartments with a
  membrane-embedded core **and** a large cytoplasmic component. They are carried by cytosolic plaque and
  scaffold proteins (ZO-1, catenins, vinculin, talin, paxillin, PSD-95, synapsins) and by axonemal and
  intraflagellar transport machinery. The genuinely surface-exposed proteins in those compartments virtually
  always also carry `Cell membrane`, so excluding these keywords costs little coverage and removes a large
  false-positive source; ZO-1 in particular is a high-degree hub.
- **`Exosome`** in UniProt denotes the **exosome complex**, the nuclear and cytoplasmic 3'-5'
  exoribonuclease machine of EXOSC subunits, **not** extracellular vesicles. It is a name trap.
- **`Amyloid`** describes an aggregation propensity rather than a location, and `Immunoglobulin` is also
  used for the immunoglobulin *domain*, which occurs in intracellular proteins such as titin and obscurin.

Conversely, `MHC I`, `MHC II`, `T cell receptor` and `Target cell membrane` **are** surface keywords; leaving
them out would have silently excluded antigen presentation from an immune analysis.

### 5.2 Precedence rules

Evidence is graded, and three precedence rules resolve conflicts. Each exists because of a concrete failure
mode:

1. **An extracellular face beats an intracellular location.** Secreted and surface proteins are routinely
   also annotated with the compartments they traverse; interleukin 15 is `Cytoplasm,Nucleus,Secreted`.
   Requiring the absence of intracellular keywords would discard genuine ligands.
2. **Membrane anchoring beats secretion.** 345 proteins carry both, because many surface receptors have a
   shed soluble isoform. PD-L1 is `Cell membrane,Endosome,Membrane,Nucleus,Secreted`; calling it secreted
   would turn the contact-dependent PD-1 / PD-L1 axis into a paracrine one.
3. **Antibody chains are an exception to rule 2.** Every immunoglobulin gene carries
   `Cell membrane,Immunoglobulin,Membrane,Secreted`, for the B cell receptor form and the antibody form
   respectively, and it is the secreted form that dominates their interactions with Fc receptors. Without
   this exception the canonical `IGHG1`-`FCGR2B` pair is classified as contact-dependent.

### 5.3 How topology modifies the keyword verdict

Topology is consulted in **both** directions:

- **Rejection.** A `cell_surface` keyword is not trusted if the protein has no membrane anchor, or if its
  annotated topological domains contain no extracellular one. This is what demotes SNAP25, PKA, adducin,
  the calpains and the catenins.
- **Promotion.** An annotated **extracellular topological domain** is decisive positive evidence and
  outweighs the keywords, because UniProt uses `Extracellular` only for the outside of the cell and
  `Lumenal` for organelle interiors, so it cannot admit organelle membranes. Many cytokine receptors need
  this: `IL15RA` and `IL10RB` are annotated with the generic `Membrane` keyword plus `Secreted` for a shed
  form, and with the compartments they traverse, but never with `Cell membrane`. Without promotion they are
  classified as secreted *ligands*, and `IL15`-`IL15RA` becomes a ligand-ligand pair.
- A cell-surface keyword contradicted by the absence of an anchor yields `intracellular`, because such a
  protein is peripheral, sitting on one side of the membrane rather than spanning it, and in practice on the
  cytoplasmic side.

`ambiguous_locality` (default `extracellular`) decides the remaining case, where a keyword is compatible with
an extracellular face and topology neither confirms nor contradicts it. **With `use_topology=True` and a
complete annotation cache this parameter is inert and the `ambiguous` class is never produced**, because
topology resolves every case the keywords leave open: measured on the precision-90 predictions, no gene is
assigned the `ambiguous` category and both settings give 2,010 intercellular programs. It becomes the
deciding parameter only with `use_topology=False`, where it gives 2,527 against 1,776 intercellular
programs.

### 5.4 The fallback chain for proteins with no cellular-component keyword

Five fallbacks, in decreasing reliability. Weak sources are deliberately restricted to establishing an
**intracellular** location, because mislabelling a within-cell complex as intercellular is the expensive
error while missing a surface protein only costs coverage.

| Order | Source | Can establish | Rationale |
| --: | :-- | :-- | :-- |
| 1 | Current UniProt cellular-component keywords | any class | The `Locality` column is a **stale snapshot**, not genuinely absent; refreshing it accounts for most of what the next fallback used to recover |
| 2 | UniProt subcellular location comment | any class | Curated to the same standard as the keywords, occasionally present when no keyword was assigned |
| 3 | Gene Ontology cellular components | **intracellular only** | Broadest coverage, but its extracellular calls are 84% wrong |
| 4 | Signal peptide | `secreted`, or `cell_surface` if anchored | Sequence evidence that the protein enters the secretory pathway |
| 5 | Biological-process keywords | **intracellular only** | 42 keywords such as `Transcription`, `mRNA splicing`, `Keratinization` |

Two subtleties:

- **Gene Ontology is restricted to a veto in the extracellular direction.** Proteomics of vesicle and granule
  preparations attaches `extracellular exosome`, `extracellular vesicle` and `blood microparticle` to large
  numbers of cytosolic proteins, and bare `plasma membrane` is attached to many proteins that only dock onto
  the cytoplasmic leaflet. Those terms therefore establish nothing in either direction, while genuine
  extracellular terms (`extracellular space`, `cell surface`, `external side of plasma membrane`) only
  **prevent** an intracellular conclusion rather than asserting a surface.
- **Gene Ontology is consulted before the signal peptide** so that proteins of the secretory pathway that are
  resident in an organelle rather than released, such as endoplasmic reticulum chaperones, are recognised as
  intracellular first.

Interactions still unresolved after all five are governed by `unresolved_locality` (default `exclude`):
524 at precision 90. They can never be classified as intercellular, so setting `intracellular` only ever
moves them into the target component.

## 6. Step two of the classification: the interaction class

| Class | Condition | Gene program shape |
| :-- | :-- | :-- |
| `juxtacrine` | both partners extracellular facing, neither purely secreted, **both able to reach across the cleft** | source → target |
| `paracrine` | both extracellular facing, at least one secreted | source → target, **soluble partner in the source** |
| `cis_complex` | both are subunits of a common complex | target only |
| `extracellular_assembly` | two **secreted** subunits of a common complex | target only |
| `intracellular` | at least one partner has no extracellular face | target only |
| `unknown` | at least one partner unresolved | excluded, or target only |

`program_type="intercellular"` keeps `paracrine` and `juxtacrine`; `"intracellular"` keeps the rest;
`"both"` keeps everything.

### 6.1 The cis-complex adjudication

Two proteins on the same cell's surface can form a complex rather than signal between cells, and
localisation cannot see the difference. Before this step, five of the six possible CD3 chain pairs, the CD8
and CD79 heterodimers, every integrin alpha-beta pair and every MHC class II alpha-beta pair were classified
as contact-dependent signalling.

Adjudication is at **pair** level, never complex level, because the Complex Portal registers genuine
ligand-receptor assemblies as complexes: interferon alpha with its receptor, erythropoietin with its
receptor, CXCL8 with CXCR1. **A shared complex in which exactly one partner is soluble is a ligand-receptor
assembly** and stays paracrine. If both partners are secreted it is an `extracellular_assembly` instead,
because secreted multimers such as collagen and laminin trimers, fibrinogen and complement C1q are assembled
by the cell that produces them.

At precision 90 this reclassifies 147 interactions as `cis_complex` and 57 as `extracellular_assembly`.

### 6.1.1 Assemblies within one gene family

Two members of one gene family are usually subunits of a single oligomer rather than partners on two cells,
and the Complex Portal registers almost none of them. A pair whose two symbols share a gene family root,
obtained by stripping the trailing member number, is therefore treated as an assembly within one cell: the
P2X receptor subunits (`P2RX2`-`P2RX3`) form a heterotrimeric channel, the Kv1 subunits (`KCNA6`-`KCNA1`) a
heterotetramer, `ABCG1`-`ABCG4` a heterodimeric transporter, and `EPHA1`-`EPHA2` cluster in cis. Where both
partners are secreted the result is an `extracellular_assembly` instead, which correctly captures the
ficolin oligomers (`FCN1`-`FCN2`), the C1q-domain multimers (`C1QL2`-`C1QL4`), the bone morphogenetic
protein heterodimers (`BMP4`-`BMP7`) and the apolipoproteins sharing a lipoprotein particle
(`APOC2`-`APOC3`).

A pair in which one symbol is the other plus a receptor or ligand suffix is also excluded, because
stripping the member number collapses a ligand and its own receptor onto one root: `CSF1` and `CSF1R` both
reduce to `CSF`, as do `CSF3`/`CSF3R`, `CSF2`/`CSF2RA` and `MST1`/`MST1R`. Without that exclusion a
membrane-anchored ligand paired with its own receptor would be silently demoted to `cis_complex`.

Seven families are **exempt**, because their members are built to engage a partner on the neighbouring
cell: the cadherins and protocadherins, the claudins with occludin and the junctional adhesion molecules,
the connexins, the nectins and nectin-like molecules, the desmosomal cadherins, the synaptic adhesion
molecules (neurexins, neuroligins, contactins, LRRTMs, DSCAM), and the siglecs and selectins. A root shorter
than three characters is also ignored as too generic.

This reclassifies 115 interactions at precision 90 **with no change to recall against curated
ligand-receptor pairs**, which stays at 97.0% — the exemptions are what make that possible. It also raises
orientation coverage from 54.2% to 58.0%, because interactions that have no direction to find are no longer
in the intercellular set.

### 6.2 The reach test (`min_extracellular_domain_length`, default `30`)

Localisation says which side of the membrane a protein faces, not how far it protrudes from it. Two
subunits of a channel or a receptor heterodimer both face outwards and neither is secreted, so on
localisation alone they are indistinguishable from a genuine interaction between two cells. A **positive
test for *trans*** is therefore applied to every candidate `juxtacrine` interaction.

The length of every annotated extracellular topological domain is parsed from `ft_topo_dom`, and the number
of membrane crossings from `ft_transmem`. A contact-dependent interaction is reclassified `cis_complex` when
either partner's **longest extracellular domain is shorter than `min_extracellular_domain_length`**, because
a protein that barely protrudes from its own membrane cannot bridge the intercellular cleft. A protein with
no annotated extracellular domain is demoted only if it crosses the membrane three or more times, since
polytopic proteins with no annotated ectodomain have short connecting loops by construction.

Two design decisions matter:

- **Only positive evidence demotes.** A protein with no topological annotation at all is left alone, so
  incomplete annotation does not silently shrink the intercellular set.
- **The test is not applied to `paracrine` interactions.** A soluble partner diffuses to its receptor and
  can engage a shallow binding pocket. Chemokine receptors are the clearest case: `CCR7` has a 35-residue
  extracellular N-terminus, and `CCL19`-`CCR7` would be at risk under any stricter threshold.

Connexins, claudins and occludin are exempt by name, because their docking with a partner on the
neighbouring cell genuinely happens through short extracellular loops.

This reclassifies **166 interactions** at precision 90 with no loss of recall, and catches exactly the
proteins localisation cannot: `CD247` has a 9-residue ectodomain, `FCER1G` 5, `TYROBP` 19, `KCNJ6` 24,
`CNGA1` 23, `HCN1` 26, and `ORAI2` has none across four membrane crossings.

### 6.3 Curated cis co-receptor families

Some complexes assemble on one cell yet have ectodomains hundreds of residues long, so no topological test
can separate them from a genuine *trans* interaction, and the Complex Portal holds almost no binary
cell-surface heteromers. **42 curated gene family patterns** cover them, 12 of which exist specifically for
co-receptor complexes: the ERBB heterodimers, the insulin and insulin-like growth factor receptors, the
transforming growth factor beta receptors including endoglin, the Toll-like receptor heterodimers, the
metabotropic GABA receptor, the Hedgehog receptor with its transducer and co-receptors, the neuropilins with
the vascular endothelial growth factor receptors, the calcitonin receptors with their receptor-activity
modifying proteins, the acid-sensing channels, the aquaporins, and the sterol and organic solute transporter
heterodimers. These account for a further **44 interactions**.

Every curated family is asserted in the unit tests to contain no secreted ligand together with its
receptor, which is the failure mode a family pattern could introduce.

### 6.4 Orientation: which partner goes into the source component

NicheCompass reconstructs the source component from the **neighbours** and the target component from the
**cell itself**, so the partner expressed by the sending cell belongs in the source component and the
partner expressed by the receiving cell in the target component. Getting this backwards asks the model to
predict the ligand in the receiving cell and its receptor in the neighbourhood.

Without evidence the orientation is the order of the two columns in the released table, which is a coin
flip: measured over the 138 contact-dependent interactions with an unambiguous canonical sender, the ligand
landed in the source component 68 times and in the target component 70 times. `orient_juxtacrine_gps`
(default `True`) therefore applies an ordered chain of rules, and the first one that fires decides.

| # | Rule | Evidence | Interactions | Accuracy |
| --: | :-- | :-- | --: | --: |
| 1 | `secreted_partner` | exactly one partner is soluble, so it is released by the neighbour | 529 | physical |
| 2 | `gene_symbol_family` | one symbol names a curated ligand family, the other a curated receptor family | 196 | 100% |
| 3 | `gene_symbol_stem` | one symbol is the other plus an HGNC `LG` or `R` suffix | 1 | 100% |
| 4 | `omnipath_exclusive_role` | OmniPath calls one partner only a ligand and the other only a receptor | 24 | 100% |
| 5 | `omnipath_surface_ligand` | OmniPath calls exactly one partner a ligand that stays on the surface, the other a receptor | 132 | 100% |
| 6 | `go_molecular_function` | one carries a ligand molecular function, the other a receptor molecular function | 18 | 100% gold / 57% held out |
| 7 | `gpi_anchor` | exactly one partner is GPI anchored, so it has no cytoplasmic domain and cannot signal inwards | 18 | 100% |
| 8 | `table_order` | no evidence applies, the released order is kept | 666 | arbitrary |

**918 of 1,584** intercellular interactions (54.2%) are oriented from evidence, and on the 215 curated
directions the chain decided it is **correct 215 times, including 116 of 116 once the ephrins are
excluded**. The rule that decided each program is stored under `orientation_rule` and the counts are
printed. `tests/data/humanppi_orientation_gold.tsv` holds the 224 curated directions and
`tests/benchmark_humanppi_classification.py` gates on both the accuracy and the coverage.

Rules 2 to 7 are reached whenever rule 1 could not decide. That is every contact-dependent interaction, but
also the **327 of 856 paracrine interactions in which both partners are soluble**, which rule 1 cannot
separate: a secreted ligand against a soluble decoy receptor is oriented by the same evidence.

Three design decisions are worth stating, because each was arrived at by measurement rather than assumption:

- **Both sides must match.** A rule fires only when one partner looks like a ligand *and* the other looks
  like a receptor. Comparing a partner that carries receptor evidence against one that carries *no*
  annotation lets absence of evidence choose the ligand; that variant was measured at 64% against
  independent curation, versus 100% for the two-sided form.
- **A bare `L` suffix is not a ligand marker.** `LG` reliably marks one (`KITLG`, `FASLG`, `CD40LG`), but a
  lone `L` far more often abbreviates "like": the rule would call `PXDNL` the ligand of `PXDN`.
- **Two rules were implemented, measured and removed.** Orienting by which partner carries a kinase,
  phosphatase or transducer keyword was correct for 33 of 42 interactions held out against NicheNet,
  CellPhoneDB and OmniPath, and orienting by the larger cytoplasmic domain for only 7 of 17, against 45% for
  the released column order. Both fire almost exclusively on pairs where *both* partners are receptors and
  there is no ligand to find, such as `NOTCH1` with `TLR4`, `DCC` with `UNC5D` and `PTPRG` with `TEK`. The
  keyword and the cytoplasmic domain lengths are still retrieved and cached, but no longer decide anything.

The residual 666 are overwhelmingly interactions between poorly characterised proteins — `FAM209A` with
`GYPA`, `SMIM5` with `TMEM52B`, `CTXN2` with `SERTM1` — where no resource states a direction and
`table_order` is the honest answer. For those programs the interaction class is meaningful and the direction
is not.

Nothing in this section adds an interaction: OmniPath is consulted only to decide which of two partners
already predicted to interact is the sender.

### 6.5 Two symmetric programs per contact-dependent interaction

`symmetric_juxtacrine_gps=True` sidesteps orientation altogether for contact-dependent interactions, by
emitting each one **twice**, once in each orientation, so neither has to be the right one. It also closes a
coverage gap: with one orientation, 557 of the 812 genes taking part in a contact-dependent interaction
appear in only one component across the whole prior set, so no named program ever reconstructs them from the
other one. With both orientations, none do.

It is nevertheless **off by default**, because it adds 769 programs at precision 90 (1,584 to 2,353
intercellular, +49%) and because the arithmetic of the biology does not justify applying it to the whole
set. Contact-dependent signalling is frequently *bidirectional* — ephrin-B cytoplasmic tails signal in the
ligand-expressing cell, so do Notch ligand intracellular domains, membrane TNF, B7 and MHC — but
bidirectional is not the same as *symmetric*: the two arms use different effectors and produce different
outputs, so the two orientations are not interchangeable descriptions of one event. Genuinely symmetric
interactions, where the two partners are the same molecule, number **0 of 863**: the interactome screened
heterodimeric pairs only, so homophilic adhesion is absent from this resource by construction. Only 177 of
863 (20.5%) have both partners in a family where bidirectional signalling is the norm, and 113 of those 177
are ephrin/Eph.

The cost is in reading rather than in statistics. Differential program testing applies a fixed log Bayes
factor threshold with no multiple-testing correction, so doubling the program count carries no statistical
penalty; but a mirrored pair occupies two rows of every program summary, two of the ten panels of
`generate_enriched_gp_info_plots`, and two arrow families in the communication network for one physical
contact. The two names differ only in gene order and read as a typo.

Turn it on when the *side* of a contact-dependent interface is itself the question — for example when asking
whether a checkpoint interaction is enriched from the tumour cell's side, the T cell's side, or both. Note
that `filter_and_combine_gp_dict_gps_v2` merges programs sharing a source gene, and every intercellular
program has exactly one source gene, so combining collapses the mirrored set to one program per source gene
and discards the interaction class; only 75 of the 863 mirrored pairs survive as pairs.

## 7. Gene program construction

- **Name**: `<source gene>_<target gene>_<interaction class>_ppi_GP`, so the class is visible in gene
  program summaries, differential gene program results and plots.
- **Gene categories**: the location class of each gene (`cell_surface`, `secreted`, `intracellular`,
  `ambiguous`), rather than a single flat label. They flow into the category masks and can be used with
  `l1_targets_categories` and `l1_sources_categories`.
- **Deduplication**: on the unordered, upper-cased gene pair, so a pair listed twice yields one program.
- **Target-only programs** have an empty source component and therefore require
  `min_source_genes_per_gp=0` downstream; the extractor emits a warning whenever it produces them.

### Species

The predictions are human. With `species="mouse"`, genes are mapped to mouse orthologs using the Ensembl
BioMart mapping shipped with NicheCompass; one human gene can map to several mouse genes, in which case the
gene category is repeated for each ortholog, and genes without a valid ortholog fall back to capitalisation.
**Gene program names keep the human symbols**, as they do for the OmniPath programs, so a program named
`TGFB1_..._GP` contains `Tgfb1`.

## 8. Output at a glance

At `precision="90"`, `species="human"`, `program_type="both"` and otherwise default settings:

| Class | Count | Share |
| :-- | --: | --: |
| `paracrine` | 815 | 4.8% |
| `juxtacrine` | 769 | 4.6% |
| `cis_complex` | 551 | 3.3% |
| `extracellular_assembly` | 98 | 0.6% |
| `intracellular` | 14,614 | 86.7% |
| **total** | **16,847** | |

Intercellular programs total 1,584. At `precision="80"` the totals are 27,823 programs of which 3,088 are
intercellular (1,424 paracrine, 1,664 juxtacrine).

With `use_topology=False` the reach test cannot run and the curated families are the only *cis* evidence
left, so the intercellular set grows rather than shrinks: 2,485 programs at precision 90 with
`ambiguous_locality="extracellular"`, 1,745 with `"intracellular"`. Both are less trustworthy than the
1,584 obtained with topology.

## 9. Downstream use and three caveats

1. **Masking thresholds.** `add_gps_from_gp_dict_to_adata` decides which programs survive against the genes
   actually measured. An intercellular program is two genes wide, so with
   `min_source_genes_per_gp=0, min_target_genes_per_gp=0` a program survives when only *one* partner is
   probed, leaving a single-gene program that encodes no interaction. Use `1` and `1` unless you also want
   target-only programs, which require `min_source_genes_per_gp=0`. The two requirements are mutually
   exclusive, because the threshold is global rather than per resource.
2. **Combining discards the interaction class.** `filter_and_combine_gp_dict_gps_v2` merges programs sharing
   a source gene and renames them `<GENE>_combined_GP`, which loses the `_paracrine_`/`_juxtacrine_` label.
3. **Targeted panels.** An intercellular program needs *both* partners probed. On the 313-gene Xenium human
   breast panel only 31 of 3,088 intercellular programs retain both partners (17 of 1,584 at
   precision 90). The resource is best suited to
   whole-transcriptome data; on a targeted panel it is more informative as a supplement to
   OmniPath, NicheNet and MEBOCOST than on its own.

## 10. Validation

- **Recall against curated ligand-receptor pairs.** Curated pairs are intercellular by definition, so every
  such pair present in the interactome should be classified paracrine or juxtacrine. Measured against
  OmniPath: **97.0%** (392 of 404 testable pairs). `tests/benchmark_humanppi_classification.py` reruns this
  and fails below a threshold.

  The twelve misses are worth reading individually, because most are not classification errors:

  | Miss | Our class | Assessment |
  | :-- | :-- | :-- |
  | `NCSTN`-`PSEN1` | `intracellular` | Gamma-secretase subunits in one membrane — reference is wrong |
  | `LRPAP1`-`SORL1` | `intracellular` | Endoplasmic reticulum chaperone pair — reference is wrong |
  | `CALM3`-`MYLK2` | `intracellular` | Cytosolic — reference is wrong |
  | `CNTF`-`CNTFR`, `CNTF`-`IL6R` | `intracellular` | Ciliary neurotrophic factor has no signal peptide and is released by injury rather than secreted |
  | `KDR`-`NRP1` | `cis_complex` | Neuropilin-1 is a VEGFR2 co-receptor **on the same cell** — our call is correct |
  | `GAS1`-`PTCH1` | `cis_complex` | GAS1 is a Hedgehog co-receptor on the same cell — our call is correct |
  | `BMPR2`-`ENG` | `cis_complex` | Endoglin is a TGF-beta co-receptor on the same cell — our call is correct |
  | `TNF`-`TNFRSF1A`, `TNF`-`TNFRSF1B`, `LTB`-`LTBR` | `cis_complex` | Genuine losses: membrane-anchored ligands inside multi-subunit Complex Portal entries |
  | `NECTIN2`-`PVRIG` | `cis_complex` | Genuine loss: a real *trans* immune checkpoint |

  So three of the twelve are cases where curated resources register a same-cell co-receptor complex as a
  ligand-receptor pair, and five are cases where the reference is wrong. **Five are genuine losses**, four of
  them already present before the reach test. Measured recall therefore understates real performance; the
  headline number is reported unadjusted regardless.
- **Canonical cis pairs.** All 15 same-cell pairs listed under limitation 2 below are now classified
  non-intercellular. Before the reach test and the curated co-receptor families, 15 of 15 were `juxtacrine`.
- **True trans pairs preserved.** `PDCD1`-`CD274`, `SIRPA`-`CD47`, `EPHB1`-`EFNB3`, `CCL19`-`CCR7`,
  `IL15`-`IL15RA` and `TNFSF4`-`TNFRSF4` all remain intercellular.
- **Orientation against curated direction.** `tests/data/humanppi_orientation_gold.tsv` holds 224
  interactions whose sender is unambiguous from established cell biology. Of the 215 the rules decide, the
  ligand is placed in the source component **215 times (100%)**, and **116 of 116** once the ephrins are
  excluded, against 104 of 224 (46.4%) for the released column order. Coverage is 918 of 1,584 (58.0%).
  Because the gold set and the curated family lexicon share provenance, the chain was additionally scored
  against NicheNet, CellPhoneDB and the OmniPath intercell annotation on interactions the gold set does not
  cover: **243 of 246 (98.8%)**, with all three disagreements on pairs that are genuinely bidirectional
  (`TNFRSF14` with `BTLA` and with `TIGIT`, `SELPLG` with `SPN`).
  `tests/benchmark_humanppi_classification.py` gates on both the accuracy and the coverage.
- **Unit tests.** `tests/test_humanppi_gene_programs.py` contains 246 offline tests pinning the protein and
  interaction classification, the precedence rules, the fallback ordering, the reach test, the segment and
  paralogue patterns, the curated families and every orientation rule including its abstentions.

## 11. Known limitations

1. **Interface topology is not used.** Whether an interaction interface lies on the extracellular side is the
   biologically correct criterion, and the resource ships residue-level contact matrices that would answer
   it, but only inside a 67 GB archive. Everything here is a protein-level proxy.
2. **The *trans* test is a physical proxy, not a functional one.** A positive test for *trans* now exists
   (section 6.2), but reach is necessary rather than sufficient: a protein with a long ectodomain may still
   only ever partner within its own membrane. The 15 canonical same-cell pairs probed are all now classified
   non-intercellular, but they divide instructively. Seven were caught by the reach test alone
   (`KCNJ6`-`KCNJ9`, `HCN1`-`HCN4`, `CNGA1`-`CNGB1`, `ORAI2`-`ORAI3`, `CD247`-`FCER1G`, `FCER1G`-`TREM2`,
   `SLC51A`-`SLC51B`). Eight had ectodomains far too long for any topological test to reject —
   `ERBB2`-`ERBB3` (630 residues), `INSR`-`IGF1R` (731), `NRP1`-`KDR` (835), `TLR1`-`TLR2` (568),
   `GABBR1`-`GABBR2` (572), `PTCH1`-`SMO` (320), `ASIC4`-`ASIC3` (349), `CALCR`-`RAMP1`, `BMPR1A`-`BMPR2`,
   `ABCG5`-`ABCG8` — and required curated families (section 6.3). **Curation does not generalise**, so
   co-receptor complexes outside those 36 families are still expected to be called `juxtacrine`. Two survive
   even on the 313-gene breast panel: `CD4`-`CD3D` and `CD8A`-`CD3D` are a T cell co-receptor engaging the
   T cell receptor complex on the same cell, and both are still classified `juxtacrine`. The residual error
   rate among the 863 juxtacrine programs is unmeasured.
3. **Precision is not measured globally.** Every specific false positive found has been fixed, but there is
   no clean negative gold standard, so the fraction of the 1,584 intercellular calls that are wrong is
   unknown. Recall is measured; precision is argued.
4. **524 interactions remain unresolved** and are dropped by default. Closing that would need sequence-based
   prediction such as SignalP or DeepTMHMM rather than another annotation source, which is exhausted.
5. **Two-gene programs are sparse** as single latent dimensions, and nothing checks that a program's partners
   are ever co-expressed in adjacent cells in the data at hand.

## 12. Complete parameter reference

Grouped by what they control. Defaults are those of
`extract_gp_dict_from_humanppi_interactions`; the training script exposes each one as
`--humanppi_<name>` and the notebook as `humanppi_<name>`.

### What is retrieved

| Parameter | Default | Effect |
| :-- | :-- | :-- |
| `species` | required | `"human"` uses the predictions as they are; `"mouse"` maps genes to mouse orthologs |
| `precision` | `"90"` | Which released table: `"90"` (17,849 rows) or `"80"` (29,257 rows, a superset) |
| `load_from_disk` | `False` | Read the predictions from `ppi_network_file_path` instead of downloading |
| `save_to_disk` | `False` | Cache the downloaded predictions, with a provenance file |
| `ppi_network_file_path` | `../data/gene_programs/humanppi_network.csv` | Cache location for the predictions |
| `humanppi_predictions_url` | `conglab.swmed.edu/...` | Where the prediction archive is downloaded from |

### Which interactions are kept

| Parameter | Default | Effect |
| :-- | :-- | :-- |
| `filter_ig_tcr_segments` | `True` | Drop immunoglobulin and T cell receptor V, D and J gene segments (section 3.2) |
| `filter_paralog_cross_pairs` | `True` | Drop paralogue combinations that form no heterodimer (section 3.3) |
| `min_rf_prob` | `None` | Minimum RoseTTAFold2-PPI probability (`RFprob`) |
| `min_af_prob` | `None` | Minimum AlphaFold2 probability (`AFprob`) |

### How interactions are classified

| Parameter | Default | Effect |
| :-- | :-- | :-- |
| `program_type` | `"intercellular"` | `"intercellular"` keeps paracrine and juxtacrine; `"intracellular"` keeps the rest; `"both"` keeps everything |
| `use_topology` | `True` | Retrieve UniProt membrane topology and annotation (section 4.1). Needs network access on first use |
| `topology_file_path` | `../data/gene_programs/humanppi_protein_topology.tsv` | Cache location for the UniProt annotation |
| `ambiguous_locality` | `"extracellular"` | How to treat proteins whose extracellular face is neither established nor contradicted (section 5.3). Inert unless `use_topology=False` |
| `unresolved_locality` | `"exclude"` | How to treat interactions whose partner locations stay unresolved: `"exclude"` drops them, `"intracellular"` keeps them as target-only programs |
| `detect_cis_complexes` | `True` | Reclassify subunits of a common complex as `cis_complex` or `extracellular_assembly` (section 6.1) |
| `orient_juxtacrine_gps` | `True` | Orient interactions from evidence about which partner sends the signal, so the ligand lands in the source component (section 6.4). `False` keeps the released column order |
| `omnipath_annotation_file_path` | `../data/gene_programs/omnipath_intercell_annotation.tsv` | Cache for the OmniPath intercell annotation used by orientation rules 4 and 5. A failure to retrieve it costs orientation coverage, not the mask |
| `min_extracellular_domain_length` | `30` | Shortest extracellular domain, in residues, that still counts as able to reach a partner on a neighbouring cell (section 6.2). `0` disables the reach test. Requires `use_topology=True` |
| `symmetric_juxtacrine_gps` | `False` | Emit each contact-dependent interaction in both orientations, since it operates in both directions between two neighbouring cells (section 6.4). Doubles the number of `juxtacrine` programs |
| `complex_portal_file_path` | `../data/gene_programs/complex_portal_human.tsv` | Cache location for the Complex Portal table |
| `complex_portal_url` | `ftp.ebi.ac.uk/...` | Where the Complex Portal table is downloaded from |
| `gene_orthologs_mapping_file_path` | `../data/gene_annotations/human_mouse_gene_orthologs.csv` | Ensembl BioMart mapping, used only when `species="mouse"` |

### Diagnostics

| Parameter | Default | Effect |
| :-- | :-- | :-- |
| `plot_gp_gene_count_distributions` | `True` | Plot the distribution of gene programs by number of source and target genes |
| `gp_gene_count_distributions_save_path` | `None` | Where to save that plot |

### A worked invocation

The settings used by the Xenium human breast cancer notebook, which isolates the intercellular
contribution of this resource on a targeted panel:

```python
humanppi_gp_dict = extract_gp_dict_from_humanppi_interactions(
    species="human",
    precision="80",                     # broader set, since the panel is small
    program_type="intercellular",       # paracrine and juxtacrine only
    ambiguous_locality="extracellular",
    unresolved_locality="exclude",
    filter_ig_tcr_segments=True,
    filter_paralog_cross_pairs=True,
    use_topology=True,
    detect_cis_complexes=True,
    min_extracellular_domain_length=30,
    orient_juxtacrine_gps=True,
    symmetric_juxtacrine_gps=False,
    min_rf_prob=None,                   # trust the authors' calibration
    min_af_prob=None)
```

## 13. References

- Zhang, J., Humphreys, I. R. et al. Predicting protein-protein interactions in the human proteome.
  *Science* **390**, eadt1630 (2025). doi:10.1126/science.adt1630
- Birk, S. et al. Quantitative characterization of cell niches in spatially resolved omics data.
  *Nature Genetics* **57**, 897-909 (2025). doi:10.1038/s41588-025-02120-6
- Meldal, B. H. M. et al. Complex Portal 2022. *Nucleic Acids Research* **50**, D578-D586 (2022).
- The UniProt Consortium. UniProt: the Universal Protein Knowledgebase in 2023.
  *Nucleic Acids Research* **51**, D523-D531 (2023).
