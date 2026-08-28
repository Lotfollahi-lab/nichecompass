# User guide

## Hyperparameter selection

We conducted various ablation experiments on both simulated and real spatial transcriptomics data to evaluate important model design choices and hyperparameters. The detailed results and interpretations can be found in the NicheCompass manuscript.

In summary, we recommend the following:
- Regarding the loss, we observed that finding a balance between gene expression and edge reconstruction is a key element for good niche identification (NID) and GP recovery (GPR) performance, while regularization of de novo GP weights is essential for GPR. The loss weights have been specified in the NicheCompass package accordingly and in most cases do not need to be changed by the user.
- With respect to size of the KNN neighborhood graph, a smaller number of neighbors is more efficient in NID and de novo GP detection while a larger number of neighbors can facilitate GPR of prior GPs. Here, we recommend users to specify a neighborhood size based on the expected range of interactions in the tissue. Empirically, we observed a neighborhood size between 4 and 12 to work well. 
- The inclusion of de novo GPs is crucial for GPR. However, the number of de novo GPs should not be too high as important genes might be split across multiple GPs. The default in the NicheCompass package is 100 and we do not recommend to increase this number.
- GP pruning can slighlty improve GPR and NID while reducing the embedding size of the model; we therefore recommend users to use the default setting of weak GP pruning.
- We recommend that users define prior GPs solely based on the biology that they are interested in (as opposed to including as many prior GPs as possible).
- We recommend users to use a GATv2 encoder layer (as opposed to a GCNConv encoder layer) unless performance is a bottleneck or niche characterization is not a priority and the data has single-cell resolution. 
- Since the use of prior GPs can significantly improve NID compared to a scenario without prior GPs, we recommend users to use the default set of prior GPs even if interpretability is not a main objective.

## Training on several GPUs

Training can be split across several GPUs, with one process per GPU. The batch sizes keep their meaning as
global batch sizes, so the number of optimizer steps per epoch is unchanged and only the work per step is
divided between the devices. Single-device training is unaffected and is what the option defaults to.
See [multi_gpu_training](multi_gpu_training.md) for how to launch it, what it guarantees numerically, and
what has and has not been verified.

## Prior gene program resources

NicheCompass ships several prior gene program resources, retrieved through the functions documented in the
API. Most are a direct translation of a curated database into gene programs. The human protein-protein
interaction resource is the exception: because the predicted human interactome is undirected and dominated by
intracellular complexes, it requires a substantial classification step before it can be used as a
neighbour-to-self signalling prior. That workflow is documented in full, step by step, here:

```{toctree}
:maxdepth: 1

humanppi_gene_programs
multi_gpu_training
```
