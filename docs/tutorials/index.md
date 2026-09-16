# Tutorials

Get started with NicheCompass by following our tutorials.

The four main notebooks each include a worked GP analysis walkthrough: automatic sign
orientation, quality diagnostics, posterior activities and uncertainty, signed
feature loadings, complete differential results, exports and communication
networks within spatial samples. The multimodal tutorial shows RNA and ATAC
members together; the reference-mapping tutorial checks that matching GPs retain
their reference signs during fine-tuning. See the
[GP analysis guide](../user_guide/gene_program_analysis.md) for API details and
interpretation limits.

Two single-sample notebooks use the same STARmap PLUS mouse CNS section. The
first builds the default prior mask from OmniPath, NicheNet and MEBOCOST; the
second builds it from the predicted human interactome alone, which makes the
classification that turns an undirected protein-pair list into a directed
neighbourhood prior visible. That second notebook stops after training and
defers the GP analysis to the first, since a single-resource mask on a
1022-gene panel retains too few prior programs to interpret. For training
across several GPUs see the
[multi-GPU guide](../user_guide/multi_gpu_training.md) - it needs one process
per GPU, which a notebook kernel cannot provide.

```{toctree}
:maxdepth: 2

single_sample_tutorials
sample_integration_tutorials
spatial_reference_mapping_tutorials
multimodal_tutorials
```
