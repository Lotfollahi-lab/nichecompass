<img src="https://github.com/Lotfollahi-lab/nichecompass/blob/main/docs/_static/nichecompass_logo_readme.png" width="800" alt="nichecompass-logo">

NicheCompass (**N**iche **I**dentification based on **C**ellular grap**H** **E**mbeddings of **COM**munication **P**rograms **A**ligned across **S**patial **S**amples) is a package for end-to-end analysis of spatial multi-omics data, including spatial atlas building, niche identification & characterization, cell-cell communication inference and spatial reference mapping. It is built on top of [PyG](https://pytorch-geometric.readthedocs.io/en/latest/) and [AnnData](https://anndata.readthedocs.io/en/latest/).

## Installation
1) Create the nichecompass conda environment:
```conda env create -f environment.yml```

2) Activate the nichecompass conda environment:
```conda activate nichecompass```

4) Install all Python dependencies via Poetry:
```poetry install``` (you might have to run ```export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring``` in your shell before)

## Tutorials
Tutorial notebooks are available in the notebooks folder.

### Single Sample Tutorial
In this tutorial, we apply NicheCompass to a single sample (sagittal brain section) of the STARmap PLUS mouse central nervous system dataset / atlas from [Shi, H. et al. Spatial Atlas of the Mouse Central Nervous System at Molecular Resolution. bioRxiv 2022.06.20.496914 (2022)](https://www.biorxiv.org/content/10.1101/2022.06.20.496914v1).

### Oneshot Sample Integration Tutorial
In this tutorial, we apply NicheCompass to integrate three samples (sagittal brain sections) of the STARmap PLUS mouse central nervous system dataset / atlas from [Shi, H. et al. Spatial Atlas of the Mouse Central Nervous System at Molecular Resolution. bioRxiv 2022.06.20.496914 (2022)](https://www.biorxiv.org/content/10.1101/2022.06.20.496914v1) in a oneshot setting (i.e. the model is trained on all samples at once).

### Reference Query Mapping Tutorial
In this tutorial, we apply NicheCompass to integrate three samples (sagittal brain sections) of the STARmap PLUS mouse central nervous system dataset / atlas from [Shi, H. et al. Spatial Atlas of the Mouse Central Nervous System at Molecular Resolution. bioRxiv 2022.06.20.496914 (2022)](https://www.biorxiv.org/content/10.1101/2022.06.20.496914v1) in a query to reference mapping setting (i.e. the model is first trained on the reference samples, and the query samples are then mapped onto the pretrained reference model by finetuning it). The first two samples are used as reference samples and the third sample is the query.

### Multimodal Tutorial
In this tutorial, we apply NicheCompass to a single multimodal sample (postnatal day 22 coronal section) of a spatial ATAC-RNA-seq mouse brain dataset from [Zhang, D. et al. Spatial epigenome–transcriptome co-profiling of mammalian tissues. Nature 1–10 (2023)](https://www.nature.com/articles/s41586-023-05795-1).

## Reference
