# NicheCompass
**N**iche **I**dentification based on **C**ellular grap**H** **E**mbeddings of **COM**munication **P**rograms **A**ligned across **S**patial **S**amples.

## Usage

NicheCompass can be run from the command line using a configuration file. It expects an `AnnData` object as input, with raw counts. Start by creating a skeleton configuration file, and customising this to your dataset and run parameters.

```bash
nichecompass skeleton run-config.yml
```

Build the gene programs and datasets required for a NicheCompass run.

```bash
nichecompass build-gene-programs run-config.yml
nichecompass build-dataset run-config.yml
```

Train a NicheCompass model. Model training is computationally intensive, and will automatically use GPU resources if available.

```bash
nichecompass train run-config.yml
```

## Contributing

### Devcontainer

It is recommended to use a `devcontainer` to create a reproducible development environment. Within the development container, install poetry and package dependencies.

```bash
pipx install poetry==1.4.2
poetry install
```

### Conda

To create a development environment using Conda.

```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
conda env create -f environment.yml
conda activate nichecompass
poetry install
```

### Tests

Run integration tests.

```bash
poetry run pytest tests/integration/training.py
```

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
