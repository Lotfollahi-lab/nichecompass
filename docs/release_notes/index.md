# Release notes

All notable changes to this project will be documented in this file. The format
is based on [keep a changelog], and this project adheres to
[Semantic Versioning]. Full commit history is available in the [commit logs].

### 0.3.2 (04.11.2025)

-   Remove fixed versioning of package dependencies for uv installation with jax CUDA 12.4 support.
[@sebastianbirk]

### 0.3.1 (04.11.2025)

-   Fixed issue in retrieval of omnipath gene programs due to omnipath db version update.
[@sebastianbirk]

### 0.3.0 (07.08.2025)

-   Fixed dependency issues in optional dependencies and updated decoupler to version 2.0.0.
[@sebastianbirk]

### 0.2.4 (02.08.2025)

-   Fixed issue in pyproject.toml that prevented "pip install nichecompass[all]" from installing optional dependencies.
[@sebastianbirk]

### 0.2.3 (17.02.2025)

-   Added numpy<2 dependency as version upgrade of NumPy to major version 2 breaks required scanpy version.
[@sebastianbirk]

### 0.2.2 (09.01.2025)

-   Synced repository with Zenodo to mint DOI for publication.
[@sebastianbirk]

### 0.2.1 (15.10.2024)

-   Added a user guide to the package documentation.
[@sebastianbirk]

### 0.2.0 (22.08.2024)

-   Fixed a bug in the configuration of random seeds.
-   Fixed a bug in the definition of MEBOCOST prior gene programs.
-   Raised the default quality filter threshold for the retrieval of OmniPath gene programs.
-   Modified the GP filtering logic to combine GPs with the same source genes and drop GPs that do not have source genes if they have a specified overlap in target genes with other GPs.
-   Changed the default hyperparameters for model training based on ablation experiments.
[@sebastianbirk]

### 0.1.2 (13.02.2024)

-   The version was incremented due to package upload requirements.
[@sebastianbirk]

### 0.1.1 (13.02.2024)

-   The version was incremented due to package upload requirements.
[@sebastianbirk]

### 0.1.0 (13.02.2024)

-   First NicheCompass version.
[@sebastianbirk]

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[Semantic Versioning]: https://semver.org/spec/v2.0.0.html
[commit logs]: https://github.com/Lotfollahi-lab/nichecompass/commits
[@sebastianbirk]: https://github.com/sebastianbirk
