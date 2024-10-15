# Release notes

All notable changes to this project will be documented in this file. The format
is based on [keep a changelog], and this project adheres to
[Semantic Versioning]. Full commit history is available in the [commit logs].

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
