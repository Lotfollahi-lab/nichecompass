# Installation

NicheCompass is available for Python 3.10. We recommend to train NicheCompass
models on a device with GPU support. Apple silicon or multi-GPU training is not
yet supported.

We do not recommend installation on your system Python. Please set up a virtual
environment, e.g. via conda through the [Mambaforge] distribution or [python-venv],
or create a [Docker] image.

If you want to train NicheCompass on multimodal data, we recommend to use conda.
For the fastest installation experience for unimodal training, use the [uv]
package manager within a [python-venv] environment. For example, run:

```
python3 -m venv ${/path/to/new/virtual/environment}
source ${/path/to/new/virtual/environment}/bin/activate
pip install uv
```
where `${/path/to/new/virtual/environment}` should be replaced with the path
where you want to install the virtual environment.

## Additional Libraries

To use NicheCompass, you first need to install some external libraries. These
include:
- [PyTorch]
- [PyTorch Scatter]
- [PyTorch Sparse]
- [bedtools]

Bedtools is only required for multimodal training including paired ATAC-seq data.

We recommend to install the PyTorch libraries with GPU support. If you have
CUDA, this can be done as:

```
uv pip install torch==${TORCH}+${CUDA} --extra-index-url https://download.pytorch.org/whl/${CUDA}
uv pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```
where `${TORCH}` and `${CUDA}` should be replaced by the specific PyTorch and
CUDA versions, respectively.

For example, for PyTorch 2.6.0 and CUDA 12.4, type:
```
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

If you want to install bedtools, we recommend to use conda:
```
conda install bedtools=2.30.0 -c bioconda
```

Alternatively, we have provided a conda environment file with all required
external libraries for PyTorch 2.0.0 and CUDA 11.7, which you can use as:
```
conda env create -f environment.yaml
```

## Installation via PyPi

Subsequently, install NicheCompass via pip:
```
uv pip install nichecompass
```

Or install optional dependencies required for benchmarking, multimodal analysis, running tutorials etc. with:
```
uv pip install nichecompass[all]
```

To enable GPU support for JAX (recommended for benchmarking), after the installation run:
```
uv pip install jax[cuda${CUDA}]
```
where `${CUDA}` should be replaced by your major CUDA version. For example:

```
uv pip install jax[cuda12]
```

[Mambaforge]: https://github.com/conda-forge/miniforge
[python-venv]: https://docs.python.org/3/library/venv.html
[uv]: https://docs.astral.sh/uv/getting-started/installation
[Docker]: https://www.docker.com
[PyTorch]: http://pytorch.org
[PyTorch Scatter]: https://github.com/rusty1s/pytorch_scatter
[PyTorch Sparse]: https://github.com/rusty1s/pytorch_sparse
[bedtools]: https://bedtools.readthedocs.io
