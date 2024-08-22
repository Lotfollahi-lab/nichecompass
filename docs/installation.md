# Installation

NicheCompass is available for Python 3.9. It does yet not support Apple silicon.


We do not recommend installation on your system Python. Please set up a virtual
environment, e.g. via conda through the [Mambaforge] distribution, or create a
[Docker] image.

## Additional Libraries

To use NicheCompass, you need to install some external libraries. These include:
- [PyTorch]
- [PyTorch Scatter]
- [PyTorch Sparse]
- [bedtools]

We recommend to install the PyTorch libraries with GPU support. If you have
CUDA, this can be done as:

```
pip install torch==${TORCH}
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```
where `${TORCH}` and `${CUDA}` should be replaced by the specific PyTorch and
CUDA versions, respectively.

For example, for PyTorch 2.0.0 and CUDA 11.7, type:
```
pip install torch==2.0.0
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

To install bedtools, you can use conda:
```
conda install bedtools=2.30.0 -c bioconda
```

Alternatively, we have provided a conda environment file with all required
external libraries, which you can use as:
```
conda env create -f environment.yaml
```

To enable GPU support for JAX, after the installation run:
```
pip install jaxlib==0.3.25+cuda${CUDA}.cudnn${CUDNN} -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

For example, for CUDA 11.7, type:
```
pip install jaxlib==0.4.7+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```


## Installation via PyPi

Subsequently, install NicheCompass via pip:
```
pip install nichecompass
```

Install optional dependencies required for benchmarking, multimodal analysis, running tutorials etc. with:
```
pip install nichecompass[all]
```

[Mambaforge]: https://github.com/conda-forge/miniforge
[Docker]: https://www.docker.com
[PyTorch]: http://pytorch.org
[PyTorch Scatter]: https://github.com/rusty1s/pytorch_scatter
[PyTorch Sparse]: https://github.com/rusty1s/pytorch_sparse
[bedtools]: https://bedtools.readthedocs.io
