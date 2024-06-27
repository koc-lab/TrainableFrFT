# Trainable Fractional Fourier Transform

In this repository, we present the source code for the experiments of our [_Trainable Fractional Fourier Transform_] (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10458263 )paper is accepted to _IEEE Signal Processing Letters_. The installable package [`torch-frft`](https://github.com/tunakasif/torch-frft) is maintained at its own [GitHub page](https://github.com/tunakasif/torch-frft). The package is available on both [PyPI](https://pypi.org/project/torch-frft/) and [Conda](https://anaconda.org/conda-forge/torch-frft). Installation instructions are provided below. Please use the following BibTeX entry to cite our work:

```bibtex
@article{trainable-frft-2024,
  author   = {Koç, Emirhan and Alikaşifoğlu, Tuna and Aras, Arda Can and Koç, Aykut},
  journal  = {IEEE Signal Processing Letters},
  title    = {Trainable Fractional Fourier Transform},
  year     = {2024},
  volume   = {31},
  number   = {},
  pages    = {751-755},
  keywords = {Vectors;Convolution;Training;Task analysis;Computational modeling;Time series analysis;Feature extraction;Machine learning;neural networks;FT;fractional FT;deep learning},
  doi      = {10.1109/LSP.2024.3372779}
}
```

## Installation of `torch-frft`

You can install the package directly from [PyPI](https://pypi.org/project/torch-frft/) using `pip` or `poetry` as follows:

```sh
pip install torch-frft
```

or

```sh
poetry add torch-frft
```

or directly from [Conda](https://anaconda.org/conda-forge/torch-frft):

```sh
conda install -c conda-forge torch-frft
```
