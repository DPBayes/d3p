# d3p - Differentially Private Probabilistic Programming

[![Build Status](https://travis-ci.com/DPBayes/d3p.svg?branch=master)](https://travis-ci.com/DPBayes/d3p)

d3p is an implementation of the differentially private variational inference algorithm [1] for [NumPyro](https://github.com/pyro-ppl/numpyro), using [JAX](https://github.com/google/jax/) for auto-differentiation and fast execution on CPU and GPU.

## Current Status

The software is under ongoing development and specific interfaces may change suddenly.

Since both NumPyro and JAX are evolving rapdily, some convenience features implemented in d3p are made obsolete by being introduced in these upstream packages. In that case, they will be gradually replaced but there might be some lag time.

## Installing

d3p is pure Python software. You can install it via pip with the following command:
```
pip install git+https://github.com/DPBayes/d3p.git@master#egg=d3p
```

Alternatively, you can clone this git repository and install with pip locally:
```
git clone https://github.com/DPBayes/d3p
cd d3p
pip install .
```

This will install d3p with all required dependencies (NumPyro, JAX, ..) for CPU usage.

### Note about dependency versions

NumPyro and JAX are still under ongoing development and its developers currently give no
guarantee that the API remains stable between releases. In order to allow for users
of d3p to benefit from latest features of NumPyro, we did not put a strict upper bound
on the NumPyro dependency version. This may lead to problems if newer NumPyro versions
introduce breaking API changes.

If you encounter such issues at some point,
you can use the `compatible-dependencies` installation target of d3p to force usage of the latest
known set of dependency versions known to be compatible with d3p:
```
pip install git+https://github.com/DPBayes/d3p.git@master#egg=d3p[compatible-dependencies]
```

### GPU installation
If you want to run on CUDA devices, replace the last command with:

```
pip install git+https://github.com/DPBayes/d3p.git@master#egg=d3p[cuda111] -f https://storage.googleapis.com/jax-releases/jax_releases.html
```

Adapt `cuda111` to your CUDA version, e.g., `cuda102` for CUDA 10.2 .

### TPU installation
Replace the installation command with
```
pip install git+https://github.com/DPBayes/d3p.git@master#egg=d3p[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

Refer to the [JAX installation instructions](https://github.com/google/jax#pip-installation-gpu-cuda)
for more details.

## Usage

For now, see the examples in `examples/`.

## Versioning Policy

The `master` branch contains the latest development version of d3p which may introduce breaking changes.

Version numbers adhere to [Semantic Versioning](https://semver.org/). Changes between releases are tracked in `ChangeLog.txt`.

## License

d3p is licensed under the Apache 2.0 License. The full license text is available
in `LICENSE.txt`. Copyright holder of each contribution are the respective
contributing developer or his or her assignee (i.e., universities or companies
owning the copyright of software contributes by their employees).

## Acknowledgements

We thank the NVIDIA AI Technology Center Finland for their contribution of GPU performance benchmarking and optimisation.

## References

1. J. Jälkö, O. Dikmen, A. Honkela:
Differentially Private Variational Inference for Non-conjugate Models
https://arxiv.org/abs/1610.08749
