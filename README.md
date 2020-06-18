# dp³ - Differentially Private Probabilistic Programming

dp³ is an implementation of the differentially private variational inference algorithm [1] for [NumPyro](https://github.com/pyro-ppl/numpyro), using [JAX](https://github.com/google/jax/) for auto-differentiation and fast execution on CPU and GPU.

## Current Status

The software is under ongoing development and specific interfaces may change suddenly.

Since both NumPyro and JAX are evolving rapdily, some convenience features implemented in dp³ are made obsolete by being introduced in these upstream packages. In that case, they will be gradually replaced but there might be some lag time.

## Installing

dp³ is pure Python software. Simply clone this git repository and install with pip:
```
git clone https://github.com/DPBayes/dppp
cd dppp
pip install -e .
```

A packaged version of dp³ will be available on pypi.org soon.

## Usage

For now, see the examples in `examples/`.

## Versioning Policy

The `master` branch contains the latest stable development version of dp³.

In the future, we will adopt semantic versioning. The first version will be released as `0.1.0` shortly. Commits corresponding to version releases will be tagged accordingly.

A list of important changes between releases will be available in `ChangeLog.txt`.

## License

dp³ is licensed under the Apache 2.0 License. The full license text is available
in `LICENSE.txt`. Copyright holder of each contribution are the respective
contributing developer or his or her assignee (i.e., universities or companies
owning the copyright of software contributes by their employees).

## References

1. J. Jälkö, O. Dikmen, A. Honkela:
Differentially Private Variational Inference for Non-conjugate Models
https://arxiv.org/abs/1610.08749