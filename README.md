# d3p - Differentially Private Probabilistic Programming

d3p is an implementation of the differentially private variational inference (DP-VI) algorithm [2] for [NumPyro](https://github.com/pyro-ppl/numpyro), using [JAX](https://github.com/google/jax/) for auto-differentiation and fast execution on CPU and GPU.

It is designed to provide differential privacy guarantees for tabular data, where each individual contributed a single sensitive record (/ data point / example).

## Current Status

The software is under ongoing development and specific interfaces may change suddenly.

Since both NumPyro and JAX are evolving rapdily, some convenience features implemented in d3p are made obsolete by being introduced in these upstream packages. In that case, they will be gradually replaced but there might be some lag time.

## Usage

The main component of `d3p` is the implementation of DP-VI in the `d3p.svi.DPSVI` class.
It is intended to work as a drop-in replacement of NumPyro's `numpyro.svi.SVI` and works
with models specified using NumPyro distributions and modelling features.

For a general introduction on writing models for NumPyro, please refer to the [NumPyro documentation](https://num.pyro.ai/en/stable/getting_started.html).

Then instead of `numpyro.svi.SVI`, you can simply use `d3p.svi.DPSVI`.

### Main Differences in `DPSVI`
Since it provides differential privacy, there are additional parameters to the `DPSVI` class compared to `numpyro.svi.SVI`:

#### `clipping_threshold`
Differential privacy requires gradients of individual data points (examples) in your data set
to be clipped to a maximum norm, to limit the contribution any single example can have. The parameter `clipping_threshold`
sets the threshold over which these gradient norms are clipped.

#### `dp_scale`
This is the scale of the noise σ for the Gaussian mechanism that is used to perturb gradients in DP-VI update steps
to turn it into a differentially-private inference algorithm. The larger `dp_scale` is, the greater the privacy
guarantees (expressed by parameters (ε,δ)) are.

Note: The final scale of the noise applied by the Gaussian mechanism is `dp_scale`·`clipping_threshold`, to properly
account for the maximum size of the gradient norm.

_How to select_: `d3p` offers the `d3p.dputil.approximate_sigma` function to find the `dp_scale` parameter for
a choice of ε and δ (and the minibatch size) using the Fourier accountant. Unfortunately, research on appropriate values for ε and δ is still ongoing, but common practice is to select δ to be much smaller than 1/N, where N is the size of your data set, and ε not larger than one.

#### `rng_suite`
JAX's default pseudo-random number generator is not known to be cryptographically secure, which is required for meaningful
privacy in practice. `d3p` therefore relies on our `jax-chacha-prng` cryptographically secure PRNG package through `d3p.random` as the default PRNG for `DPSVI` to sample noise related to privacy.

However, you if you want to use JAX's default PRNG, which is a bit faster, during debugging, you can use
the `rng_suite` parameter to pass the `d3p.random.debug` module, which is a slim wrapper around JAX's PRNG.

Note: The choice for `rng_suite` does not affect definition of the model (or variational guide)
or any purely modelling and sampling related functionality of NumPyro/JAX. These still all use JAX's default PRNG.

### Minibatch sampling
DP-VI relies on uniform sampling of random minibatches for each update step to guarantee privacy. This is different
from simply shuffling the data set and taking consecutive batches from it, as is often done in machine learning.

`d3p` implements a high-performant GPU-optimised minibatch sampling routine in `d3p.minibatch.subsample_batchify_data`.
It accepts the data set and a minibatch size (or, equivalently, a subsampling ratio) and returns functions
`batchify_init` and `batchify_sample` which initialise a batchifier state from a `rng_key` and sample a random minibatch given the batchifier state respectively.

### Requirements for Model Implementation
`d3p.svi.DPSVI` only requires the `model` function to be properly setup for minibatches of independent data. This
means users must use the `plate` primitive of NumPyro in their model implementation to properly annotate individual
data points as stochastically independent. If this is not properly done, the relative contribution of the prior
distributions for parameters will be overemphasized during the inference.

### Short Example

As a very brief example, we consider logistic regression and define the model as:
```python
import jax.numpy as jnp
import jax

from numpyro.infer import Trace_ELBO
from numpyro.optim import Adam
from numpyro.primitives import sample, plate
from numpyro.distributions import Normal, Bernoulli
from numpyro.infer.autoguide import AutoDiagonalNormal

from d3p.minibatch import subsample_batchify_data
from d3p.svi import DPSVI
from d3p.dputil import approximate_sigma
import d3p.random

def sigmoid(x):
    return 1./(1+jnp.exp(-x))

# specifies the model p(ys, w | xs) using NumPyro features
def model(xs, ys, N):
    # obtain data dimensions
    batch_size, d = xs.shape
    # the prior for w
    w = sample('w', Normal(0, 4),sample_shape=(d,))
    # distribution of label y for each record x
    with plate('batch', N, batch_size):
        theta = sigmoid(xs.dot(w))
        sample('ys', Bernoulli(theta), obs=ys)

guide = AutoDiagonalNormal(model) # variational guide approximates posterior of theta and w with independent Gaussians

def infer(data, labels, batch_size, num_iter, epsilon, delta, seed):
    rng_key = d3p.random.PRNGKey(seed)
    # set up minibatch sampling
    batchifier_init, get_batch = subsample_batchify_data((data, labels), batch_size)
    _, batchifier_state = batchifier_init(rng_key)

    # set up DP-VI algorithm
    q = batch_size / len(data)
    dp_scale, _, _ = approximate_sigma(epsilon, delta, q, num_iter)
    loss = Trace_ELBO()
    optimiser = Adam(1e-3)
    clipping_threshold = 10.
    dpsvi = DPSVI(model, guide, optimiser, loss, clipping_threshold, dp_scale, N=len(data))
    svi_state = dpsvi.init(rng_key, *get_batch(0, batchifier_state))

    # run inference
    def step_function(i, svi_state):
        data_batch, label_batch = get_batch(i, batchifier_state)
        svi_state, loss = dpsvi.update(svi_state, data_batch, label_batch)
        return svi_state

    svi_state = jax.lax.fori_loop(0, num_iter, step_function, svi_state)
    return dpsvi.get_params(svi_state)
```

See the `examples/` for more examples.

## Installing

d3p is pure Python software. You can install it via the pip command line tool
```
pip install d3p
```

This will install d3p with all required dependencies (NumPyro, JAX, ..) for CPU usage.


You can also install the latest development version
via pip with the following command:
```
pip install git+https://github.com/DPBayes/d3p.git@master#egg=d3p
```

Alternatively, you can clone this git repository and install with pip locally:
```
git clone https://github.com/DPBayes/d3p
cd d3p
pip install .
```

If you want to run the included examples, use the `examples` installation target:
```
pip install .[examples]
```

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
d3p supports hardware acceleration on CUDA devices. You will need to make sure that
you have the CUDA libraries set up on your system as well as a working compiler for C++.

You can then install d3p using
```
pip install d3p[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Unfortunately, the `jax-chacha-prng` package which provides the secure randomness
generator d3p relies on does not provide prebuilt packages for CUDA, so will need to
reinstall it from sources. To do so, issue the following command after the previous one:
```
pip install --force-reinstall --no-binary jax-chacha-prng "jax-chacha-prng<2"
```

### TPU installation
TPUs are currently not supported.

## Versioning Policy

The `master` branch contains the latest development version of d3p which may introduce breaking changes.

Version numbers adhere to [Semantic Versioning](https://semver.org/). Changes between releases are tracked in `ChangeLog.txt`.

## License

d3p is licensed under the Apache 2.0 License. The full license text is available
in `LICENSES/Apache-2.0.txt`. Copyright holder of each contribution are the respective
contributing developer or his or her assignee (i.e., universities or companies
owning the copyright of software contributed by their employees).

## Acknowledgements

We thank the NVIDIA AI Technology Center Finland for their contribution of GPU performance benchmarking and optimisation.

## References and Citing

When using d3p, please cite the following papers:

1. L. Prediger, N. Loppi, S. Kaski, A. Honkela.
d3p - A Python Package for Differentially-Private Probabilistic Programming
In *Proceedings on Privacy Enhancing Technologies, 2022(2)*.
Link: [https://doi.org/10.2478/popets-2022-0052](https://doi.org/10.2478/popets-2022-0052)

1. J. Jälkö, O. Dikmen, A. Honkela. Differentially Private Variational Inference for Non-conjugate Models
In *Uncertainty in Artificial Intelligence 2017 Proceedings of the 33rd Conference, UAI 2017*.
Link: [http://auai.org/uai2017/proceedings/papers/152.pdf](http://auai.org/uai2017/proceedings/papers/152.pdf)
