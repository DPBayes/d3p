"""Gaussian mixture model example 2.

This example also demonstrates the gaussian mixture model but instead
of composing it from primitive distributions in the model and guide
respectively, we define a MixGaus distribution class to compute the probability
function for a Gaussian mixture, which we then directly use in the model.

note(lumip): Currently only supports 2 mixture components and infers fewer
priors than the other example.
"""

import os

# allow example to find dppp without installing
import sys
sys.path.append(os.path.dirname(sys.path[0]))
#### 

import argparse
import time

import matplotlib.pyplot as plt
import numpy as onp

import jax.numpy as np
from jax import jit, lax, random
from jax.experimental import optimizers, stax
from jax.random import PRNGKey
import jax

import numpyro.distributions as dist
from numpyro.handlers import param, sample, seed, trace, substitute

from dppp.svi import per_example_elbo, dpsvi

from jax.scipy.special import logsumexp
from numpyro.distributions.distribution import Distribution, TransformedDistribution
from numpyro.distributions import constraints
from numpyro.distributions.util import (
    cumsum,
    matrix_to_tril_vec,
    multigammaln,
    promote_shapes,
    signed_stick_breaking_tril,
    standard_gamma,
    vec_to_tril_matrix
)

from datasets import batchify_data
from example_util import softmax

class MixGaus(Distribution):
    arg_constraints = {'locs': constraints.real, 'scales': constraints.positive, 'pis' : constraints.simplex}
    support = constraints.real

    def __init__(self, locs=0., scales=1., pis=1.0, validate_args=None):
        self.locs, self.scales, self.pis = promote_shapes(locs, scales, pis)
        batch_shape = lax.broadcast_shapes(np.shape(locs), np.shape(scales), np.shape(pis))
        super(MixGaus, self).__init__(batch_shape=batch_shape, validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_pis = np.log(self.pis)
        log_phis = np.array([dist.Normal(loc, scale).log_prob(value).sum(-1) for loc, scale in zip(self.locs, self.scales)]).T
        return logsumexp(log_pis + log_phis, axis=-1)

    def sample(self, *args):
        # ignoring this for now as we only care to compute the log_probability
        return 1

def model(k, obs, _ks):
    assert(obs is not None)
    _, d = np.atleast_2d(obs).shape
    pis = sample('pis', dist.Dirichlet(np.ones(k)))
    mus = sample('mus', dist.Normal(np.zeros((k, d)), 10.))
    sigs = np.ones((k, d))
    return sample('obs', MixGaus(mus, sigs, pis), obs=obs)

def guide(k, obs, _ks):
    assert(obs is not None)
    _, d = np.atleast_2d(obs).shape
    mus_loc = param('mus_loc', np.zeros((k, d)))
    mus = sample('mus', dist.Normal(mus_loc, 1.))
    sigs = np.ones((k, d))
    alpha = param('alpha', np.array([1.,2.]))
    pis = sample('pis', dist.Dirichlet(alpha))
    return pis, mus, sigs

def create_toy_data(N, k, d):
    ## Create some toy data
    onp.random.seed(122)

    # We create some toy data. To spice things up, it is imbalanced:
    #   The last component has twice as many samples as the others.
    z = onp.random.randint(0, k+1, N)
    z[z == k] = k - 1
    X = onp.zeros((N, d))

    assert(k < 4)
    mus = [-10. * onp.ones(d), 10. * onp.ones(d), -2. * onp.ones(d)]
    sigs = [onp.sqrt(0.1), 1., onp.sqrt(0.1)]
    for i in range(k):
        N_i = onp.sum(z == i)
        X_i = mus[i] + sigs[i] * onp.random.randn(N_i, d)
        X[z == i] = X_i

    mus = np.array(mus)
    sigs = np.array(sigs)

    latent_vals = (z, mus, sigs)
    return X, latent_vals


def main(args):
    N = args.num_samples
    k = args.num_components
    k_gen = args.num_components_generated
    d = args.dimensions

    X, latent_vals = create_toy_data(N, k_gen, d)
    train_init, train_fetch = batchify_data((X,), args.batch_size)

    ## Init optimizer and training algorithms
    opt_init, opt_update, get_params = optimizers.adam(args.learning_rate)

    # note(lumip): fix the parameters in the models
    def fix_params(model_fn, k, z):
        def fixed_params_fn(obs):
            return model_fn(k, obs, z)
        return fixed_params_fn

    model_fixed = fix_params(model, k, latent_vals[0])
    guide_fixed = fix_params(guide, k, latent_vals[0])

    # note(lumip): value for c currently completely made up
    svi_init, svi_update, svi_eval = dpsvi(
        model_fixed, guide_fixed, per_example_elbo, opt_init,
        opt_update, get_params, clipping_threshold=20.,
        per_example_variables={'obs'}
    )

    svi_update = jit(svi_update)

    rng = PRNGKey(123)

    rng, svi_init_rng = random.split(rng, 2)
    batch = train_fetch(0)
    opt_state = svi_init(svi_init_rng, batch, batch)

    @jit
    def epoch_train(opt_state, rng, data_idx, num_batch):
        def body_fn(i, val):
            loss_sum, opt_state, rng = val
            rng, update_rng = random.split(rng, 2)
            batch = train_fetch(i, data_idx)
            loss, opt_state, rng = svi_update(
                i, opt_state, update_rng, batch, batch
            )
            loss_sum += loss / len(batch[0])
            return loss_sum, opt_state, rng

        loss, opt_state, rng = lax.fori_loop(0, num_batch, body_fn, (0., opt_state, rng))
        loss /= num_batch
        return loss, opt_state, rng

    
    @jit
    def eval_test(opt_state, rng, data_idx, num_batch):
        def body_fn(i, val):
            loss_sum, rng = val
            batch = train_fetch(i, data_idx)
            loss = svi_eval(opt_state, rng, batch, batch) / len(batch[0])
            loss_sum += loss
            return loss_sum, rng

        loss, _ = lax.fori_loop(0, num_batch, body_fn, (0., rng))
        loss = loss / num_batch
        return loss

    smoothed_loss_window = onp.empty(5)
    smoothed_loss_window.fill(onp.nan)
    window_idx = 0

	## Train model
    for i in range(args.num_epochs):
        t_start = time.time()
        rng, data_fetch_rng, test_rng = random.split(rng, 3)

        num_train, train_idx = train_init(rng=data_fetch_rng)
        _, opt_state, rng = epoch_train(
            opt_state, rng, train_idx, num_train
        )

        if i % 100 == 0:
            # computing loss over training data (for now?)
            test_loss = eval_test(
                opt_state, test_rng, train_idx, num_train
            )
            smoothed_loss_window[window_idx] = test_loss
            smoothed_loss = onp.nanmean(smoothed_loss_window)
            window_idx = (window_idx + 1) % 5

            print("Epoch {}: loss = {}, smoothed loss = {} ({:.2f} s.)".format(
                    i, test_loss, smoothed_loss, time.time() - t_start
                ))

    params = get_params(opt_state)
    print(params)
    print("MAP estimate of mixture weights: {}".format(dist.Dirichlet(params['alpha']).mean))
    print("MAP estimate of mixture modes  : {}".format(params['mus_loc']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=10000, type=int, help='number of training epochs')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('-batch-size', default=32, type=int, help='batch size')
    parser.add_argument('-d', '--dimensions', default=2, type=int, help='data dimension')
    parser.add_argument('-N', '--num-samples', default=2048, type=int, help='data samples count')
    parser.add_argument('-k', '--num-components', default=2, type=int, help='number of components in the mixture model')
    parser.add_argument('-K', '--num-components-generated', default=2, type=int, help='number of components in generated data')
    args = parser.parse_args()
    main(args)