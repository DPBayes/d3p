# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2019- d3p Developers and their Assignees

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gaussian mixture model example

This example demonstrates inferring a Gaussian mixture model.
"""

import logging
logging.getLogger().setLevel('INFO')

import os

# allow example to find d3p without installing
import sys
sys.path.append(os.path.dirname(sys.path[0]))
####

import argparse
import time

import jax
import jax.numpy as jnp
from jax import jit, lax

import numpyro
import numpyro.distributions as dist
import numpyro.optim as optimizers
from numpyro.primitives import sample, param, plate
from numpyro.infer import Trace_ELBO as ELBO

from d3p.svi import DPSVI
from d3p.modelling import sample_prior_predictive
from d3p.minibatch import split_batchify_data, poisson_batchify_data
from d3p.gmm import GaussianMixture
import d3p.random as rng_suite
from d3p.dputil import approximate_sigma_remove_relation

def model(k, obs=None, num_obs_total=None, d=None):
    # this is our model function using the GaussianMixture distribution
    # with prior belief
    if obs is not None:
        assert(jnp.ndim(obs) == 2)
        batch_size, d = jnp.shape(obs)
    else:
        assert(num_obs_total is not None)
        batch_size = num_obs_total
        assert(d is not None)
    num_obs_total = batch_size if num_obs_total is None else num_obs_total

    pis = sample('pis', dist.Dirichlet(jnp.ones(k)))
    mus = sample('mus', dist.Normal(jnp.zeros((k, d)), 10.))
    sigs = sample('sigs', dist.InverseGamma(1., 1.), sample_shape=jnp.shape(mus))
    with plate('batch', num_obs_total, batch_size):
        return sample('obs', GaussianMixture(mus, sigs, pis), obs=obs, sample_shape=(batch_size,))

def guide(k, obs=None, num_obs_total=None, d=None):
    # the latent MixGaus distribution which learns the parameters
    if obs is not None:
        assert(jnp.ndim(obs) == 2)
        _, d = jnp.shape(obs)
    else:
        assert(num_obs_total is not None)
        assert(d is not None)

    alpha_log = param('alpha_log', jnp.zeros(k))
    alpha = jnp.exp(alpha_log)
    pis = sample('pis', dist.Dirichlet(alpha))

    mus_loc = param('mus_loc', jnp.zeros((k, d)))
    mus = sample('mus', dist.Normal(mus_loc, 1.))
    sigs = sample('sigs', dist.InverseGamma(1., 1.), sample_shape=jnp.shape(mus))
    return pis, mus, sigs

def create_toy_data(rng_key, N, d):
    """Creates some toy data (for training and testing)"""
    # To spice things up, it is imbalanced:
    # The last component has twice as many samples as the others.
    mus = jnp.array([-10. * jnp.ones(d), 10. * jnp.ones(d), -2. * jnp.ones(d)])
    sigs = jnp.reshape(jnp.array([0.1, 1., 0.1]), (3,1))
    pis = jnp.array([1/4, 1/4, 2/4])

    samples = sample_prior_predictive(rng_key, model, (3, None, 2*N, d), substitutes={
        'pis': pis, 'mus': mus, 'sigs': sigs
    }, with_intermediates=True)

    X = samples['obs'][0]
    z = samples['obs'][1][0]

    z_train = z[:N]
    X_train = X[:N]
    z_test = z[N:]
    X_test = X[N:]

    latent_vals = (z_train, z_test, mus, sigs)
    return X_train, X_test, latent_vals

## the following two functions are not relevant to the training but will
#   assign test data to the learned posterior components of the model to
#   check the quality of the learned model
def compute_assignment_log_posterior(k, obs, mus, sigs, pis_prior):
    """computes the unnormalized log-posterior for each value of assignment z
       for each data point
    """
    N = jnp.atleast_1d(obs).shape[0]

    def per_component_fun(j):
        log_prob_x_zj = jnp.sum(dist.Normal(mus[j], sigs[j]).log_prob(obs), axis=1).flatten()
        assert(jnp.atleast_1d(log_prob_x_zj).shape == (N,))
        log_prob_zj = dist.Categorical(pis_prior).log_prob(j)
        log_prob = log_prob_x_zj + log_prob_zj
        assert(jnp.atleast_1d(log_prob).shape == (N,))
        return log_prob

    z_log_post = jax.vmap(per_component_fun)(jnp.arange(k))
    return z_log_post.T

def compute_assignment_accuracy(
    X_test, original_assignment, original_modes, posterior_modes, posterior_pis):
    """computes the accuracy score for attributing data to the mixture
    components based on the learned model
    """
    k, d = jnp.shape(original_modes)
    # we first map our true modes to the ones learned in the model using the
    # log posterior for z
    mode_assignment_posterior = compute_assignment_log_posterior(
        k, original_modes, posterior_modes, jnp.ones((k, d)), posterior_pis
    )
    mode_map = jnp.argmax(mode_assignment_posterior, axis=1)._value
    # a potential problem could be that mode_map might not be bijective, skewing
    # the results of the mapping. we build the inverse map and use identity
    # mapping as a base to counter that
    inv_mode_map = {j:j for j in range(k)}
    inv_mode_map.update({mode_map[j]:j for j in range(k)})

    # we next obtain the assignments for the data according to the model and
    # pass them through the inverse map we just build
    post_data_assignment = compute_assignment_log_posterior(
        k, X_test, posterior_modes, jnp.ones((k, d)), posterior_pis
    )
    post_data_assignment = jnp.argmax(post_data_assignment, axis=1)
    remapped_data_assignment = jnp.array(
        [inv_mode_map[j] for j in post_data_assignment._value]
    )

    # finally, we can compare the results with the original assigments and compute
    # the accuracy
    acc = jnp.mean(original_assignment == remapped_data_assignment)
    return acc


## main function: inference setup and main loop as well as subsequent
#   model quality check
def main(args):
    N = args.num_samples
    k = args.num_components
    d = args.dimensions

    toy_data_rng = jax.random.PRNGKey(1234)
    q = args.batch_size / N

    X_train, X_test, latent_vals = create_toy_data(toy_data_rng, N, d)
    train_init, train_fetch = poisson_batchify_data((X_train,), q=q, max_batch_size=.99)
    test_init, test_fetch = split_batchify_data((X_test,), batch_size=args.batch_size)

    dpsvi_rng = rng_suite.PRNGKey(0)
    dpsvi_rng, svi_init_rng, fetch_rng = rng_suite.split(dpsvi_rng, 3)
    iters_per_batch, batchifier_state = train_init(fetch_rng)

    ## Init optimizer and training algorithms
    optimizer = optimizers.Adam(args.learning_rate)

    # note(lumip): fix the parameters in the models
    def fix_params(model_fn, k):
        def fixed_params_fn(obs, **kwargs):
            return model_fn(k, obs, **kwargs)
        return fixed_params_fn

    model_fixed = fix_params(model, k)
    guide_fixed = fix_params(guide, k)

    delta = 1 / N**2
    dp_scale, _, _ = approximate_sigma_remove_relation(args.epsilon, delta, q, num_iter=iters_per_batch * args.num_epochs)
    print(f"dp_scale={dp_scale}")

    svi = DPSVI(
        model_fixed, guide_fixed, optimizer, ELBO(),
        dp_scale=dp_scale,  clipping_threshold=20., num_obs_total=args.num_samples
    )

    batch, _ = train_fetch(0, batchifier_state)
    svi_state = svi.init(svi_init_rng, *batch)

    @jit
    def epoch_train(svi_state, batchifier_state, num_batch):
        def body_fn(i, val):
            svi_state, loss = val
            batch, mask = train_fetch(i, batchifier_state)
            svi_state, batch_loss = svi.update(
                svi_state, *batch, mask=mask
            )
            loss += batch_loss / (args.num_samples * num_batch)
            return svi_state, loss

        return lax.fori_loop(0, num_batch, body_fn, (svi_state, 0.))

    @jit
    def eval_test(svi_state, batchifier_state, num_batch):
        def body_fn(i, loss_sum):
            batch = test_fetch(i, batchifier_state)
            loss = svi.evaluate(svi_state, *batch)
            loss_sum += loss / (args.num_samples * num_batch)
            return loss_sum

        return lax.fori_loop(0, num_batch, body_fn, 0.)

	## Train model
    for i in range(args.num_epochs):
        t_start = time.time()
        dpsvi_rng, data_fetch_rng = rng_suite.split(dpsvi_rng, 2)

        num_train_batches, train_batchifier_state = train_init(rng_key=data_fetch_rng)
        svi_state, train_loss = epoch_train(
            svi_state, train_batchifier_state, num_train_batches
        )
        train_loss.block_until_ready()
        t_end = time.time()

        if i % 100 == 0:
            dpsvi_rng, test_fetch_rng = rng_suite.split(dpsvi_rng, 2)
            num_test_batches, test_batchifier_state = test_init(rng_key=test_fetch_rng)
            test_loss = eval_test(
                svi_state, test_batchifier_state, num_test_batches
            )

            print("Epoch {}: loss = {} (on training set = {}) ({:.2f} s.)".format(
                    i, test_loss, train_loss, t_end - t_start
                ))

    params = svi.get_params(svi_state)
    print(params)
    posterior_modes = params['mus_loc']
    posterior_pis = dist.Dirichlet(jnp.exp(params['alpha_log'])).mean
    print("MAP estimate of mixture weights: {}".format(posterior_pis))
    print("MAP estimate of mixture modes  : {}".format(posterior_modes))

    acc = compute_assignment_accuracy(
        X_test, latent_vals[1], latent_vals[2], posterior_modes, posterior_pis
    )
    print("assignment accuracy: {}".format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-e', '--epsilon', default=10., type=float, help='privacy parameter epsilon')
    parser.add_argument('-n', '--num-epochs', default=400, type=int, help='number of training epochs')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('-batch-size', default=32, type=int, help='batch size')
    parser.add_argument('-d', '--dimensions', default=2, type=int, help='data dimension')
    parser.add_argument('-N', '--num-samples', default=10000, type=int, help='data samples count')
    parser.add_argument('-k', '--num-components', default=3, type=int, help='number of components in the mixture model')
    args = parser.parse_args()
    main(args)
