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

"""Logistic regression example.
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
import d3p.random

import numpyro
import numpyro.distributions as dist
from numpyro.primitives import param, sample, plate
from numpyro.infer import Trace_ELBO as ELBO
import numpyro.optim as optimizers

from d3p.util import example_count, normalize
from d3p.svi import DPSVI
from d3p.modelling import sample_prior_predictive, sample_multi_prior_predictive, sample_multi_posterior_predictive
from d3p.minibatch import split_batchify_data, subsample_batchify_data

def model(batch_X, batch_y=None, num_obs_total=None):
    """Defines the generative probabilistic model: p(y|z,X)p(z)

    The model is conditioned on the observed data
    :param batch_X: a batch of predictors
    :param batch_y: a batch of observations
    """
    assert(jnp.ndim(batch_X) == 2)
    batch_size, d = jnp.shape(batch_X)
    num_obs_total = batch_size if num_obs_total is None else num_obs_total
    assert(batch_y is None or example_count(batch_y) == batch_size)

    z_w = sample('w', dist.Normal(jnp.zeros((d,)), jnp.ones((d,)))) # prior is N(0,I)
    z_intercept = sample('intercept', dist.Normal(0,1)) # prior is N(0,1)
    logits = batch_X.dot(z_w)+z_intercept

    with plate("batch", num_obs_total, batch_size):
        return sample('obs', dist.Bernoulli(logits=logits), obs=batch_y)


def guide(batch_X, batch_y=None, num_obs_total=None):
    """Defines the probabilistic guide for z (variational approximation to posterior): q(z) ~ p(z|x)
    """
    # we are interested in the posterior of w and intercept
    # since this is a fairly simple model, we just initialize them according
    # to our prior believe and let the optimization handle the rest
    assert(jnp.ndim(batch_X) == 2)
    d = jnp.shape(batch_X)[1]

    z_w_loc = param("w_loc", jnp.zeros((d,)))
    z_w_std = jnp.exp(param("w_std_log", jnp.zeros((d,))))
    z_w = sample('w', dist.Normal(z_w_loc, z_w_std))

    z_intercept_loc = param("intercept_loc", 0.)
    z_interpet_std = jnp.exp(param("intercept_std_log", 0.))
    z_intercept = sample('intercept', dist.Normal(z_intercept_loc, z_interpet_std))

    return (z_w, z_intercept)

def create_toy_data(rng_key, N, d):
    ## Create some toy data
    X_rng_key, prior_pred_rng_key = jax.random.split(rng_key)

    X = jax.random.normal(X_rng_key, shape=(2*N, d))

    sampled_data = sample_prior_predictive(prior_pred_rng_key, model, (X,))
    y = sampled_data['obs']
    w_true = sampled_data['w']
    intercept_true = sampled_data['intercept']

    X_train = X[:N]
    y_train = y[:N]
    X_test = X[N:]
    y_test = y[N:]

    return (X_train, y_train), (X_test, y_test), (w_true, intercept_true)

def estimate_accuracy_fixed_params(X, y, w, intercept, rng, num_iterations=1):
    samples = sample_multi_prior_predictive(rng, num_iterations, model, (X,), {'w': w, 'intercept': intercept})
    return jnp.average(samples['obs'] == y)

def estimate_accuracy(X, y, params, rng, num_iterations=1):

    samples = sample_multi_posterior_predictive(
        rng, num_iterations, model, (X,), guide, (X,), params
    )

    return jnp.average(samples['obs'] == y)

def main(args):
    rng = jax.random.PRNGKey(123)

    rng, toy_data_rng = jax.random.split(rng, 2)
    train_data, test_data, true_params = create_toy_data(
        toy_data_rng, args.num_samples, args.dimensions
    )

    train_init, train_fetch = subsample_batchify_data(train_data, batch_size=args.batch_size)
    test_init, test_fetch = split_batchify_data(test_data, batch_size=args.batch_size)

    ## Init optimizer and training algorithms
    optimizer = optimizers.Adam(args.learning_rate)

    svi = DPSVI(model, guide, optimizer, ELBO(),
        dp_scale=0.01, clipping_threshold=20., num_obs_total=args.num_samples
    )

    dpsvi_rng = d3p.random.PRNGKey(0)
    dpsvi_rng, svi_init_rng, data_fetch_rng = d3p.random.split(dpsvi_rng, 3)
    _, batchifier_state = train_init(rng_key=data_fetch_rng)
    sample_batch = train_fetch(0, batchifier_state)
    svi_state = svi.init(svi_init_rng, *sample_batch)

    @jit
    def epoch_train(svi_state, batchifier_state, num_batch):
        def body_fn(i, val):
            svi_state, loss = val
            batch = train_fetch(i, batchifier_state)
            batch_X, batch_Y = batch

            svi_state, batch_loss = svi.update(svi_state, batch_X, batch_Y)
            loss += batch_loss / (args.num_samples * num_batch)
            return svi_state, loss

        return lax.fori_loop(0, num_batch, body_fn, (svi_state, 0.))

    @jit
    def eval_test(svi_state, batchifier_state, num_batch, rng):
        params = svi.get_params(svi_state)

        def body_fn(i, val):
            loss_sum, acc_sum = val

            batch = test_fetch(i, batchifier_state)
            batch_X, batch_Y = batch

            loss = svi.evaluate(svi_state, batch_X, batch_Y)
            loss_sum += loss / (args.num_samples * num_batch)

            acc_rng = jax.random.fold_in(rng, i)
            acc = estimate_accuracy(batch_X, batch_Y, params, acc_rng, 1)
            acc_sum += acc / num_batch

            return loss_sum, acc_sum

        return lax.fori_loop(0, num_batch, body_fn, (0., 0.))

	## Train model
    for i in range(args.num_epochs):
        t_start = time.time()
        dpsvi_rng, data_fetch_rng = d3p.random.split(dpsvi_rng, 2)

        num_train_batches, train_batchifier_state = train_init(rng_key=data_fetch_rng)
        svi_state, train_loss = epoch_train(
            svi_state, train_batchifier_state, num_train_batches
        )
        train_loss.block_until_ready()
        t_end = time.time()

        if (i % (args.num_epochs//10)) == 0:
            dpsvi_rng, test_rng, test_fetch_rng = d3p.random.split(dpsvi_rng, 3)
            test_rng = d3p.random.convert_to_jax_rng_key(test_rng)
            num_test_batches, test_batchifier_state = test_init(rng_key=test_fetch_rng)
            test_loss, test_acc = eval_test(
                svi_state, test_batchifier_state, num_test_batches, test_rng
            )
            print("Epoch {}: loss = {}, acc = {} (loss on training set: {}) ({:.2f} s.)".format(
                i, test_loss, test_acc, train_loss, t_end - t_start
            ))

    # parameters for logistic regression may be scaled arbitrarily. normalize
    #   w (and scale intercept accordingly) for comparison
    w_true = normalize(true_params[0])
    scale_true = jnp.linalg.norm(true_params[0])
    intercept_true = true_params[1] / scale_true

    params = svi.get_params(svi_state)
    w_post = normalize(params['w_loc'])
    scale_post = jnp.linalg.norm(params['w_loc'])
    intercept_post = params['intercept_loc'] / scale_post

    print("w_loc: {}\nexpected: {}\nerror: {}".format(w_post, w_true, jnp.linalg.norm(w_post-w_true)))
    print("w_std: {}".format(jnp.exp(params['w_std_log'])))
    print("")
    print("intercept_loc: {}\nexpected: {}\nerror: {}".format(intercept_post, intercept_true, jnp.abs(intercept_post-intercept_true)))
    print("intercept_std: {}".format(jnp.exp(params['intercept_std_log'])))
    print("")

    X_test, y_test = test_data
    rng, rng_acc_true, rng_acc_post = jax.random.split(rng, 3)
    # for evaluation accuracy with true parameters, we scale them to the same
    #   scale as the found posterior. (gives better results than normalized
    #   parameters (probably due to numerical instabilities))
    acc_true = estimate_accuracy_fixed_params(X_test, y_test, w_true, intercept_true, rng_acc_true, 10)
    acc_post = estimate_accuracy(X_test, y_test, params, rng_acc_post, 10)

    print("avg accuracy on test set:  with true parameters: {} ; with found posterior: {}\n".format(acc_true, acc_post))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=600, type=int, help='number of training epochs')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('-batch-size', default=200, type=int, help='batch size')
    parser.add_argument('-d', '--dimensions', default=4, type=int, help='data dimension')
    parser.add_argument('-N', '--num-samples', default=10000, type=int, help='data samples count')
    args = parser.parse_args()
    main(args)
