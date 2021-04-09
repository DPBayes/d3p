# Copyright 2019- d3p Developers and their Assignees

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Fitting a simple gaussian posterior

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
from jax import jit, lax, random
from jax.random import PRNGKey

import numpyro
import numpyro.distributions as dist
from numpyro.primitives import param, sample, plate
from numpyro.infer import Trace_ELBO as ELBO
import numpyro.optim as optimizers

from d3p.svi import DPSVI
from d3p.minibatch import split_batchify_data, subsample_batchify_data
from d3p.modelling import sample_prior_predictive


def model(obs=None, num_obs_total=None, d=None):
    """Defines the generative probabilistic model: p(x|z)p(z)
    """
    if obs is not None:
        assert(jnp.ndim(obs) == 2)
        batch_size, d = jnp.shape(obs)
    else:
        assert(num_obs_total is not None)
        batch_size = num_obs_total
        assert(d != None)
    num_obs_total = batch_size if num_obs_total is None else num_obs_total

    z_mu = sample('mu', dist.Normal(jnp.zeros((d,)), 1.))
    x_var = .1
    with plate('batch', num_obs_total, batch_size):
        x = sample('obs', dist.Normal(z_mu, x_var).to_event(1), obs=obs, sample_shape=(batch_size,))
    return x

def guide(obs=None, num_obs_total=None, d=None):
    """Defines the probabilistic guide for z (variational approximation to posterior): q(z) ~ p(z|x)
    """
    # # very smart guide: starts with analytical solution
    # assert(obs != None)
    # mu_loc, mu_std = analytical_solution(obs)
    # mu_loc = param('mu_loc', mu_loc)
    # mu_std = jnp.exp(param('mu_std_log', jnp.log(mu_std)))

    # not so smart guide: starts from prior for mu
    assert(d != None)
    mu_loc = param('mu_loc', jnp.zeros(d))
    mu_std = jnp.exp(param('mu_std_log', jnp.zeros(d)))

    z_mu = sample('mu', dist.Normal(mu_loc, mu_std))
    return z_mu, mu_loc, mu_std

def analytical_solution(obs):
    N = jnp.atleast_1d(obs).shape[0]
    x_var = .1
    x_var_inv = 1/x_var
    mu_var = 1/(x_var_inv*N+1)
    mu_std = jnp.sqrt(mu_var)
    mu_loc = mu_var*jnp.sum(x_var_inv*obs, axis=0)

    return mu_loc, mu_std

def ml_estimate(obs):
    N = jnp.atleast_1d(obs).shape[0]
    mu_loc = (1/N)*jnp.sum(obs, axis=0)
    mu_var = 1/jnp.sqrt(N+1)
    mu_std = jnp.sqrt(mu_var)

    return mu_loc, mu_std

def create_toy_data(rng_key, N, d):
    ## Create some toy data
    mu_true = jnp.ones(d)
    samples = sample_prior_predictive(rng_key, model, (None, 2*N, d), {'mu': mu_true})
    X = samples['obs']

    X_train = X[:N]
    X_test = X[N:]

    return X_train, X_test, mu_true

def main(args):
    rng = PRNGKey(1234)
    rng, toy_data_rng = jax.random.split(rng, 2)
    X_train, X_test, mu_true = create_toy_data(toy_data_rng, args.num_samples, args.dimensions)

    train_init, train_fetch = subsample_batchify_data((X_train,), batch_size=args.batch_size)
    test_init, test_fetch = split_batchify_data((X_test,), batch_size=args.batch_size)

    ## Init optimizer and training algorithms
    optimizer = optimizers.Adam(args.learning_rate)

    svi = DPSVI(
        model, guide, optimizer, ELBO(),
        dp_scale=args.sigma, clipping_threshold=args.clip_threshold,
        d=args.dimensions, num_obs_total=args.num_samples
    )

    rng, svi_init_rng, batchifier_rng = random.split(rng, 3)
    _, batchifier_state = train_init(rng_key=batchifier_rng)
    batch = train_fetch(0, batchifier_state)
    svi_state = svi.init(svi_init_rng, *batch)

    q = args.batch_size/args.num_samples
    eps = svi.get_epsilon(args.delta, q, num_epochs=args.num_epochs)
    print("Privacy epsilon {} (for sigma: {}, delta: {}, C: {}, q: {})".format(
        eps, args.sigma, args.clip_threshold, args.delta, q
    ))

    @jit
    def epoch_train(svi_state, batchifier_state, num_batch):
        def body_fn(i, val):
            svi_state, loss = val
            batch = train_fetch(i, batchifier_state)
            svi_state, batch_loss = svi.update(
                svi_state, *batch
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
        rng, data_fetch_rng = random.split(rng, 2)

        num_train_batches, train_batchifier_state = train_init(rng_key=data_fetch_rng)
        svi_state, train_loss = epoch_train(
            svi_state, train_batchifier_state, num_train_batches
        )
        train_loss.block_until_ready() # todo: blocking on loss will probabyl ignore rest of optimization
        t_end = time.time()

        if (i % (args.num_epochs // 10) == 0):
            rng, test_fetch_rng = random.split(rng, 2)
            num_test_batches, test_batchifier_state = test_init(rng_key=test_fetch_rng)
            test_loss = eval_test(
                svi_state, test_batchifier_state, num_test_batches
            )

            print("Epoch {}: loss = {} (on training set: {}) ({:.2f} s.)".format(
                i, test_loss, train_loss, t_end - t_start
            ))

    params = svi.get_params(svi_state)
    mu_loc = params['mu_loc']
    mu_std = jnp.exp(params['mu_std_log'])
    print("### expected: {}".format(mu_true))
    print("### svi result\nmu_loc: {}\nerror: {}\nmu_std: {}".format(mu_loc, jnp.linalg.norm(mu_loc-mu_true), mu_std))
    mu_loc, mu_std = analytical_solution(X_train)
    print("### analytical solution\nmu_loc: {}\nerror: {}\nmu_std: {}".format(mu_loc, jnp.linalg.norm(mu_loc-mu_true), mu_std))
    mu_loc, mu_std = ml_estimate(X_train)
    print("### ml estimate\nmu_loc: {}\nerror: {}\nmu_std: {}".format(mu_loc, jnp.linalg.norm(mu_loc-mu_true), mu_std))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=100, type=int, help='number of training epochs')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('-batch-size', default=100, type=int, help='batch size')
    parser.add_argument('-d', '--dimensions', default=4, type=int, help='data dimension')
    parser.add_argument('-N', '--num-samples', default=10000, type=int, help='data samples count')
    parser.add_argument('--sigma', default=1., type=float, help='privacy scale')
    parser.add_argument('--delta', default=1/10000, type=float, help='privacy slack parameter delta')
    parser.add_argument('-C', '--clip-threshold', default=1., type=float, help='clipping threshold for gradients')
    args = parser.parse_args()
    main(args)
