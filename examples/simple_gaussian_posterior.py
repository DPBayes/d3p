"""Fitting a simple gaussian posterior

"""

import os

# allow example to find dppp without installing
import sys
sys.path.append(os.path.dirname(sys.path[0]))
#### 

import argparse
import time

import numpy as onp

import jax.numpy as np
from jax import jit, lax, random
from jax.experimental import optimizers, stax
from jax.random import PRNGKey
import jax

import numpyro.distributions as dist
from numpyro.handlers import sample, param, seed, substitute, trace
from numpyro.svi import elbo

from dppp.svi import dpsvi, minibatch, example_count

from datasets import batchify_data

def model(obs=None, num_obs_total=None, d=None):
    """Defines the generative probabilistic model: p(x|z)p(z)
    """
    # note(lumip): the following if construct is currently necessary because
    #   the per-example value_and_grad function uses vmap internally, applying
    #   model and guide to 1-example batches (and stripping the first dimension)
    #   this is not nice because it means that model/guide have to be adapted
    #   if they do the kind of checks as below..
    if obs is not None:
        if np.ndim(obs) == 2:
            B = example_count(obs)
            d = obs.shape[1]
        elif np.ndim(obs) == 1:
            B = 1
            d = obs.shape[0]
    else:
        assert(num_obs_total is not None)
        B = num_obs_total
        assert(d is not None)

    z_mu = sample('mu', dist.Normal(np.zeros((d,)), 1.))
    x_var = .1
    with minibatch(B, num_obs_total):
        x = sample('obs', dist.Normal(z_mu, x_var), obs=obs, sample_shape=(B,))
    return x

def guide(obs, num_obs_total=None, d=None):
    """Defines the probabilistic guide for z (variational approximation to posterior): q(z) ~ p(z|x)
    """
    # very smart guide: starts with analytical solution
    mu_loc, mu_std = analytical_solution(obs)
    mu_loc = param('mu_loc', mu_loc)
    mu_std = np.exp(param('mu_std_log', np.log(mu_std)))

    # # not so smart guide: starts from prior for mu
    # mu_loc = param('mu_loc', np.zeros(d))
    # mu_std = np.exp(param('mu_std_log', np.zeros(d)))

    z_mu = sample('mu', dist.Normal(mu_loc, mu_std))
    return z_mu, mu_loc, mu_std

@jit
def analytical_solution(obs):
    N = np.atleast_1d(obs).shape[0]
    x_var = .1
    x_var_inv = 1/x_var
    mu_var = 1/(x_var_inv*N+1)
    mu_std = np.sqrt(mu_var)
    mu_loc = mu_var*np.sum(x_var_inv*obs, axis=0)

    return mu_loc, mu_std

def ml_estimate(obs):
    N = np.atleast_1d(obs).shape[0]
    mu_loc = (1/N)*np.sum(obs, axis=0)
    mu_var = 1/np.sqrt(N+1)
    mu_std = np.sqrt(mu_var)

    return mu_loc, mu_std

def create_toy_data(N, d):
    ## Create some toy data
    mu_true = np.ones(d)
    X = substitute(seed(model, jax.random.PRNGKey(54795)), {'mu': mu_true})(
        num_obs_total=N, d=d
    )

    return X, mu_true

def main(args):
    X, mu_true = create_toy_data(args.num_samples, args.dimensions)
    train_init, train_fetch = batchify_data((X,), args.batch_size)

    ## Init optimizer and training algorithms
    opt_init, opt_update, get_params = optimizers.adam(args.learning_rate)

    # note(lumip): value for c currently completely made up
    svi_init, svi_update, svi_eval = dpsvi(
        model, guide, elbo, opt_init, opt_update, 
        get_params, num_obs_total=args.num_samples,
        clipping_threshold=20., per_example_variables={'obs'}
    )

    svi_update = jit(svi_update)

    rng = PRNGKey(1234)

    rng, svi_init_rng = random.split(rng, 2)
    _, train_idx = train_init()
    batch = train_fetch(0, train_idx)
    opt_state, _ = svi_init(svi_init_rng, batch, batch)

    @jit
    def epoch_train(rng, opt_state, data_idx, num_batch):
        def body_fn(i, val):
            opt_state, rng = val
            rng, update_rng = random.split(rng, 2)
            batch = train_fetch(i, data_idx)
            _, opt_state, rng = svi_update(
                i, update_rng, opt_state, batch, batch,
            )
            return opt_state, rng

        return lax.fori_loop(0, num_batch, body_fn, (opt_state, rng))

    @jit
    def eval_test(rng, opt_state, data_idx, num_batch):
        def body_fn(i, val):
            loss_sum, rng = val
            batch = train_fetch(i, data_idx)
            rng, eval_rng = jax.random.split(rng, 2)
            loss = svi_eval(eval_rng, opt_state, batch, batch) / args.num_samples
            loss_sum += loss

            return loss_sum, rng

        loss, _ = lax.fori_loop(0, num_batch, body_fn, (0., rng))
        loss /= num_batch
        return loss

	## Train model
    for i in range(args.num_epochs):
        t_start = time.time()
        rng, data_fetch_rng, test_rng = random.split(rng, 3)

        num_train, train_idx = train_init(rng=data_fetch_rng)
        opt_state, rng = epoch_train(
            rng, opt_state, train_idx, num_train
        )

        if (i % (args.num_epochs // 10) == 0):
            # computing loss over training data (for now?)
            test_loss = eval_test(
                test_rng, opt_state, train_idx, num_train
            )

            print("Epoch {}: loss = {} ({:.2f} s.)".format(
                i, test_loss, time.time() - t_start
            ))

    params = get_params(opt_state)
    mu_loc = params['mu_loc']
    mu_std = np.exp(params['mu_std_log'])
    print("### expected: {}".format(mu_true))
    print("### svi result\nmu_loc: {}\nerror: {}\nmu_std: {}".format(mu_loc, np.linalg.norm(mu_loc-mu_true), mu_std))
    mu_loc, mu_std = analytical_solution(X)
    print("### analytical solution\nmu_loc: {}\nerror: {}\nmu_std: {}".format(mu_loc, np.linalg.norm(mu_loc-mu_true), mu_std))
    mu_loc, mu_std = ml_estimate(X)
    print("### ml estimate\nmu_loc: {}\nerror: {}\nmu_std: {}".format(mu_loc, np.linalg.norm(mu_loc-mu_true), mu_std))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=400, type=int, help='number of training epochs')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('-batch-size', default=100, type=int, help='batch size')
    parser.add_argument('-d', '--dimensions', default=4, type=int, help='data dimension')
    parser.add_argument('-N', '--num-samples', default=10000, type=int, help='data samples count')
    args = parser.parse_args()
    main(args)
