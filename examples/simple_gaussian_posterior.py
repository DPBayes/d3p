"""Fitting a simple gaussian posterior

"""

import os

# allow example to find dppp without installing
import sys
sys.path.append(os.path.dirname(sys.path[0]))
#### 

import argparse
import time

import jax
import jax.numpy as np
from jax import jit, lax, random
from jax.random import PRNGKey

import numpyro.distributions as dist
from numpyro.primitives import param, sample
from numpyro.infer import ELBO
import numpyro.optim as optimizers

from dppp.svi import DPSVI
from dppp.minibatch import minibatch, split_batchify_data, subsample_batchify_data
from dppp.util import unvectorize_shape_2d
from dppp.modelling import sample_prior_predictive

def model(N, d, num_obs_total=None):
    """Defines the generative probabilistic model: p(x|z)p(z)
    """
    assert(N is not None)
    assert(d is not None)

    z_mu = sample('mu', dist.Normal(np.zeros((d,)), 1.))
    x_var = .1
    with minibatch(N, num_obs_total):
        x = sample('obs', dist.Normal(z_mu, x_var), sample_shape=(N,))
    return x

def map_model_args(obs, num_obs_total=None):
    """Maps arguments from batch model likelihood call to model function."""
    assert(np.ndim(obs) <= 2)
    N, d = unvectorize_shape_2d(obs)
    return (N,d), {'num_obs_total': num_obs_total}, {'obs': obs}

def guide(d):
    """Defines the probabilistic guide for z (variational approximation to posterior): q(z) ~ p(z|x)
    """
    # guide starts from prior for mu
    assert(d is not None)
    mu_loc = param('mu_loc', np.zeros(d))
    mu_std = np.exp(param('mu_std_log', np.zeros(d)))

    z_mu = sample('mu', dist.Normal(mu_loc, mu_std))
    return z_mu, mu_loc, mu_std

def map_guide_args(obs, num_obs_total):
    """Maps arguments from batch guide call to guide function."""
    assert(np.ndim(obs) <= 2)
    _, d = unvectorize_shape_2d(obs)
    return (d,), {}, {}

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

def create_toy_data(rng_key, N, d):
    ## Create some toy data
    mu_true = np.ones(d)
    samples = sample_prior_predictive(rng_key, model, (2*N, d), {'mu': mu_true})
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

    # note(lumip): value for c currently completely made up
    #   value for dp_scale completely made up currently.
    svi = DPSVI(
        model, guide, optimizer, ELBO(), 
        dp_scale=0.01, clipping_threshold=20.,
        num_obs_total=args.num_samples,
        map_model_args_fn=map_model_args,
        map_guide_args_fn=map_guide_args
    )

    rng, svi_init_rng, batchifier_rng = random.split(rng, 3)
    _, batchifier_state = train_init(rng_key=batchifier_rng)
    batch = train_fetch(0, batchifier_state)
    svi_state = svi.init(svi_init_rng, *batch)

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
    mu_std = np.exp(params['mu_std_log'])
    print("### expected: {}".format(mu_true))
    print("### svi result\nmu_loc: {}\nerror: {}\nmu_std: {}".format(mu_loc, np.linalg.norm(mu_loc-mu_true), mu_std))
    mu_loc, mu_std = analytical_solution(X_train)
    print("### analytical solution\nmu_loc: {}\nerror: {}\nmu_std: {}".format(mu_loc, np.linalg.norm(mu_loc-mu_true), mu_std))
    mu_loc, mu_std = ml_estimate(X_train)
    print("### ml estimate\nmu_loc: {}\nerror: {}\nmu_std: {}".format(mu_loc, np.linalg.norm(mu_loc-mu_true), mu_std))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=100, type=int, help='number of training epochs')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('-batch-size', default=100, type=int, help='batch size')
    parser.add_argument('-d', '--dimensions', default=4, type=int, help='data dimension')
    parser.add_argument('-N', '--num-samples', default=10000, type=int, help='data samples count')
    args = parser.parse_args()
    main(args)
