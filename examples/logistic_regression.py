"""Logistic regression example from numpyro.

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
from numpyro.handlers import param, sample

from dppp.svi import per_example_elbo, svi

def model(batch_X, batch_y, **kwargs):
    """Defines the generative probabilistic model: p(x|z)p(z)

    The model is conditioned on the observed data
    :param batch_X: a batch of predictors
    :param batch_y: a batch of observations
    :param other keyword arguments: are accepted but ignored
    """
    z_dim = batch_X.shape[1]
    z_w = sample('w', dist.Normal(np.zeros((z_dim,)), np.ones((z_dim,)))) # prior is N(0,I)
    z_intercept = sample('intercept', dist.Normal(0,1)) # prior is N(0,1)
    logits = batch_X.dot(z_w)+z_intercept
    return sample('obs', dist.Bernoulli(logits=logits), obs=batch_y)

def guide(batch_X, batch_y, z_dim):
    """Defines the probabilistic guide for z (variational approximation to posterior): q(z) ~ p(z|x)
    """
    # note(lumip): we are interested in the posterior of w and intercept
    #   so.. like this then?
    z_w_loc = param("w_loc", np.zeros((z_dim,)))
    z_w_std = np.exp(param("w_std_log", np.zeros((z_dim,))))
    z_w = sample('w', dist.Normal(z_w_loc, z_w_std))
    z_intercept_loc = param("intercept_loc", 0.)
    z_interpet_std = np.exp(param("intercept_std_log", 0.))
    z_intercept = sample('intercept', dist.Normal(z_intercept_loc, z_interpet_std))
    return (z_w, z_intercept)

def create_toy_data(N, d):
    ## Create some toy data
    onp.random.seed(123)

    w_true = onp.random.randn(d)
    intercept_true = onp.random.randn()

    X = np.array(onp.random.randn(N, d))

    logits_true = X.dot(w_true)+intercept_true
    sigmoid = lambda x : 1./(1.+onp.exp(-x)) # todo(lumip): also defined in other example. move it to utils?
    y = 1.*(sigmoid(logits_true)>onp.random.rand(N))
    return X, y, w_true, intercept_true

def batchify(X, y, batch_size):
    # note(lumip): almost identical to the load_dataset routine in datasets
    # todo(lumip): put common ground in some nice method
    arrays = (X, y)
    num_records = len(arrays[0])
    idxs = np.arange(num_records)
    if not batch_size:
        batch_size = num_records

    def init(rng=None):
        return num_records // batch_size, jax.random.shuffle(rng, idxs) if rng is not None else idxs

    def get_batch(i=0, idxs=idxs):
        ret_idx = lax.dynamic_slice_in_dim(idxs, i * batch_size, batch_size)
        return tuple(lax.index_take(a, (ret_idx,), axes=(0,)) if isinstance(a, jax.interpreters.xla.DeviceArray)
                     else np.take(a, ret_idx, axis=0) for a in arrays)

    return init, get_batch

def main(args):
    X, y, w_true, intercept_true = create_toy_data(args.num_samples, args.dimensions)
    train_init, train_fetch = batchify(X, y, args.batch_size)

    ## Init optimizer and training algorithms
    opt_init, opt_update, get_params = optimizers.adam(args.learning_rate)

    per_example_loss = per_example_elbo
    combined_loss = np.sum
    svi_init, svi_update, svi_eval = svi(
        model, guide, per_example_loss, combined_loss, opt_init, opt_update, 
        get_params, per_example_variables={'obs'}, z_dim=args.dimensions
    )

    svi_update = jit(svi_update)

    rng = PRNGKey(123)

    rng, svi_init_rng, data_fetch_rng = random.split(rng, 3)
    _, train_idx = train_init(rng=data_fetch_rng)
    batch_X, batch_Y = train_fetch(0, train_idx)
    opt_state = svi_init(svi_init_rng, (batch_X, batch_Y), (batch_X, batch_Y))

    @jit
    def epoch_train(opt_state, rng, data_idx, num_batch):
        def body_fn(i, val):
            loss_sum, opt_state, rng = val
            rng, update_rng = random.split(rng, 2)
            batch = train_fetch(i, data_idx)
            loss, opt_state, rng = svi_update(
                i, opt_state, update_rng, batch, batch,
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

	## Train model
    for i in range(args.num_epochs):
        t_start = time.time()
        rng, data_fetch_rng, test_rng = random.split(rng, 3)

        num_train, train_idx = train_init(rng=data_fetch_rng)
        _, opt_state, rng = epoch_train(
            opt_state, rng, train_idx, num_train
        )

        # computing loss over training data (for now?)
        test_loss = eval_test(
            opt_state, test_rng, train_idx, num_train
        )

        if (i % 100) == 0:
            print("Epoch {}: loss = {} ({:.2f} s.)".format(
                i, test_loss, time.time() - t_start
            ))
            params = get_params(opt_state)
            print("w_loc: {}".format(params['w_loc']))
            print("w_std: {}".format(np.exp(params['w_std_log'])))

    params = get_params(opt_state)
    print("\tw_loc: {}\n\texpected: {}\n\terror: {}".format(params['w_loc'], w_true, np.linalg.norm(params['w_loc']-w_true)))
    print("\tw_std: {}".format(np.exp(params['w_std_log'])))
    print("\tintercept_loc: {}\n\texpected: {}\n\terror: {}".format(params['intercept_loc'], intercept_true, np.abs(params['intercept_loc']-intercept_true)))
    print("\tintercept_std: {}".format(np.exp(params['intercept_std_log'])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int, help='number of training epochs')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('-batch-size', default=128, type=int, help='batch size')
    parser.add_argument('-d', '--dimensions', default=10, type=int, help='data dimension')
    parser.add_argument('-N', '--num-samples', default=10000, type=int, help='data samples count')
    args = parser.parse_args()
    main(args)
