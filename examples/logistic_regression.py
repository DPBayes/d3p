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

import jax.numpy as np
from jax import jit, lax, random
from jax.experimental import optimizers, stax
from jax.random import PRNGKey
import jax

import numpyro.distributions as dist
from numpyro.handlers import param, sample

# note(lumip): unfortunately per_sample_elbo does not work for this case atm
# from dppp.svi import per_sample_elbo, svi
from numpyro.svi import svi, elbo


def model(batch_X, batch_y, **kwargs):
    """Defines the generative probabilistic model: p(x|z)p(z)

    The model is conditioned on the observed data
    :param batch_X: a batch of predictors
    :param batch_y: a batch of observations
    :param other keyword arguments' are accepted but ignored
    """
    z_dim = batch_X.shape[1]
    z_w = sample('w', dist.Normal(np.zeros((z_dim,)), np.ones((z_dim,)))) # prior is N(0,I)
    z_intercept = sample('intercept', dist.Normal(0,1)) # prior is N(0,1)
    logits = batch_X.dot(z_w)+z_intercept
    return sample('obs', dist.Bernoulli(logits=logits), obs=batch_y)

def guide(batch_X, batch_y, z_dim):
    """Defines the probabilistic guide for z (variational approximation to posterior): q(z) ~ p(z|q)
    """
    # note(lumip): we are interested in the posterior of w and intercept
    #   so.. like this then?
    z_w_loc = param("w_loc", np.zeros((z_dim,)))
    z_w_std = param("w_std", np.ones((z_dim,)))
    z_w = sample('w', dist.Normal(z_w_loc, z_w_std))
    z_intercept_loc = param("intercept_loc", 0.)
    z_interpet_std = param("intercept_std", 1.)
    z_intercept = sample('intercept', dist.Normal(z_intercept_loc, z_interpet_std))
    return (z_w, z_intercept)

def create_toy_data(N, d):
    ## Create some toy data
    import numpy as onp
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

    # note(lumip): unfortunately the per_sample_elbo (or rather the interally 
    #   used per_sample_log_density) method is currently unable to deal with the
    #   case where multiple samples influence the same instance of a latent
    #   variable, which is the case here. for now, falling back to the original
    #   numpyro svi procedures to get things running

    # per_sample_loss = per_sample_elbo
    # combined_loss = np.sum
    # svi_init, svi_update, svi_eval = svi(
    #     model, guide, per_sample_loss, combined_loss, opt_init, opt_update, 
    #     get_params, z_dim=args.dimensions
    # )

    svi_init, svi_update, svi_eval = svi(
        model, guide, elbo, opt_init, opt_update, 
        get_params, z_dim=args.dimensions
    )
    svi_update = jit(svi_update)

    rng = PRNGKey(123)

    rng, svi_init_rng, data_fetch_rng = random.split(rng, 3)
    _, train_idx = train_init(rng=data_fetch_rng)
    batch_X, batch_Y = train_fetch(0, train_idx)
    opt_state = svi_init(svi_init_rng, (batch_X, batch_Y), (batch_X, batch_Y))

    @jit
    def epoch_train(opt_state, rng):
        def body_fn(i, val):
            loss_sum, opt_state, rng = val
            rng, update_rng = random.split(rng, 2)
            batch = train_fetch(i, train_idx)
            loss, opt_state, rng = svi_update(
                i, opt_state, update_rng, batch, batch,
            )
            loss_sum += loss
            return loss_sum, opt_state, rng

        return lax.fori_loop(0, num_train, body_fn, (0., opt_state, rng))

    
    @jit
    def eval_test(opt_state, rng):
        # computing loss over training data (for now?)
        def body_fn(i, val):
            loss_sum, rng = val
            batch = train_fetch(i, train_idx)
            loss = svi_eval(opt_state, rng, batch, batch) / len(batch)
            loss_sum += loss
            return loss_sum, rng

        loss, _ = lax.fori_loop(0, num_train, body_fn, (0., rng))
        loss = loss / num_train
        return loss

	## Train model
    for i in range(args.num_epochs):
        t_start = time.time()
        rng, data_fetch_rng, test_rng = random.split(rng, 3)

        num_train, train_idx = train_init(rng=data_fetch_rng)
        _, opt_state, rng = epoch_train(opt_state, rng)

        num_train, train_idx = train_init(rng=data_fetch_rng)
        test_loss = eval_test(opt_state, test_rng)

        print("Epoch {}: loss = {} ({:.2f} s.)".format(
            i, test_loss, time.time() - t_start
        ))
        params = get_params(opt_state)
        print(params['w_loc'])

    print(w_true)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=100, type=int, help='number of training epochs')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('-batch-size', default=128, type=int, help='batch size')
    parser.add_argument('-d', '--dimensions', default=10, type=int, help='data dimension')
    parser.add_argument('-N', '--num-samples', default=1000, type=int, help='data samples count')
    args = parser.parse_args()
    main(args)
