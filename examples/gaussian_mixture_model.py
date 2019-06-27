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
from numpyro.handlers import param, sample, seed, trace, substitute

#from dppp.svi import per_example_elbo, svi
from numpyro.svi import elbo, svi

from datasets import batchify_data

def model(k, d, N=1, obs=None):
    """Defines the generative probabilistic model: p(x|z)p(z)

    :param k: number of components in the mixture
    :param d: number of dimensions per data item
    :param N: number of samples (default: 1) (ignored and set to obs.shape[0] if obs is given)
    :param obs: observed samples to condition the model with (default: None)
    """
    # f(x) = sum_k pi_k * phi(x; mu_k, sigma_k^2), where phi denotes Gaussian pdf
    # 	* pi_k ~ Dirichlet(alpha), where alpha \in R_+^k
    # 	* mu_k ~ Normal(0, 1)
    # * sigma_k ~ Gamma(a0, b0), where a0,b0 > 0

    if obs is not None:
        assert(N == np.atleast_1d(obs).shape[0])

    alpha = np.ones(k)*5.
    a0, b0 = np.ones((k,d))*2., np.ones((k,d))*2.

    pis = np.broadcast_to(sample('pis', dist.Dirichlet(alpha)), (N,k))
    mus = sample('mus', dist.Normal(np.zeros((k, d)), 1.))
    sigs = sample('sigmas', dist.Gamma(a0, b0))
    var = np.square(sigs)

    ks = sample('ks', dist.Categorical(pis))
    return sample('obs', dist.Normal(mus[ks], var[ks]), obs=obs)

def guide(k, d, N=1, obs=None):
    """Defines the probabilistic guide for z (variational approximation to posterior): q(z) ~ p(z|x)

    :param k: number of components in the mixture
    :param d: number of dimensions per data item
    :param N: number of samples (default: 1) (ignored and set to obs.shape[0] if obs is given)
    :param obs: observed samples to condition the model with (default: None)
    """
    # todo(lumip): guide is too simple! model does not converge! fix
    # note(lumip): various sources claim that SVI through discrete random
    #   variables such as the choice of category (ks) is very unstable. Pyro
    #   suggests using either conditional independence or baselines.. does
    #   numpyro support that as well?
    if obs is not None:
        assert(N == np.atleast_1d(obs).shape[0])

    a0, b0 = param('a0', np.ones((k, d))*2.), param('b0', np.ones((k, d))*2.)
    alpha = param('alpha', np.ones(k)*5)
    mus_loc = param('mus_loc', np.zeros((k,d)))
    mus_var = np.exp(param('mus_var_log', np.zeros((k, d))))

    pis = np.broadcast_to(sample('pis', dist.Dirichlet(alpha)), (N,k))
    mus = sample('mus', dist.Normal(mus_loc, mus_var))
    sigs = sample('sigmas', dist.Gamma(a0, b0))

    ks = sample('ks', dist.Categorical(pis))
    return pis, ks, mus, sigs

def create_toy_data(N, k, d):
    ## Create some toy data
    rng = jax.random.PRNGKey(122)

    # we simply use our defined model to sample some random toy data
    # trace is used to also obtain values of the latent variables
    m = seed(model, rng)
    tr = trace(m).get_trace(k, d, N)
    pis_true = tr['pis']['value']
    mus_true = tr['mus']['value']
    sigs_true = tr['sigmas']['value']
    ks_true = tr['ks']['value']
    X = tr['obs']['value']

    latent_vals = (pis_true, ks_true, mus_true, sigs_true)
    print(pis_true)
    print(ks_true[10])
    print("")
    return X, latent_vals

def main(args):
    N = args.num_samples
    k = args.num_components
    d = args.dimensions

    X, latent_vals = create_toy_data(N, k, d)
    # print(X)
    # print(latent_vals[0])
    # print(latent_vals[1])
    # print(latent_vals[2])
    # print(latent_vals[3])
    train_init, train_fetch = batchify_data((X,), args.batch_size)

    ## Init optimizer and training algorithms
    opt_init, opt_update, get_params = optimizers.adam(args.learning_rate)

    def fix_params(model_fn, N, k, d):
        def fixed_params_fn(obs=None):
            return model_fn(k, d, N, obs)
        return fixed_params_fn

    model_fixed = fix_params(model, args.batch_size, k, d)
    guide_fixed = fix_params(guide, args.batch_size, k, d)

    # per_example_loss = per_example_elbo
    # combined_loss = np.sum
    # svi_init, svi_update, svi_eval = svi(
    #     model_fixed, guide_fixed, per_example_loss, combined_loss, opt_init,
    #     opt_update, get_params, per_example_variables={'obs', 'ks'}
    # )
    svi_init, svi_update, svi_eval = svi(
        model_fixed, guide_fixed, elbo, opt_init,
        opt_update, get_params
    )

    svi_update = jit(svi_update)

    rng = PRNGKey(123)

    rng, svi_init_rng, data_fetch_rng = random.split(rng, 3)
    # _, train_idx = train_init(rng=data_fetch_rng)
    # batch, = train_fetch(0, train_idx)
    opt_state = svi_init(svi_init_rng, (), ())
    # opt_state = {}

    @jit
    def epoch_train(opt_state, rng, data_idx, num_batch):
        def body_fn(i, val):
            loss_sum, opt_state, rng = val
            rng, update_rng = random.split(rng, 2)
            batch = train_fetch(i, data_idx)
            loss, opt_state, rng = svi_update(
                i, opt_state, update_rng, batch, batch
            )
            loss = 0.
            loss_sum += loss / len(batch)
            return loss_sum, opt_state, rng

        loss, opt_state, rng = lax.fori_loop(0, num_batch, body_fn, (0., opt_state, rng))
        loss /= num_batch
        return loss, opt_state, rng

    
    @jit
    def eval_test(opt_state, rng, data_idx, num_batch):
        def body_fn(i, val):
            loss_sum, rng = val
            batch = train_fetch(i, data_idx)
            loss = svi_eval(opt_state, rng, batch, batch) / len(batch)
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

        print("Epoch {}: loss = {} ({:.2f} s.)".format(
                i, test_loss, time.time() - t_start
            ))

        # if (i % 100) == 0:
        #     print("Epoch {}: loss = {} ({:.2f} s.)".format(
        #         i, test_loss, time.time() - t_start
        #     ))
        #     params = get_params(opt_state)
        #     print("w_loc: {}".format(params['w_loc']))
        #     print("w_std: {}".format(np.exp(params['w_std_log'])))

    params = get_params(opt_state)
    print(params)
    trained_model = substitute(model, params)
    tr = trace(seed(trained_model, jax.random.PRNGKey(0))).get_trace(k, d)
    print(dist.Dirichlet(params['alpha']).mean)
    print(tr['pis']['value'])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=1000, type=int, help='number of training epochs')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('-batch-size', default=100, type=int, help='batch size')
    parser.add_argument('-d', '--dimensions', default=10, type=int, help='data dimension')
    parser.add_argument('-N', '--num-samples', default=1000, type=int, help='data samples count')
    parser.add_argument('-k', '--num-components', default=3, type=int, help='number of components in the mixture model')
    args = parser.parse_args()
    main(args)
