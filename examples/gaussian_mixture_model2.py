"""Gaussian mixture model example 2.

This example also demonstrates the gaussian mixture model but instead
of composing it from primitive distributions in the model and guide
respectively, we define a MixGaus distribution class to compute the probability
function for a Gaussian mixture, which we then directly use in the model.

note(lumip): Currently infers fewer priors than the other example and seems to
be unable to learn learn empty clusters (if the model is configured with more
components than there are in the data. This is (was) possible in the alternative
implementation). Somewhat prone to producing NaN values in certain conditions.

note(lumip): The reported loss from svi_update (training set) is roughly 1/2 of
that returned by svi_eval (on test or training set). Something seems not right
there..
"""

import os

# allow example to find dppp without installing
import sys
sys.path.append(os.path.dirname(sys.path[0]))
#### 

import argparse
import time

import numpy as onp

import jax
import jax.numpy as np
from jax import jit, lax, random
from jax.experimental import optimizers
from jax.random import PRNGKey
from jax.scipy.special import logsumexp

import numpyro.distributions as dist
from numpyro.handlers import seed
from numpyro.primitives import sample, param
from numpyro.svi import elbo
from numpyro.distributions.distribution import Distribution
from numpyro.distributions import constraints

from dppp.svi import dpsvi, minibatch

from datasets import batchify_data

# we define a Distribution subclass for the gaussian mixture model
class MixGaus(Distribution):
    arg_constraints = {
        'locs': constraints.real, 
        'scales': constraints.positive, 
        'pis' : constraints.simplex
    }
    support = constraints.real

    def __init__(self, locs=0., scales=1., pis=1.0, validate_args=None):
        self.locs, self.scales, self.pis = locs, scales, pis
        batch_shape = np.shape(locs[0])
        super(MixGaus, self).__init__(
            batch_shape=batch_shape, validate_args=validate_args
        )

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_pis = np.log(self.pis)
        log_phis = np.array([
            dist.Normal(loc, scale).log_prob(value).sum(-1)
            for loc, scale
            in zip(self.locs, self.scales)
        ]).T
        return logsumexp(log_pis + log_phis, axis=-1)

    def sample(self, *args):
        # ignoring this for now as we only care to compute the log_probability
        raise NotImplementedError()

def model(k, obs, num_obs_total=None):
    # this is our model function using the MixGaus distribution defined above
    # with prior belief
    assert(obs is not None)
    assert(np.ndim(obs) <= 2)
    # np.atleast_2d necessary because batch_size dimension is strapped during gradient computation
    batch_size, d = np.shape(np.atleast_2d(obs))

    pis = sample('pis', dist.Dirichlet(np.ones(k)))
    mus = sample('mus', dist.Normal(np.zeros((k, d)), 10.))
    sigs = np.ones((k, d))
    with minibatch(batch_size, num_obs_total=num_obs_total):
        return sample('obs', MixGaus(mus, sigs, pis), obs=obs)

def guide(k, obs, num_obs_total=None):
    # the latent MixGaus distribution which learns the parameters
    assert(obs is not None)
    _, d = np.shape(np.atleast_2d(obs))

    mus_loc = param('mus_loc', np.zeros((k, d)))
    mus = sample('mus', dist.Normal(mus_loc, 1.))
    sigs = np.ones((k, d))
    alpha = param('alpha', np.ones(k))
    pis = sample('pis', dist.Dirichlet(alpha))
    return pis, mus, sigs

def create_toy_data(N, k, d):
    """Creates some toy data (for training and testing)"""
    onp.random.seed(122)

    # We create some toy data. To spice things up, it is imbalanced:
    #   The last component has twice as many samples as the others.
    z = onp.random.randint(0, k+1, 2*N)
    z[z == k] = k - 1
    X = onp.zeros((2*N, d))

    assert(k < 4)
    mus = [-10. * onp.ones(d), 10. * onp.ones(d), -2. * onp.ones(d)]
    sigs = [onp.sqrt(0.1), 1., onp.sqrt(0.1)]
    for i in range(k):
        N_i = onp.sum(z == i)
        X_i = mus[i] + sigs[i] * onp.random.randn(N_i, d)
        X[z == i] = X_i

    mus = np.array(mus)
    sigs = np.array(sigs)

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
    N = np.atleast_1d(obs).shape[0]

    def per_component_fun(j):
        log_prob_x_zj = np.sum(dist.Normal(mus[j], sigs[j]).log_prob(obs), axis=1).flatten()
        assert(np.atleast_1d(log_prob_x_zj).shape == (N,))
        log_prob_zj = dist.Categorical(pis_prior).log_prob(j)
        log_prob = log_prob_x_zj + log_prob_zj
        assert(np.atleast_1d(log_prob).shape == (N,))
        return log_prob

    z_log_post = jax.vmap(per_component_fun)(np.arange(k))
    return z_log_post.T

def compute_assignment_accuracy(
    X_test, original_assignment, original_modes, posterior_modes, posterior_pis):
    """computes the accuracy score for attributing data to the mixture
    components based on the learned model
    """
    k, d = np.shape(original_modes)
    # we first map our true modes to the ones learned in the model using the
    # log posterior for z
    mode_assignment_posterior = compute_assignment_log_posterior(
        k, original_modes, posterior_modes, np.ones((k, d)), posterior_pis
    )
    mode_map = np.argmax(mode_assignment_posterior, axis=1)._value
    # a potential problem could be that mode_map might not be bijective, skewing
    # the results of the mapping. we build the inverse map and use identity
    # mapping as a base to counter that
    inv_mode_map = {j:j for j in range(k)}
    inv_mode_map.update({mode_map[j]:j for j in range(k)})
    
    # we next obtain the assignments for the data according to the model and
    # pass them through the inverse map we just build
    post_data_assignment = compute_assignment_log_posterior(
        k, X_test, posterior_modes, np.ones((k, d)), posterior_pis
    )
    post_data_assignment = np.argmax(post_data_assignment, axis=1)
    remapped_data_assignment = np.array(
        [inv_mode_map[j] for j in post_data_assignment._value]
    )

    # finally, we can compare the results with the original assigments and compute
    # the accuracy
    acc = np.mean(original_assignment == remapped_data_assignment)
    return acc


## main function: inference setup and main loop as well as subsequent
#   model quality check
def main(args):
    N = args.num_samples
    k = args.num_components
    k_gen = args.num_components_generated
    d = args.dimensions

    X_train, X_test, latent_vals = create_toy_data(N, k_gen, d)
    train_init, train_fetch = batchify_data((X_train,), args.batch_size)
    test_init, test_fetch = batchify_data((X_test,), args.batch_size)

    ## Init optimizer and training algorithms
    opt_init, opt_update, get_params = optimizers.adam(args.learning_rate)

    # note(lumip): fix the parameters in the models
    def fix_params(model_fn, k):
        def fixed_params_fn(obs, **kwargs):
            return model_fn(k, obs, **kwargs)
        return fixed_params_fn

    model_fixed = fix_params(model, k)
    guide_fixed = fix_params(guide, k)

    rng = PRNGKey(123)
    rng, dp_rng = random.split(rng, 2)

    # note(lumip): value for c currently completely made up
    #   value for dp_scale completely made up currently.
    svi_init, svi_update, svi_eval = dpsvi(
        model_fixed, guide_fixed, elbo, opt_init, opt_update, get_params, 
        rng=dp_rng, dp_scale=0.01, num_obs_total=args.num_samples,
        clipping_threshold=20.
    )

    svi_update = jit(svi_update)

    rng, svi_init_rng = random.split(rng, 2)
    batch = train_fetch(0)
    opt_state, _ = svi_init(svi_init_rng, batch, batch)

    @jit
    def epoch_train(rng, opt_state, data_idx, num_batch):
        def body_fn(i, val):
            loss, opt_state, rng = val
            rng, update_rng = random.split(rng, 2)
            batch = train_fetch(i, data_idx)
            batch_loss, opt_state, rng = svi_update(
                i, update_rng, opt_state, batch, batch
            )
            loss += batch_loss / (args.num_samples * num_batch)
            return loss, opt_state, rng

        return lax.fori_loop(0, num_batch, body_fn, (0., opt_state, rng))
    
    @jit
    def eval_test(rng, opt_state, data_idx, num_batch):
        def body_fn(i, val):
            loss_sum, rng = val
            batch = test_fetch(i, data_idx)
            loss = svi_eval(rng, opt_state, batch, batch)
            loss_sum += loss / (args.num_samples * num_batch)
            return loss_sum, rng

        loss, _ = lax.fori_loop(0, num_batch, body_fn, (0., rng))
        return loss

    smoothed_loss_window = onp.empty(5)
    smoothed_loss_window.fill(onp.nan)
    window_idx = 0

	## Train model
    for i in range(args.num_epochs):
        t_start = time.time()
        rng, train_rng, data_fetch_rng = random.split(rng, 3)

        num_train_batches, train_idx = train_init(rng=data_fetch_rng)
        train_loss, opt_state, _ = epoch_train(
            train_rng, opt_state, train_idx, num_train_batches
        )

        if i % 100 == 0:
            rng, test_rng, test_fetch_rng = random.split(rng, 3)
            num_test_batches, test_idx = test_init(rng=test_fetch_rng)
            test_loss = eval_test(
                test_rng, opt_state, test_idx, num_test_batches
            )
            smoothed_loss_window[window_idx] = test_loss
            smoothed_loss = onp.nanmean(smoothed_loss_window)
            window_idx = (window_idx + 1) % 5

            print("Epoch {}: loss = {} (smoothed = {}) (on training set = {}) ({:.2f} s.)".format(
                    i, test_loss, smoothed_loss, train_loss, time.time() - t_start
                ))

    params = get_params(opt_state)
    print(params)
    posterior_modes = params['mus_loc']
    posterior_pis = dist.Dirichlet(params['alpha']).mean
    print("MAP estimate of mixture weights: {}".format(posterior_pis))
    print("MAP estimate of mixture modes  : {}".format(posterior_modes))

    acc = compute_assignment_accuracy(
        X_test, latent_vals[1], latent_vals[2], posterior_modes, posterior_pis
    )
    print("assignment accuracy: {}".format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=2000, type=int, help='number of training epochs')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('-batch-size', default=32, type=int, help='batch size')
    parser.add_argument('-d', '--dimensions', default=2, type=int, help='data dimension')
    parser.add_argument('-N', '--num-samples', default=2048, type=int, help='data samples count')
    parser.add_argument('-k', '--num-components', default=3, type=int, help='number of components in the mixture model')
    parser.add_argument('-K', '--num-components-generated', default=3, type=int, help='number of components in generated data')
    args = parser.parse_args()
    main(args)
