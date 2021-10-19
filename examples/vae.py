# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © -2019 Copyright Contributors to the Pyro project.
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

"""VAE example from numpyro.

original: https://github.com/pyro-ppl/numpyro/blob/master/examples/vae.py
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

import matplotlib.pyplot as plt

import jax.numpy as jnp
from jax import jit, lax, random
from jax.experimental import stax
from jax.random import PRNGKey
import jax

import numpy as np

import numpyro
import numpyro.optim as optimizers
import numpyro.distributions as dist
from numpyro.primitives import sample, plate
from numpyro.handlers import scale
from numpyro.infer import Trace_ELBO as ELBO, SVI

from d3p.svi import DPSVI
from d3p.modelling import sample_multi_posterior_predictive
from d3p.minibatch import split_batchify_data, subsample_batchify_data
from d3p.dputil import approximate_sigma
from d3p.util import is_int_scalar

from datasets import MNIST, load_dataset

RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__),
                              '.results'))
os.makedirs(RESULTS_DIR, exist_ok=True)


def encoder(hidden_dim, z_dim):
    """Defines the encoder, i.e., the network taking us from observations
        to (a distribution of) latent variables.

    z is following a normal distribution, thus needs mean and variance.

    Network structure:
    x -> dense layer of hidden_dim with softplus activation --> dense layer of z_dim ( = means/loc of z)
                                                            |-> dense layer of z_dim with (elementwise) exp() as activation func ( = variance of z )
    (note: the exp() as activation function serves solely to ensure positivity of the variance)

    :param hidden_dim: number of nodes in the hidden layer
    :param z_dim: dimension of the latent variable z
    :return: (init_fun, apply_fun) pair of the encoder: (encoder_init, encode)
    """
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()), stax.Softplus,
        stax.FanOut(2),
        stax.parallel(stax.Dense(z_dim, W_init=stax.randn()),
                      stax.serial(stax.Dense(z_dim, W_init=stax.randn()), stax.Exp)),
    )


def decoder(hidden_dim, out_dim):
    """Defines the decoder, i.e., the network taking us from latent
        variables back to observations (or at least observation space).

    Network structure:
    z -> dense layer of hidden_dim with softplus activation -> dense layer of out_dim with sigmoid activation

    :param hidden_dim: number of nodes in the hidden layer
    :param out_dim: dimensions of the observations

    :return: (init_fun, apply_fun) pair of the decoder: (decoder_init, decode)
    """
    return stax.serial(
        stax.Dense(hidden_dim, W_init=stax.randn()), stax.Softplus,
        stax.Dense(out_dim, W_init=stax.randn()), stax.Sigmoid,
    )

def model(batch_or_batchsize, z_dim, hidden_dim, out_dim=None, num_obs_total=None):
    """Defines the generative probabilistic model: p(x|z)p(z)

    The model is conditioned on the observed data

    :param batch: a batch of observations
    :param hidden_dim: dimensions of the hidden layers in the VAE
    :param z_dim: dimensions of the latent variable / code
    :param out_dim: number of dimensions in a single output sample (flattened)

    :return: (named) sample x from the model observation distribution p(x|z)p(z)
    """
    if is_int_scalar(batch_or_batchsize):
        batch = None
        batch_size = batch_or_batchsize
        if out_dim is None:
            raise ValueError("if no batch is provided, out_dim must be given")
    else:
        batch = batch_or_batchsize
        assert(jnp.ndim(batch) == 3)
        batch_size = jnp.shape(batch)[0]
        batch = jnp.reshape(batch, (batch_size, -1)) # squash each data item into a one-dimensional array (preserving only the batch size on the first axis)
        out_dim = jnp.shape(batch)[1]
    num_obs_total = batch_size if num_obs_total is None else num_obs_total

    decode = numpyro.module('decoder', decoder(hidden_dim, out_dim), (batch_size, z_dim))
    with plate('batch', num_obs_total, batch_size):
        z = sample('z', dist.Normal(jnp.zeros((z_dim,)), jnp.ones((z_dim,))).to_event(1)) # prior on z is N(0,I)
        img_loc = decode(z) # evaluate decoder (p(x|z)) on sampled z to get means for output bernoulli distribution
        x = sample('obs', dist.Bernoulli(img_loc).to_event(1), obs=batch) # outputs x are sampled from bernoulli distribution depending on z and conditioned on the observed data
        return x


def guide(batch, z_dim, hidden_dim, out_dim=None, num_obs_total=None):
    """Defines the probabilistic guide for z (variational approximation to posterior): q(z) ~ p(z|q)
    :param batch: a batch of observations
    :return: (named) sampled z from the variational (guide) distribution q(z)
    """
    assert(jnp.ndim(batch) == 3)
    batch_size = jnp.shape(batch)[0]
    batch = jnp.reshape(batch, (batch_size, -1)) # squash each data item into a one-dimensional array (preserving only the batch size on the first axis)
    out_dim = jnp.shape(batch)[1]
    num_obs_total = batch_size if num_obs_total is None else num_obs_total

    encode = numpyro.module('encoder', encoder(hidden_dim, z_dim), (batch_size, out_dim))
    with plate('batch', num_obs_total, batch_size):
        z_loc, z_std = encode(batch) # obtain mean and variance for q(z) ~ p(z|x) from encoder
        z = sample('z', dist.Normal(z_loc, z_std).to_event(1)) # z follows q(z)
        return z


@jit
def binarize(rng, batch):
    """Binarizes a batch of observations with values in [0,1] by sampling from
        a Bernoulli distribution and using the original observations as means.

    Reason: This example assumes a Bernoulli distribution for the decoder output
    and thus requires inputs to be binary values as well.

    :param rng: rng seed key
    :param batch: Batch of data with continous values in interval [0, 1]
    :return: tuple(rng, binarized_batch).
    """
    return random.bernoulli(rng, batch).astype(batch.dtype)


def main(args):
    # loading data
    (train_init, train_fetch_plain), num_samples = load_dataset(MNIST, batch_size=args.batch_size, split='train', batchifier=subsample_batchify_data)
    (test_init, test_fetch_plain), _ = load_dataset(MNIST, batch_size=args.batch_size, split='test', batchifier=split_batchify_data)

    def binarize_fetch(fetch_fn):
        @jit
        def fetch_binarized(batch_nr, batchifier_state, binarize_rng):
            batch = fetch_fn(batch_nr, batchifier_state)
            return binarize(binarize_rng, batch[0]), batch[1]
        return fetch_binarized

    train_fetch = binarize_fetch(train_fetch_plain)
    test_fetch = binarize_fetch(test_fetch_plain)

    # setting up optimizer
    optimizer = optimizers.Adam(args.learning_rate)

    # the plate environment in our model scales individual
    # records' contributions to the loss up by num_samples.
    # This can cause numerical instabilities so we scale down
    # the loss by 1/num_samples here.
    sc_model = scale(model, scale=1/num_samples)
    sc_guide = scale(guide, scale=1/num_samples)

    if args.no_dp:
        svi = SVI(sc_model, sc_guide, optimizer, ELBO(), num_obs_total=num_samples, z_dim=args.z_dim, hidden_dim=args.hidden_dim)
    else:
        q = args.batch_size / num_samples
        target_eps = args.epsilon
        dp_scale, act_eps, _ = approximate_sigma(
            target_eps=target_eps,
            delta=1/num_samples, q=q,
            num_iter=int(1/q) * args.num_epochs,
            force_smaller=True
        )
        print(f"using noise scale {dp_scale} for epsilon of {act_eps} (targeted: {target_eps})")

        svi = DPSVI(
            sc_model, sc_guide, optimizer, ELBO(),
            dp_scale=dp_scale, clipping_threshold=10.,
            num_obs_total=num_samples, z_dim=args.z_dim, hidden_dim=args.hidden_dim
        )

    # preparing random number generators and initializing svi
    rng = PRNGKey(0)
    rng, binarize_rng, svi_init_rng, batchifier_rng = random.split(rng, 4)
    _, batchifier_state = train_init(rng_key=batchifier_rng)
    sample_batch = train_fetch(0, batchifier_state, binarize_rng)[0]
    svi_state = svi.init(svi_init_rng, sample_batch)

    # functions for training tasks
    @jit
    def epoch_train(svi_state, batchifier_state, num_batches, rng):
        """Trains one epoch

        :param svi_state: current state of the optimizer
        :param rng: rng key

        :return: overall training loss over the epoch
        """

        def body_fn(i, val):
            svi_state, loss = val
            binarize_rng = random.fold_in(rng, i)
            batch = train_fetch(i, batchifier_state, binarize_rng)[0]
            svi_state, batch_loss = svi.update(
                svi_state, batch
            )
            loss += batch_loss / num_batches
            return svi_state, loss

        svi_state, loss = lax.fori_loop(0, num_batches, body_fn, (svi_state, 0.))
        return svi_state, loss

    @jit
    def eval_test(svi_state, batchifier_state, num_batches, rng):
        """Evaluates current model state on test data.

        :param svi_state: current state of the optimizer
        :param rng: rng key

        :return: loss over the test split
        """
        def body_fn(i, loss_sum):
            binarize_rng = random.fold_in(rng, i)
            batch = test_fetch(i, batchifier_state, binarize_rng)[0]
            batch_loss = svi.evaluate(svi_state, batch)
            loss_sum += batch_loss / num_batches
            return loss_sum

        return lax.fori_loop(0, num_batches, body_fn, 0.)

    def reconstruct_img(epoch, num_epochs, batchifier_state, svi_state, rng):
        """Reconstructs an image for the given epoch

        Obtains a sample from the testing data set and passes it through the
        VAE. Stores the result as image file 'epoch_{epoch}_recons.png' and
        the original input as 'epoch_{epoch}_original.png' in folder '.results'.

        :param epoch: Number of the current epoch
        :param num_epochs: Number of total epochs
        :param opt_state: Current state of the optimizer
        :param rng: rng key
        """
        assert(num_epochs > 0)
        img = test_fetch_plain(0, batchifier_state)[0][0]
        plt.imsave(
            os.path.join(RESULTS_DIR, "epoch_{:0{}d}_original.png".format(
                epoch, (int(jnp.log10(num_epochs))+1))
            ),
            img,
            cmap='gray'
        )
        rng, rng_binarize = random.split(rng, 2)
        test_sample = binarize(rng_binarize, img)
        test_sample = jnp.reshape(test_sample, (1, *jnp.shape(test_sample)))
        params = svi.get_params(svi_state)

        samples = sample_multi_posterior_predictive(
            rng, 10, model, (1, args.z_dim, args.hidden_dim, np.prod(test_sample.shape[1:])),
            guide, (test_sample, args.z_dim, args.hidden_dim), params
        )

        img_loc = samples['obs'][0].reshape([28, 28])
        avg_img_loc = jnp.mean(samples['obs'], axis=0).reshape([28, 28])
        plt.imsave(
            os.path.join(RESULTS_DIR, "epoch_{:0{}d}_recons_single.png".format(
                epoch, (int(jnp.log10(num_epochs))+1))
            ),
            img_loc,
            cmap='gray'
        )
        plt.imsave(
            os.path.join(RESULTS_DIR, "epoch_{:0{}d}_recons_avg.png".format(
                epoch, (int(jnp.log10(num_epochs))+1))
            ),
            avg_img_loc,
            cmap='gray'
        )

    # main training loop
    for i in range(args.num_epochs):
        t_start = time.time()
        rng, data_fetch_rng, train_rng = random.split(rng, 3)
        num_train_batches, train_batchifier_state, = train_init(rng_key=data_fetch_rng)
        svi_state, train_loss = epoch_train(
            svi_state, train_batchifier_state, num_train_batches, train_rng
        )

        rng, test_fetch_rng, test_rng, recons_rng = random.split(rng, 4)
        num_test_batches, test_batchifier_state = test_init(rng_key=test_fetch_rng)
        test_loss = eval_test(svi_state, test_batchifier_state, num_test_batches, test_rng)

        reconstruct_img(i, args.num_epochs, test_batchifier_state, svi_state, recons_rng)
        print("Epoch {}: loss = {} (on training set: {}) ({:.2f} s.)".format(
            i, test_loss, train_loss, time.time() - t_start
        ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=20, type=int, help='number of training epochs')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    parser.add_argument('--z-dim', default=50, type=int, help='size of latent')
    parser.add_argument('--hidden-dim', default=400, type=int, help='size of hidden layer in encoder/decoder networks')
    parser.add_argument('--epsilon', default=1., type=float, help='targeted value for privacy parameter epsilon')
    parser.add_argument('--no_dp', default=False, action='store_true', help='Use plain SVI instead of DPSVI algorithm')
    args = parser.parse_args()
    main(args)
