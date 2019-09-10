"""VAE example from numpyro.

original: https://github.com/pyro-ppl/numpyro/blob/master/examples/vae.py
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

import numpyro.distributions as dist
from numpyro.primitives import param, sample
from numpyro.svi import elbo

from dppp.svi import dpsvi, minibatch
from dppp.util import example_count, unvectorize_shape_3d

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
    note(lumip): I believe the exp() as activation function is solely to ensure positivity of the variance

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


def model(batch, decode, z_dim, num_obs_total=None, **kwargs):
    """Defines the generative probabilistic model: p(x|z)p(z)

    The model is conditioned on the observed data
    
    :param batch: a batch of observations
    :param decode: function implementing the decoder (latent -> observations)
    :param z_dim: dimensions of the latent variable / code
    :param other keyword arguments' are accepted but ignored

    :return: (named) sample x from the model observation distribution p(x|z)p(z)
    """
    assert(np.ndim(batch) == 3 or np.ndim(batch) == 2)
    batch_size = unvectorize_shape_3d(batch)[0]
    batch = np.reshape(batch, (batch_size, -1)) # squash each data item into a one-dimensional array (preserving only the batch size on the first axis)

    decoder_params = param('decoder', None) # advertise/register decoder parameters
    with minibatch(batch_size, num_obs_total=num_obs_total):
        z = sample('z', dist.Normal(np.zeros((z_dim,)), np.ones((z_dim,)))) # prior on z is N(0,I)
        img_loc = decode(decoder_params, z) # evaluate decoder (p(x|z)) on sampled z to get means for output bernoulli distribution
        x = sample('obs', dist.Bernoulli(img_loc), obs=batch) # outputs x are sampled from bernoulli distribution depending on z and conditioned on the observed data
        return x


def guide(batch, encode, num_obs_total=None, **kwargs):
    """Defines the probabilistic guide for z (variational approximation to posterior): q(z) ~ p(z|q)

    :param batch: a batch of observations
    :param encode: function implementing the encoder (observations -> latent)
    :param other keyword arguments: are accepted but ignored
    :return: (named) sampled z from the variational (guide) distribution q(z)
    """
    assert(np.ndim(batch) == 3 or np.ndim(batch) == 2)
    batch_size = unvectorize_shape_3d(batch)[0]
    batch = np.reshape(batch, (batch_size, -1)) # squash each data item into a one-dimensional array (preserving only the batch size on the first axis)

    encoder_params = param('encoder', None) # advertise/register encoder parameters
    with minibatch(batch_size, num_obs_total=num_obs_total):
        z_loc, z_std = encode(encoder_params, batch) # obtain mean and variance for q(z) ~ p(z|x) from encoder
        z = sample('z', dist.Normal(z_loc, z_std)) # z follows q(z)
        return z


@jit
def binarize(rng, batch):
    """Binarizes a batch of observations with values in [0,1] by sampling from
        a Bernoulli distribution and using the original observations as means.
    
    Reason: This example assumes a Bernoulli distribution for the decoder output
    and thus requires inputs to be binary values as well.

    note(lumip): From an answer to a pyro github issue for similar VAE example
    code using MNIST ( https://github.com/pyro-ppl/pyro/issues/529#issuecomment-342670366 ):
    "be aware to only do this once, as repeated sampling of the data provides
    unfair regularization to the model and also inflates likelihood scores".
    This is not how binarize is currently used throughout this file, i.e.,
    repeated sampling occurs. Is that intended? what is "unfair" regularization
    supposed to mean in that context, though?

    :param rng: rng seed key
    :param batch: Batch of data with continous values in interval [0, 1]
    :return: tuple(rng, binarized_batch).
    """
    return random.bernoulli(rng, batch).astype(batch.dtype)


def main(args):
    # loading data
    train_init, train_fetch_plain, num_samples = load_dataset(MNIST, batch_size=args.batch_size, split='train')
    test_init, test_fetch_plain, _ = load_dataset(MNIST, batch_size=args.batch_size, split='test')

    # obtaining model and training algorithms
    out_dim = 28*28
    encoder_init, encode = encoder(args.hidden_dim, args.z_dim)
    decoder_init, decode = decoder(args.hidden_dim, out_dim)
    opt_init, opt_update, get_params = optimizers.adam(args.learning_rate)

    # preparing random number generators
    rng = PRNGKey(0)
    rng, dp_rng = random.split(rng, 2)

    # note(lumip): choice of c is somewhat arbitrary at the moment.
    #   in early iterations gradient norm values are typically
    #   between 100 and 200 but in epoch 20 usually at 280 to 290.
    #   value for dp_scale completely made up currently.
    svi_init, svi_update, svi_eval = dpsvi(
        model, guide, elbo, opt_init, opt_update, 
        get_params, clipping_threshold=300.,
        rng=dp_rng, dp_scale=0.01, num_obs_total=num_samples,
        encode=encode, decode=decode, z_dim=args.z_dim
    )

    svi_update = jit(svi_update)

    # decorate batch fetching routines with binarization
    rng, rng_enc, rng_dec, rng_shuffle_train = random.split(rng, 4)

    def binarize_fetch(fetch_fn):
        def fetch_binarized(batch_nr, idxs, binarize_rng):
            batch = fetch_fn(batch_nr, idxs)
            return binarize(binarize_rng, batch[0]), batch[1]
        return jit(fetch_binarized)

    train_fetch = binarize_fetch(train_fetch_plain)
    test_fetch = binarize_fetch(test_fetch_plain)

    # initializing model and training algorithms
    _, encoder_params = encoder_init(rng_enc, (args.batch_size, out_dim))
    _, decoder_params = decoder_init(rng_dec, (args.batch_size, args.z_dim))
    params = {'encoder': encoder_params, 'decoder': decoder_params}

    rng, svi_init_rng = random.split(rng, 2)
    rng_shuffle_train, binarize_rng = random.split(rng_shuffle_train, 2)
    _, train_idx = train_init()
    sample_batch = train_fetch(0, train_idx, binarize_rng)[0]
    opt_state, _ = svi_init(svi_init_rng, (sample_batch,), (sample_batch,), params)

    # functions for training tasks
    @jit
    def epoch_train(rng, opt_state, train_idx, num_batch):
        """Trains one epoch

        :param opt_state: current state of the optimizer
        :param rng: rng key

        :return: overall training loss over the epoch
        """
        def body_fn(i, val):
            loss, opt_state, rng = val
            rng, binarize_rng, update_rng = random.split(rng, 3)
            batch = train_fetch(i, train_idx, binarize_rng)[0]
            batch_loss, opt_state, rng = svi_update(
                i, update_rng, opt_state, (batch,), (batch,),
            )
            loss += batch_loss / (num_samples * num_batch)
            return loss, opt_state, rng

        return lax.fori_loop(0, num_batch, body_fn, (0., opt_state, rng))

    @jit
    def eval_test(rng, opt_state, test_idx, num_batch):
        """Evaluates current model state on test data.

        :param opt_state: current state of the optimizer
        :param rng: rng key

        :return: loss over the test split
        """
        def body_fn(i, val):
            loss_sum, rng = val
            rng, binarize_rng, eval_rng = random.split(rng, 3)
            batch = test_fetch(i, test_idx, binarize_rng)[0]
            loss = svi_eval(eval_rng, opt_state, (batch,), (batch,))
            loss_sum += loss / (num_samples * num_batch)
            return loss_sum, rng

        loss, _ = lax.fori_loop(0, num_batch, body_fn, (0., rng))
        return loss

    def reconstruct_img(epoch, num_epochs, opt_state, rng):
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
        img = test_fetch_plain(0, test_idx)[0][0]
        plt.imsave(
            os.path.join(RESULTS_DIR, "epoch_{:0{}d}_original.png".format(
                epoch, (int(np.log10(num_epochs))+1))
            ),
            img,
            cmap='gray'
        )
        rng, rng_binarize = random.split(rng, 2)
        test_sample = binarize(rng_binarize, img)
        params = get_params(opt_state)
        z_mean, z_var = encode(params['encoder'], test_sample.reshape([1, -1]))
        z = dist.Normal(z_mean, z_var).sample(rng)
        img_loc = decode(params['decoder'], z).reshape([28, 28])
        plt.imsave(
            os.path.join(RESULTS_DIR, "epoch_{:0{}d}_recons.png".format(
                epoch, (int(np.log10(num_epochs))+1))
            ),
            img_loc,
            cmap='gray'
        )

    # main training loop
    for i in range(args.num_epochs):
        t_start = time.time()
        rng_shuffle_train, rng_train_init, rng_test_init = random.split(
            rng_shuffle_train, 3
        )
        num_train_batches, train_idx = train_init(rng=rng_train_init)
        train_loss, opt_state, rng = epoch_train(
            rng, opt_state, train_idx, num_train_batches
        )

        rng, rng_test, rng_recons = random.split(rng, 3)
        num_test_batches, test_idx = test_init(rng=rng_test_init)
        test_loss = eval_test(rng_test, opt_state, test_idx, num_test_batches)

        reconstruct_img(i, args.num_epochs, opt_state, rng_recons)
        print("Epoch {}: loss = {} (on training set: {}) ({:.2f} s.)".format(
            i, test_loss, train_loss, time.time() - t_start
        ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="parse args")
    parser.add_argument('-n', '--num-epochs', default=20, type=int, help='number of training epochs')
    parser.add_argument('-lr', '--learning-rate', default=1.0e-3, type=float, help='learning rate')
    parser.add_argument('-batch-size', default=128, type=int, help='batch size')
    parser.add_argument('-z-dim', default=50, type=int, help='size of latent')
    parser.add_argument('-hidden-dim', default=400, type=int, help='size of hidden layer in encoder/decoder networks')
    args = parser.parse_args()
    main(args)
