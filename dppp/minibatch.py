
from dppp.util import is_int_scalar, is_array, example_count
from numpyro.handlers import scale
import jax.numpy as np
import jax
from functools import partial

__all__ = [
    'minibatch', 'subsample_batchify_data', 'split_batchify_data',
    'q_to_batch_size', 'batch_size_to_q'
]

def minibatch(batch_or_batchsize, num_obs_total=None):
    """Returns a context within which all samples are treated as being a
    minibatch of a larger data set.

    In essence, this marks the (log)likelihood of the sampled examples to be
    scaled to the total loss value over the whole data set.

    :param batch_or_batchsize: An integer indicating the batch size or an array
        indicating the shape of the batch where the length of the first axis
        is interpreted as batch size.
    :param num_obs_total: The total number of examples/observations in the
        full data set. Optional, defaults to the given batch size.
    """
    if is_int_scalar(batch_or_batchsize):
        if not np.isscalar(batch_or_batchsize):
            raise ValueError("if a scalar is given for batch_or_batchsize, it "
                "can't be traced through jit. consider using static_argnums "
                "for the jit invocation.")
        batch_size = batch_or_batchsize
    elif is_array(batch_or_batchsize):
        batch_size = example_count(batch_or_batchsize)
    else:
        raise ValueError("batch_or_batchsize must be an array or an integer")
    if num_obs_total is None:
        num_obs_total = batch_size
    return scale(scale_factor = num_obs_total / batch_size)

def subsample_batchify_data(dataset, batch_size=None, q=None, with_replacement=False):
    """Returns functions to fetch (randomized) batches of a given dataset by 
    uniformly random subsampling.

    As `split_batchify_data`, takes the common epoch viewpoint to training,
    where an epoch is understood to be one pass over the data set. However,
    the data set is not shuffled and split to generate batches - instead
    every batch is drawn uniformly at random from the data set. An epoch thus
    merely refers to a number of batches that make up the same amount of data
    as the full data set.
    
    While each element of the data set in expectation occurs once per epoch,
    there are no guarantees to the exact number of appearances.

    The subsampling can be performed with or without replacement per batch.
    In the latter case (default), an element cannot occur more than once in a batch.

    The batches are guaranteed to always be of size batch_size. If the number of
    items in the data set is not evenly divisible by batch_size, the total number
    of elements contained in batches per epoch will be slightly less than the
    size of the data set.

    :param arrays: Tuple of arrays constituting the data set to be batchified.
        All arrays must have the same length on the first axis.
    :param batch_size: Size of the batches as absolute number. Mutually exclusive with q.
    :param q: Size of batches as ratio of the data set size. Mutually exlusive with batch_size.
    :return: tuple (init_fn: () -> (num_batches, batchifier_state), get_batch: (i, batchifier_state) -> batch)
        init_fn() returns the number of batches per epoch and an initialized state of the batchifier for the epoch
        get_batch() returns the next batch_size amount of items from the data set
    """
    if batch_size is None and q is None:
        raise ValueError("Either batch_size or batch ratio q must be given")
    if batch_size is not None and q is not None:
        raise ValueError("Only one of batch_size and batch ratio q must be given")
    if not dataset:
        raise ValueError("The data set must not be empty")

    num_records = example_count(dataset[0])
    for arr in dataset:
        if num_records != example_count(arr):
            raise ValueError("All arrays constituting the data set must have the same number of records")

    if batch_size is None:
        batch_size = q_to_batch_size(q, num_records)

    @jax.jit
    def init(rng_key):
        """ Initializes the batchifier for a new epoch.

        :param rng_key: The base PRNG key the batchifier will use for randomness.
        :return: tuple consisting of: number of batches in the epoch, 
            initialized state of the batchifier for the epoch
        """
        return num_records // batch_size, rng_key

    @jax.jit
    def get_batch_with_replacement(i, batchifier_state):
        """ Fetches the next batch for the current epoch.

        :param i: The number of the batch in the epoch.
        :param batchifier_state: The initialized state returned by init.
        :return: the batch
        """
        rng_key = batchifier_state
        batch_rng_key = jax.random.fold_in(rng_key, i)
        ret_idx = jax.random.randint(rng_key, (batch_size,), 0, num_records)
        return tuple(np.take(a, ret_idx, axis=0) for a in dataset)

    @jax.jit
    def get_batch_without_replacement(i, rng_key):
        """ Fetches the next batch for the current epoch.

        :param i: The number of the batch in the epoch.
        :param batchifier_state: The initialized state returned by init.
        :return: the batch
        """
        batch_rng_key = jax.random.fold_in(rng_key, i)
        ret_idx = jax.random.shuffle(rng_key, np.arange(num_records))
        ret_idx = jax.lax.dynamic_slice_in_dim(ret_idx, 0, batch_size)
        return tuple(np.take(a, ret_idx, axis=0) for a in dataset)

    return init, get_batch_with_replacement if with_replacement else get_batch_without_replacement


def split_batchify_data(dataset, batch_size=None, q=None):
    """Returns functions to fetch (randomized) batches of a given data set by 
    shuffling and splitting the data set.

    Takes the common epoch viewpoint to training, where an epoch is understood
    to be one pass over the data set. Every element is guaranteed to be included
    not more than once per epoch. If the data set size is divisable by the batch
    size, every element is guaranteed to be included exactly once per epoch.

    The batches are guaranteed to always be of size batch_size. If the number of
    items in the data set is not evenly divisible by batch_size, some elements
    are left out of the batchification.

    :param arrays: Tuple of arrays constituting the data set to be batchified.
        All arrays must have the same length on the first axis.
    :param batch_size: Size of the batches as absolute number. Mutually exclusive with q.
    :param q: Size of batches as ratio of the data set size. Mutually exlusive with batch_size.
    :return: tuple (init_fn: () -> (num_batches, batchifier_state), get_batch: (i, batchifier_state) -> batch)
        init_fn() returns the number of batches per epoch and an initialized state of the batchifier for the epoch
        get_batch() returns the next batch_size amount of items from the data set
    """
    if batch_size is None and q is None:
        raise ValueError("Either batch_size or batch ratio q must be given")
    if batch_size is not None and q is not None:
        raise ValueError("Only one of batch_size and batch ratio q must be given")
    if not dataset:
        raise ValueError("The data set must not be empty")

    num_records = example_count(dataset[0])
    for arr in dataset:
        if num_records != example_count(arr):
            raise ValueError("All arrays constituting the data set must have the same number of records")

    if batch_size is None:
        batch_size = q_to_batch_size(q, num_records)

    @jax.jit
    def init(rng_key):
        """ Initializes the batchifier for a new epoch.

        :param rng_key: The base PRNG key the batchifier will use for randomness.
        :return: tuple consisting of: number of batches in the epoch, 
            initialized state of the batchifier for the epoch
        """
        idxs = np.arange(num_records)
        return num_records // batch_size, jax.random.shuffle(rng_key, idxs)

    @jax.jit
    def get_batch(i, idxs):
        """ Fetches the next batch for the current epoch.

        :param i: The number of the batch in the epoch.
        :param batchifier_state: The initialized state returned by init.
        :return: the batch
        """
        ret_idx = jax.lax.dynamic_slice_in_dim(idxs, i * batch_size, batch_size)
        return tuple(np.take(a, ret_idx, axis=0) for a in dataset)

    return init, get_batch

def q_to_batch_size(q, N):
    """ Returns the batch size for a given subsampling ratio q. """
    return int(N * q)

def batch_size_to_q(batch_size, N):
    """ Returns the subsampling ratio q corresponding to a given batch size. """
    return batch_size / N
