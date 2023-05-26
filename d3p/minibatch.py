# SPDX-License-Identifier: Apache-2.0
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

from d3p.util import example_count, sample_from_array
import d3p.random as strong_rng
import jax.numpy as jnp
import jax
import scipy.stats
from functools import partial

__all__ = [
    'subsample_batchify_data', 'split_batchify_data', 'poisson_batchify_data',
    'q_to_batch_size', 'batch_size_to_q'
]


@partial(jax.jit, static_argnames=("N", "rng_suite", "cutoff_size"))
def poisson_sample_idxs(rng_key, q, N, rng_suite, cutoff_size=None):
    if cutoff_size is None or cutoff_size > N:
        cutoff_size = N

    selectors = rng_suite.uniform(rng_key, (N,), dtype=jnp.float32) <= q
    num_selected = jnp.sum(selectors)

    idxs = jnp.argsort(selectors)[::-1][:cutoff_size]

    return idxs, num_selected


def poisson_batchify_data(dataset, q, max_batch_size, handle_oversized_batch='truncate', rng_suite=strong_rng):
    """ Returns functions to fetch (randomized) batches of a given dataset by
    Poisson sampling.

    As `split_batchify_data` and `subsample_batchify_data, this takes the common epoch
    viewpoint to training, where an epoch is understood to be one pass over the data set.
    However, the data set is not shuffled and split to generate batches - instead
    every element is included in a batch with probability `q`, independent of any other element.

    This means that the number of elements in a batch varies (it follows a Poisson distribution).
    In order to facilitate efficient computation, returned batches are always of size
    `max_batch_size` - padding is used whenever the number of sampled elements in a batch is lower
    and a boolean mask is returned additionally to the batch to indicate which elements are valid (not padded).

    If the number of sampled elements is larger than `max_batch_size`, two different behaviors are possible
    depending on the value of `handle_oversized_batch`:
        - `truncate`: the batch is simply truncated to `max_batch_size`
        - `suppress`: the batch sample is discarded as invalid, i.e., an empty batch is returned.


    :param arrays: Tuple of arrays constituting the data set to be batchified.
        All arrays must have the same length on the first axis.
    :param q: The subsampling ratio, i.e., the probability with which an element is included in a batch.
    :param max_batch_size: The structural size of batches returned as an integer. Alternatively, a float between 0 and 1,
        in which case max_batch_size is set to the corresponding quantile of the Poisson distribution of batch sizes.
    :param handle_oversized_batch: Optional. How to handle cases when the number of sampled elements exceeds `max_batch_size`.
        Must be `truncate` or `suppress`. Default: `truncate`.
    :param rng_suite: Optional. The PRNG suite to use. Defaults to the cryptographically-secure d3p.random.
    :return: tuple (init_fn: () -> (num_batches, batchifier_state), get_batch: (i, batchifier_state) -> batch)
        init_fn() returns the number of batches per epoch and an initialized state of the batchifier for the epoch.
        get_batch() returns the next batch of elements from the data set and a Boolean mask indicating padding (True indicates a valid element).
    """
        
    if not dataset:
        raise ValueError("The data set must not be empty")
    if not isinstance(dataset, tuple):
        raise ValueError("Parameter dataset must be a tuple containing arrays of equal length.")
    if q < 0 or q > 1:
        raise ValueError("Parameter q must be >=0 and <=1.")

    num_records = example_count(dataset[0])
    for arr in dataset:
        if num_records != example_count(arr):
            raise ValueError("All arrays constituting the data set must have the same number of records")

    if max_batch_size < 0:
        raise ValueError("max_batch_size must be a positive integer denoting the maximum batch size,"
                         " or a float between 0 and 1 denoting the maximum batch size in terms of Poisson probability mass.")
    if not isinstance(max_batch_size, int):
        max_batch_size = int(scipy.stats.poisson(num_records * q).ppf(max_batch_size))
    
    @jax.jit
    def init(rng_key: rng_suite.PRNGState):
        """ Initializes the batchifier for a new epoch.

        :param rng_key: The base PRNG key the batchifier will use for randomness.
        :return: tuple consisting of: number of batches in the epoch,
            initialized state of the batchifier for the epoch
        """
        return num_records // int(q * num_records), rng_key

    @jax.jit
    def get_batch(i, batchifier_state):
        """ Fetches the next batch for the current epoch.

        :param i: The number of the batch in the epoch.
        :param batchifier_state: The initialized state returned by init.
        :return: tuple (batch, mask), where
            - batch is a tuple of arrays, each of length `max_batch_size` and containing
                the sampled elements from the arrays in `data` corresponding to the batch,
            - mask is a Boolean array of length `max_batch_size` indicating which elements in
                batch arrays correspond to batch elements (`True`) and which constitute padding (`False`).
        """
        rng_key = rng_suite.fold_in(batchifier_state, i)
        idxs, num_selected = poisson_sample_idxs(rng_key, q, num_records, rng_suite, cutoff_size=max_batch_size)
        assert len(idxs) == max_batch_size

        if handle_oversized_batch == "suppress":
            num_selected = (num_selected <= max_batch_size) * num_selected # => 0 if num_selected > max_batch_size
        else:
            num_selected = jnp.minimum(num_selected, max_batch_size)

        mask = jnp.arange(max_batch_size) < num_selected

        def map_single(a):
            taken = jnp.take(a, idxs, axis=0, unique_indices=True)
            mask_shape = (-1,) + (1,) * len(taken.shape[1:])
            return jnp.reshape(mask, mask_shape) * taken

        return tuple(map_single(a) for a in dataset), mask

    return init, get_batch


def subsample_batchify_data(dataset, batch_size=None, q=None, with_replacement=False, rng_suite=strong_rng, return_mask=False):
    """ Returns functions to fetch (randomized) batches of a given dataset by
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
    :param with_replacement: Optional. Sample with replacements. Default: true.
    :param rng_suite: Optional. The PRNG suite to use. Defaults to the cryptographically-secure d3p.random.
    :param return_mask: Optional. If True, `get_batch` returns a Boolean mask indicating valid (unpadded) elements in the batch.
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
    def init(rng_key: rng_suite.PRNGState):
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
        :return: batch or, if `return_mask` was `True`, tuple (batch, mask), where
            - batch is a tuple of arrays, each of length `batch_size` and containing
                the sampled elements from the arrays in `data` corresponding to the batch,
            - mask is a Boolean array of length `batch_size` indicating which elements in
                batch arrays correspond to batch elements (`True`) and which constitute padding (`False`).
        """
        rng_key = batchifier_state
        batch_rng_key = rng_suite.fold_in(rng_key, i)
        ret_idx = rng_suite.randint(batch_rng_key, (batch_size,), 0, num_records)

        batch = tuple(jnp.take(a, ret_idx, axis=0) for a in dataset)
        if return_mask:
            mask = jnp.ones(batch_size, dtype=bool)
            return batch, mask
        return batch
        

    @jax.jit
    def get_batch_without_replacement(i, batchifier_state):
        """ Fetches the next batch for the current epoch.

        :param i: The number of the batch in the epoch.
        :param batchifier_state: The initialized state returned by init.
        :return: batch or, if `return_mask` was `True`, tuple (batch, mask), where
            - batch is a tuple of arrays, each of length `batch_size` and containing
                the sampled elements from the arrays in `data` corresponding to the batch,
            - mask is a Boolean array of length `batch_size` indicating which elements in
                batch arrays correspond to batch elements (`True`) and which constitute padding (`False`).
        """
        rng_key = batchifier_state
        batch_rng_key = rng_suite.fold_in(rng_key, i)
        ret_idx = sample_from_array(batch_rng_key, jnp.arange(num_records), batch_size, 0, rng_suite=rng_suite)

        batch = tuple(jnp.take(a, ret_idx, axis=0) for a in dataset)
        if return_mask:
            mask = jnp.ones(batch_size, dtype=bool)
            return batch, mask
        return batch

    return init, get_batch_with_replacement if with_replacement else get_batch_without_replacement


def split_batchify_data(dataset, batch_size=None, q=None, rng_suite=strong_rng, return_mask=False):
    """ Returns functions to fetch (randomized) batches of a given data set by
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
    :param rng_suite: Optional. The PRNG suite to use. Defaults to the cryptographically-secure d3p.random.
    :param return_mask: Optional. If True, `get_batch` returns a Boolean mask indicating valid (unpadded) elements in the batch.
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
    def init(rng_key: rng_suite.PRNGState):
        """ Initializes the batchifier for a new epoch.

        :param rng_key: The base PRNG key the batchifier will use for randomness.
        :return: tuple consisting of: number of batches in the epoch,
            initialized state of the batchifier for the epoch
        """
        idxs = jnp.arange(num_records)
        shuffled_idxs = sample_from_array(rng_key, idxs, num_records, 0, rng_suite=rng_suite)
        return num_records // batch_size, shuffled_idxs

    @jax.jit
    def get_batch(i, idxs):
        """ Fetches the next batch for the current epoch.

        :param i: The number of the batch in the epoch.
        :param batchifier_state: The initialized state returned by init.
        :return: batch or, if `return_mask` was `True`, tuple (batch, mask), where
            - batch is a tuple of arrays, each of length `batch_size` and containing
                the sampled elements from the arrays in `data` corresponding to the batch,
            - mask is a Boolean array of length `batch_size` indicating which elements in
                batch arrays correspond to batch elements (`True`) and which constitute padding (`False`).
        """
        ret_idx = jax.lax.dynamic_slice_in_dim(idxs, i * batch_size, batch_size)
    
        batch = tuple(jnp.take(a, ret_idx, axis=0) for a in dataset)
        if return_mask:
            mask = jnp.ones(batch_size, dtype=bool)
            return batch, mask
        return batch

    return init, get_batch


def q_to_batch_size(q, N):
    """ Returns the batch size for a given subsampling ratio q. """
    return int(N * q)


def batch_size_to_q(batch_size, N):
    """ Returns the subsampling ratio q corresponding to a given batch size. """
    return batch_size / N
