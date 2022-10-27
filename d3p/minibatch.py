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

__all__ = [
    'subsample_batchify_data', 'split_batchify_data',
    'q_to_batch_size', 'batch_size_to_q'
]


def subsample_batchify_data(dataset, batch_size=None, q=None, with_replacement=False, rng_suite=strong_rng):
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
        :return: the batch
        """
        rng_key = batchifier_state
        batch_rng_key = rng_suite.fold_in(rng_key, i)
        batch_jax_rng_key = rng_suite.convert_to_jax_rng_key(batch_rng_key)
        # TODO: introduce own randint to avoid falling back to jax randint here
        ret_idx = jax.random.randint(batch_jax_rng_key, (batch_size,), 0, num_records)
        return tuple(jnp.take(a, ret_idx, axis=0) for a in dataset)

    @jax.jit
    def get_batch_without_replacement(i, batchifier_state):
        """ Fetches the next batch for the current epoch.

        :param i: The number of the batch in the epoch.
        :param batchifier_state: The initialized state returned by init.
        :return: the batch
        """
        rng_key = batchifier_state
        batch_rng_key = rng_suite.fold_in(rng_key, i)
        ret_idx = sample_from_array(batch_rng_key, jnp.arange(num_records), batch_size, 0, rng_suite=rng_suite)
        return tuple(jnp.take(a, ret_idx, axis=0) for a in dataset)

    return init, get_batch_with_replacement if with_replacement else get_batch_without_replacement


def split_batchify_data(dataset, batch_size=None, q=None, rng_suite=strong_rng):
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
        :return: the batch
        """
        ret_idx = jax.lax.dynamic_slice_in_dim(idxs, i * batch_size, batch_size)
        return tuple(jnp.take(a, ret_idx, axis=0) for a in dataset)

    return init, get_batch


def q_to_batch_size(q, N):
    """ Returns the batch size for a given subsampling ratio q. """
    return int(N * q)


def batch_size_to_q(batch_size, N):
    """ Returns the subsampling ratio q corresponding to a given batch size. """
    return batch_size / N
