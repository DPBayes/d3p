# Copyright 2019- d3p Developers and their Assignees

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" tests that the minibatch context manager leads to correct scaling of the
affected sample sites in the numpyro.log_density method
"""
import unittest

import jax.numpy as jnp
import jax
import numpy as np

from d3p.minibatch import split_batchify_data, subsample_batchify_data


class SplitBatchifierTests(unittest.TestCase):

    def test_split_batchify_init(self):
        data = jnp.arange(0, 100)
        init, fetch = split_batchify_data((data,), 10)

        rng_key = jax.random.PRNGKey(0)
        num_batches, batchifier_state = init(rng_key)

        self.assertEqual(10, num_batches)
        self.assertEqual(jnp.size(data), jnp.size(batchifier_state))
        self.assertTrue(np.allclose(np.unique(batchifier_state), data))

    def test_split_batchify_init_non_divisiable_size(self):
        data = jnp.arange(0, 105)
        init, fetch = split_batchify_data((data,), 10)

        rng_key = jax.random.PRNGKey(0)
        num_batches, batchifier_state = init(rng_key)

        self.assertEqual(10, num_batches)
        self.assertTrue(np.alltrue(np.unique(batchifier_state, return_counts=True)[1] < 2))

    def test_split_batchify_fetch(self):
        data = np.arange(105) + 100
        init, fetch = split_batchify_data((data,), 10)
        batchifier_state = jax.random.permutation(jax.random.PRNGKey(0), jnp.arange(0, 105))
        num_batches = 10

        counts = np.zeros(105)
        for i in range(num_batches):
            batch = fetch(i, batchifier_state)
            batch = batch[0]
            unq_idxs, unq_counts = np.unique(batch, return_counts=True)
            counts[unq_idxs - 100] = unq_counts
            # ensure each item occurs at most once in the batch
            self.assertTrue(np.alltrue(unq_counts <= 1))
            # ensure batch was plausibly drawn from data
            self.assertTrue(np.alltrue(batch >= 100) and np.alltrue(batch < 205))

        # ensure each item occurs at most once in the epoch
        self.assertTrue(np.alltrue(counts <= 1))
        # ensure that amount of elements in batches cover an epoch worth of data
        self.assertEqual(100, np.sum(counts))

    def test_split_batchify_batches_differ(self):
        data = np.arange(105) + 100
        init, fetch = split_batchify_data((data,), 10)
        num_batches, batchifier_state = init(jax.random.PRNGKey(10))

        batch_0 = fetch(3, batchifier_state)
        batch_1 = fetch(8, batchifier_state)
        self.assertFalse(np.allclose(batch_0, batch_1))  # ensure batches are different

    def test_split_batchify_fetch_correct_shape(self):
        data = np.random.normal(size=(105, 3))
        init, fetch = split_batchify_data((data,), 10)
        batchifier_state = jax.random.permutation(
            jax.random.PRNGKey(0), jnp.arange(0, 105)
        )

        batch = fetch(6, batchifier_state)
        batch = batch[0]
        self.assertEqual((10, 3), jnp.shape(batch))


class SubsamplingBatchifierTests(unittest.TestCase):

    def test_subsample_batchify_init(self):
        data = jnp.arange(0, 100)
        init, fetch = subsample_batchify_data((data,), 10)

        rng_key = jax.random.PRNGKey(0)
        num_batches, batchifier_state = init(rng_key)

        self.assertEqual(10, num_batches)
        self.assertTrue(np.allclose(rng_key, batchifier_state))

    def test_subsample_batchify_init_non_divisiable_size(self):
        data = jnp.arange(0, 105)
        init, fetch = subsample_batchify_data((data,), 10)

        rng_key = jax.random.PRNGKey(0)
        num_batches, batchifier_state = init(rng_key)

        self.assertEqual(10, num_batches)
        self.assertTrue(np.allclose(rng_key, batchifier_state))

    def test_subsample_batchify_fetch_without_replacement(self):
        data = np.arange(105) + 100
        init, fetch = subsample_batchify_data((data,), 10)
        batchifier_state = jax.random.PRNGKey(2)
        num_batches = 10

        for i in range(num_batches):
            batch = fetch(i, batchifier_state)
            batch = batch[0]
            _, unq_counts = np.unique(batch, return_counts=True)
            # ensure each item occurs at most once in the batch
            self.assertTrue(np.alltrue(unq_counts <= 1))
            # ensure batch was plausibly drawn from data
            self.assertTrue(np.alltrue(batch >= 100) and np.alltrue(batch < 205))

    def test_subsample_batchify_fetch_batches_differ_without_replacement(self):
        data = np.arange(105) + 100
        init, fetch = subsample_batchify_data((data,), 10)
        batchifier_state = jax.random.PRNGKey(2)

        batch_0 = fetch(3, batchifier_state)
        batch_1 = fetch(8, batchifier_state)
        self.assertFalse(np.allclose(batch_0, batch_1))  # ensure batches are different

    def test_subsample_batchify_fetch_correct_shape_without_replacement(self):
        data = np.random.normal(size=(105, 3))
        init, fetch = subsample_batchify_data((data,), 10)
        batchifier_state = jax.random.PRNGKey(2)
        # num_batches = 10

        batch = fetch(6, batchifier_state)
        batch = batch[0]
        self.assertEqual((10, 3), jnp.shape(batch))

    def test_subsample_batchify_fetch_with_replacement(self):
        data = np.arange(105) + 100
        init, fetch = subsample_batchify_data((data,), 10, with_replacement=True)
        batchifier_state = jax.random.PRNGKey(2)
        num_batches = 10

        for i in range(num_batches):
            batch = fetch(i, batchifier_state)
            batch = batch[0]
            # ensure batch was plausibly drawn from data
            self.assertTrue(np.alltrue(batch >= 100) and np.alltrue(batch < 205))

    def test_subsample_batchify_fetch_batches_differ_with_replacement(self):
        data = np.arange(105) + 100
        init, fetch = subsample_batchify_data((data,), 10, with_replacement=True)
        batchifier_state = jax.random.PRNGKey(2)
        # num_batches = 10

        batch_0 = fetch(3, batchifier_state)
        batch_1 = fetch(8, batchifier_state)
        self.assertFalse(np.allclose(batch_0, batch_1))  # ensure batches are different

    def test_subsample_batchify_fetch_correct_shape_with_replacement(self):
        data = np.random.normal(size=(105, 3))
        init, fetch = subsample_batchify_data((data,), 10, with_replacement=True)
        batchifier_state = jax.random.PRNGKey(2)
        # num_batches = 10

        batch = fetch(6, batchifier_state)
        batch = batch[0]
        self.assertEqual((10, 3), jnp.shape(batch))


if __name__ == '__main__':
    unittest.main()
