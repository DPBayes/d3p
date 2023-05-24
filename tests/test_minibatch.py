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

import unittest

import jax.numpy as jnp
import jax
import numpy as np
from scipy.stats import poisson, chisquare

from d3p.minibatch import split_batchify_data, subsample_batchify_data, poisson_batchify_data
import d3p.random
import d3p.random.debug


class SplitBatchifierTestsBase:

    def test_split_batchify_init(self):
        data = jnp.arange(0, 100)
        init, _ = split_batchify_data((data,), 10, rng_suite=self.rng_suite)

        rng_key = self.rng_suite.PRNGKey(0)
        num_batches, batchifier_state = init(rng_key)

        self.assertEqual(10, num_batches)
        self.assertEqual(jnp.size(data), jnp.size(batchifier_state))
        self.assertTrue(np.allclose(np.unique(batchifier_state), data))

    def test_split_batchify_init_non_divisiable_size(self):
        data = jnp.arange(0, 105)
        init, _ = split_batchify_data((data,), 10, rng_suite=self.rng_suite)

        rng_key = self.rng_suite.PRNGKey(0)
        num_batches, batchifier_state = init(rng_key)

        self.assertEqual(10, num_batches)
        self.assertTrue(np.alltrue(np.unique(batchifier_state, return_counts=True)[1] < 2))

    def test_split_batchify_fetch(self):
        data = np.arange(105) + 100
        _, fetch = split_batchify_data((data,), 10, rng_suite=self.rng_suite)
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
        init, fetch = split_batchify_data((data,), 10, rng_suite=self.rng_suite)
        _, batchifier_state = init(self.rng_suite.PRNGKey(10))

        batch_0 = fetch(3, batchifier_state)
        batch_1 = fetch(8, batchifier_state)
        self.assertFalse(np.allclose(batch_0, batch_1))  # ensure batches are different

    def test_split_batchify_fetch_correct_shape(self):
        data = np.random.normal(size=(105, 3))
        _, fetch = split_batchify_data((data,), 10, rng_suite=self.rng_suite)
        batchifier_state = jax.random.permutation(jax.random.PRNGKey(0), jnp.arange(0, 105))

        batch = fetch(6, batchifier_state)
        batch = batch[0]
        self.assertEqual((10, 3), jnp.shape(batch))


class SplitBatchifierStrongRNGTests(SplitBatchifierTestsBase, unittest.TestCase):

    def setUp(self) -> None:
        self.rng_suite = d3p.random


class SplitBatchifierDebugRNGTests(SplitBatchifierTestsBase, unittest.TestCase):

    def setUp(self) -> None:
        self.rng_suite = d3p.random.debug


class SubsamplingBatchifierTestsBase:

    def test_subsample_batchify_init(self):
        data = jnp.arange(0, 100)
        init, _ = subsample_batchify_data((data,), 10, rng_suite=self.rng_suite)

        rng_key = self.rng_suite.PRNGKey(0)
        num_batches, batchifier_state = init(rng_key)

        self.assertEqual(10, num_batches)
        self.assertTrue(np.allclose(rng_key, batchifier_state))

    def test_subsample_batchify_init_non_divisiable_size(self):
        data = jnp.arange(0, 105)
        init, _ = subsample_batchify_data((data,), 10, rng_suite=self.rng_suite)

        rng_key = self.rng_suite.PRNGKey(0)
        num_batches, batchifier_state = init(rng_key)

        self.assertEqual(10, num_batches)
        self.assertTrue(np.allclose(rng_key, batchifier_state))

    def test_subsample_batchify_fetch_without_replacement(self):
        data = np.arange(105) + 100
        _, fetch = subsample_batchify_data((data,), 10, rng_suite=self.rng_suite)
        batchifier_state = self.rng_suite.PRNGKey(2)
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
        _, fetch = subsample_batchify_data((data,), 10, rng_suite=self.rng_suite)
        batchifier_state = self.rng_suite.PRNGKey(2)
        # num_batches = 10

        batch_0 = fetch(3, batchifier_state)
        batch_1 = fetch(8, batchifier_state)
        self.assertFalse(np.allclose(batch_0, batch_1))  # ensure batches are different

    def test_subsample_batchify_fetch_correct_shape_without_replacement(self):
        data = np.random.normal(size=(105, 3))
        _, fetch = subsample_batchify_data((data,), 10, rng_suite=self.rng_suite)
        batchifier_state = self.rng_suite.PRNGKey(2)
        # num_batches = 10

        batch = fetch(6, batchifier_state)
        batch = batch[0]
        self.assertEqual((10, 3), jnp.shape(batch))

    def test_subsample_batchify_fetch_with_replacement(self):
        data = np.arange(105) + 100
        _, fetch = subsample_batchify_data((data,), 10, with_replacement=True, rng_suite=self.rng_suite)
        batchifier_state = self.rng_suite.PRNGKey(2)
        num_batches = 10

        for i in range(num_batches):
            batch = fetch(i, batchifier_state)
            batch = batch[0]
            # ensure batch was plausibly drawn from data
            self.assertTrue(np.alltrue(batch >= 100) and np.alltrue(batch < 205))

    def test_subsample_batchify_fetch_batches_differ_with_replacement(self):
        data = np.arange(105) + 100
        _, fetch = subsample_batchify_data((data,), 10, with_replacement=True, rng_suite=self.rng_suite)
        batchifier_state = self.rng_suite.PRNGKey(2)
        # num_batches = 10

        batch_0 = fetch(3, batchifier_state)
        batch_1 = fetch(8, batchifier_state)
        self.assertFalse(np.allclose(batch_0, batch_1))  # ensure batches are different

    def test_subsample_batchify_fetch_correct_shape_with_replacement(self):
        data = np.random.normal(size=(105, 3))
        _, fetch = subsample_batchify_data((data,), 10, with_replacement=True, rng_suite=self.rng_suite)
        batchifier_state = self.rng_suite.PRNGKey(2)
        # num_batches = 10

        batch = fetch(6, batchifier_state)
        batch = batch[0]
        self.assertEqual((10, 3), jnp.shape(batch))


class SubsamplingBatchifierStrongRNGTests(SubsamplingBatchifierTestsBase, unittest.TestCase):

    def setUp(self) -> None:
        self.rng_suite = d3p.random


class SubsamplingBatchifierDebugRNGTests(SubsamplingBatchifierTestsBase, unittest.TestCase):

    def setUp(self) -> None:
        self.rng_suite = d3p.random.debug


class PoissonBatchifierTestsBase:

    def test_poisson_batchify_init(self):
        data = jnp.arange(0, 100)
        init, _ = poisson_batchify_data((data,), q=.1, max_batch_size=20, rng_suite=self.rng_suite)

        rng_key = self.rng_suite.PRNGKey(0)
        num_batches, batchifier_state = init(rng_key)

        self.assertEqual(10, num_batches)
        self.assertTrue(np.allclose(rng_key, batchifier_state))

    def test_poisson_batchify_fetch(self):
        N = 105
        data = np.arange(N) + 100
        q = .1
        _, fetch = poisson_batchify_data((data,), q=q, max_batch_size=N, rng_suite=self.rng_suite)
        batchifier_state = self.rng_suite.PRNGKey(2)

        num_trials = 1000
        size_counts = np.zeros(N, dtype=np.int32)

        for i in range(num_trials):
            batch, mask = fetch(i, batchifier_state)
            self.assertIsInstance(batch, tuple)
            self.assertEqual(len(mask), len(batch[0]))
            size = np.sum(mask)
            size_counts[size] += 1
            batch = batch[0][mask]
            _, unq_counts = np.unique(batch, return_counts=True)
            # ensure each item occurs at most once in the batch
            self.assertTrue(np.alltrue(unq_counts <= 1))
            # ensure batch was plausibly drawn from data
            self.assertTrue(np.alltrue(batch >= 100) and np.alltrue(batch < 205))

        # check that retrieved batch sizes follows Poisson distribution
        size_frequencies = size_counts / num_trials
        expected_frequencies = poisson(q * N).pmf(np.arange(N, dtype=np.int32))
        val = chisquare(size_frequencies, expected_frequencies)
        self.assertTrue(val.pvalue >= 0.05)

    def test_poisson_batchify_fetch_batches_differ(self):
        N = 105
        data = np.arange(N) + 100
        q = .1
        _, fetch = poisson_batchify_data((data,), q=q, max_batch_size=N, rng_suite=self.rng_suite)
        batchifier_state = self.rng_suite.PRNGKey(2)

        batch_0, _ = fetch(3, batchifier_state)
        batch_1, _ = fetch(8, batchifier_state)

        self.assertFalse(np.allclose(batch_0, batch_1))  # ensure batches are different

    def test_poisson_batchify_fetch_correct_shape(self):
        N = 105
        data = (np.random.normal(size=(N, 3)) + 100, np.random.normal(size=(N,)))
        q = .1
        _, fetch = poisson_batchify_data(data, q=q, max_batch_size=N//2, rng_suite=self.rng_suite)
        batchifier_state = self.rng_suite.PRNGKey(2)

        batch, mask = fetch(6, batchifier_state)
        self.assertIsInstance(batch, tuple)
        self.assertEqual(2, len(batch))
        self.assertEqual(len(mask), len(batch[0]))
        self.assertEqual(len(mask), len(batch[1]))
        self.assertEqual((N//2, 3), jnp.shape(batch[0]))
        self.assertEqual((N//2, ), jnp.shape(batch[1]))

    def test_poisson_batchify_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            poisson_batchify_data(None, .1, 10, rng_suite=self.rng_suite)
            poisson_batchify_data(jnp.zeros(11,), 1.1, 10, rng_suite=self.rng_suite)
            poisson_batchify_data(jnp.zeros(11,), -.1, 10, rng_suite=self.rng_suite)
            poisson_batchify_data(jnp.zeros(11,), .2, -1, rng_suite=self.rng_suite)

    def test_poisson_batchify_oversize_ignore(self):
        N = 105
        data = (np.arange(N), np.random.normal(size=(N, 3)))
        q = .3
        max_batch_size = 3
        _, fetch = poisson_batchify_data(data, q=q, max_batch_size=max_batch_size, handle_oversized_batch='truncate', rng_suite=self.rng_suite)
        batchifier_state = self.rng_suite.PRNGKey(2)

        batch, mask = fetch(0, batchifier_state)
        self.assertEqual(max_batch_size, np.sum(mask))
        self.assertTrue(np.all((batch[0] < N) & (batch[0] >= 0)))
        self.assertTrue([
            np.any(np.all(data[1] == batch[1][i], axis=-1)) for i in range(max_batch_size)]
        )
    
    def test_poisson_batchify_oversize_suppress(self):
        N = 105
        data = (np.arange(N), np.random.normal(size=(N, 3)))
        q = .3
        max_batch_size = 3
        _, fetch = poisson_batchify_data(data, q=q, max_batch_size=max_batch_size, handle_oversized_batch='suppress', rng_suite=self.rng_suite)
        batchifier_state = self.rng_suite.PRNGKey(2)

        _, mask = fetch(0, batchifier_state)
        self.assertEqual(0, np.sum(mask))

    def test_poisson_batchify_max_batch_size_float(self):
        N = 105
        data = (np.arange(N) + 100,)
        q = .3
        max_batch_size = .9
        _, fetch = poisson_batchify_data(data, q=q, max_batch_size=max_batch_size, handle_oversized_batch='suppress', rng_suite=self.rng_suite)
        batchifier_state = self.rng_suite.PRNGKey(2)

        batch, _ = fetch(0, batchifier_state)
        expected_batch_size = 39
        self.assertEqual(expected_batch_size, len(batch[0]))
        

class PoissonBatchifierStrongRNGTests(PoissonBatchifierTestsBase, unittest.TestCase):

    def setUp(self) -> None:
        self.rng_suite = d3p.random


class PoissonBatchifierDebugRNGTests(PoissonBatchifierTestsBase, unittest.TestCase):

    def setUp(self) -> None:
        self.rng_suite = d3p.random.debug


if __name__ == '__main__':
    unittest.main()
