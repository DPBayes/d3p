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

import numpyro.distributions as dist
from numpyro.infer.util import log_density
from numpyro.handlers import seed, trace
from numpyro.primitives import sample, deterministic

from dppp.minibatch import minibatch, split_batchify_data, subsample_batchify_data

class MinibatchTests(unittest.TestCase):

    class DummyDist(dist.Distribution):

        def sample(self, key, sample_shape=()):
            return jnp.ones(sample_shape)

        def log_prob(self, value):
            return jnp.ones_like(value)

    def test_minibatch_scale_correct_over_full_data(self):
        batch_size = 100
        num_obs_total = 100
        expected_scale = num_obs_total / batch_size

        result = minibatch(batch_size, num_obs_total=num_obs_total)

        self.assertAlmostEqual(expected_scale, result.scale)

    def test_minibatch_scale_correct_over_single_sample(self):
        batch_size = 1
        num_obs_total = 100
        expected_scale = num_obs_total / batch_size

        result = minibatch(batch_size, num_obs_total=num_obs_total)

        self.assertAlmostEqual(expected_scale, result.scale)

    def test_minibatch_scale_correct_for_true_minibatch(self):
        batch_size = 10
        num_obs_total = 100
        expected_scale = num_obs_total / batch_size

        result = minibatch(batch_size, num_obs_total=num_obs_total)

        self.assertAlmostEqual(expected_scale, result.scale)

    def test_minibatch_rejects_batch_size_none(self):
        batch_size = None
        with self.assertRaises(TypeError):
            minibatch(batch_size)

    def test_minibatch_rejects_tuple_batch_size_argument(self):
        batch_size = (2, 3)
        with self.assertRaises(TypeError):
            minibatch(batch_size)

    def test_minibatch_rejects_float_batch_size_argument(self):
        batch_size = 10.
        with self.assertRaises(TypeError):
            minibatch(batch_size)

    def test_minibatch_batch_size_deduced_from_array(self):
        batch_size = 20
        num_obs_total = 100
        expected_scale = num_obs_total / batch_size

        X = jnp.ones((20, 3))
        result = minibatch(X, num_obs_total=num_obs_total)

        self.assertAlmostEqual(expected_scale, result.scale)

    def test_minibatch_num_total_obs_not_given(self):
        batch_size = 20
        expected_scale = 1.

        result = minibatch(batch_size)

        self.assertAlmostEqual(expected_scale, result.scale)

    def test_minibatch_batch_size_deduced_from_array_and_num_total_obs_not_given(self):
        batch_size = 20
        expected_scale = 1.

        X = jnp.ones((batch_size, 3))
        result = minibatch(X)

        self.assertAlmostEqual(expected_scale, result.scale)

    def test_minibatch_fixed_batch_size_scale_correct_with_static_argnums_in_model_under_jit(self):
        # note(lumip): we need to ensure that minibatch works well under jit, however, we cannot
        #   test it separately as it returns a class (which is no longer permitted by jit apparently).
        #   We therefore test it by wrapping it in a dummy model function, get the trace and
        #   check that the scale applied in the trace is correct.

        def test_model(batch_size, num_obs_total):
            with minibatch(batch_size, num_obs_total):
                sample('test', MinibatchTests.DummyDist(), sample_shape=(batch_size,))

        minibatch_jitted = jax.jit(test_model, static_argnums=(0, 1))

        batch_size = 10
        num_obs_total = 100
        expected_scale = num_obs_total / batch_size
        tr = trace(minibatch_jitted).get_trace(batch_size, num_obs_total)
        observed_scale = tr['test']['scale']

        self.assertAlmostEqual(expected_scale, observed_scale)

    def test_minibatch_batch_size_deduced_from_batch_in_model_under_jit(self):

        def test_model(X, num_obs_total):
            with minibatch(X, num_obs_total):
                sample('test', MinibatchTests.DummyDist(), sample_shape=X.shape)

        minibatch_jitted = jax.jit(test_model, static_argnums=1)

        batch_size = 10
        num_obs_total = 100
        expected_scale = num_obs_total / batch_size

        X = jnp.ones((batch_size, 3))
        tr = trace(minibatch_jitted).get_trace(X, num_obs_total)
        observed_scale = tr['test']['scale']

        self.assertAlmostEqual(expected_scale, observed_scale)

class MinibatchIntegrationTests(unittest.TestCase):

    def setUp(self):
        def model_fn(X, N=None, num_obs_total=None):
            if N is None:
                N = jnp.shape(X)[0]
            if num_obs_total is None:
                num_obs_total = N

            mu = sample("theta", dist.Normal(1.))
            with minibatch(N, num_obs_total=num_obs_total):
                X = sample("X", dist.Normal(mu), obs=X, sample_shape=(N,))
            return X, mu

        self.model = seed(model_fn, jax.random.PRNGKey(0))
        self.num_samples = 100
        self.X, self.mu = self.model(None, self.num_samples)

    def run_minibatch_test_for_batch_size(self, batch_size):
        batch = self.X[:batch_size]
        self.assertEqual(batch_size, jnp.shape(batch)[0])

        prior_log_prob = dist.Normal(1.).log_prob(self.mu)
        data_log_prob = jnp.sum(dist.Normal(self.mu).log_prob(batch))
        expected_log_joint = prior_log_prob + (self.num_samples/batch_size) * data_log_prob

        log_joint, _ = log_density(
            self.model, (batch,),
            {'num_obs_total': self.num_samples}, {"theta": self.mu}
        )
        self.assertTrue(jnp.allclose(expected_log_joint, log_joint))

    def test_minibatch_scale_correct_over_full_data(self):
        batch_size = self.num_samples
        self.run_minibatch_test_for_batch_size(batch_size)

    def test_minibatch_scale_correct_over_single_sample(self):
        batch_size = 1
        self.run_minibatch_test_for_batch_size(batch_size)

    def test_minibatch_scale_correct_for_true_minibatch(self):
        batch_size = int(0.1 * self.num_samples)
        self.run_minibatch_test_for_batch_size(batch_size)

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
            self.assertTrue(np.alltrue(unq_counts <= 1)) # ensure each item occurs at most once in the batch
            self.assertTrue(np.alltrue(batch >= 100) and np.alltrue(batch < 205)) # ensure batch was plausibly drawn from data

        self.assertTrue(np.alltrue(counts <= 1)) # ensure each item occurs at most once in the epoch
        self.assertEqual(100, np.sum(counts)) # ensure that amount of elements in batches cover an epoch worth of data

    def test_split_batchify_batches_differ(self):
        data = np.arange(105) + 100
        init, fetch = split_batchify_data((data,), 10)
        num_batches, batchifier_state = init(jax.random.PRNGKey(10))

        batch_0 = fetch(3, batchifier_state)
        batch_1 = fetch(8, batchifier_state)
        self.assertFalse(jnp.allclose(batch_0, batch_1)) # ensure batches are different

    def test_split_batchify_fetch_correct_shape(self):
        data = np.random.normal(size=(105, 3))
        init, fetch = split_batchify_data((data,), 10)
        batchifier_state = jax.random.permutation(jax.random.PRNGKey(0), jnp.arange(0, 105))

        batch = fetch(6, batchifier_state)
        batch = batch[0]
        self.assertEqual((10,3), jnp.shape(batch))

class SubsamplingBatchifierTests(unittest.TestCase):

    def test_subsample_batchify_init(self):
        data = jnp.arange(0, 100)
        init, fetch = subsample_batchify_data((data,), 10)

        rng_key = jax.random.PRNGKey(0)
        num_batches, batchifier_state = init(rng_key)

        self.assertEqual(10, num_batches)
        self.assertTrue(jnp.allclose(rng_key, batchifier_state))

    def test_subsample_batchify_init_non_divisiable_size(self):
        data = jnp.arange(0, 105)
        init, fetch = subsample_batchify_data((data,), 10)

        rng_key = jax.random.PRNGKey(0)
        num_batches, batchifier_state = init(rng_key)

        self.assertEqual(10, num_batches)
        self.assertTrue(jnp.allclose(rng_key, batchifier_state))

    def test_subsample_batchify_fetch_without_replacement(self):
        data = np.arange(105) + 100
        init, fetch = subsample_batchify_data((data,), 10)
        batchifier_state = jax.random.PRNGKey(2)
        num_batches = 10

        for i in range(num_batches):
            batch = fetch(i, batchifier_state)
            batch = batch[0]
            _, unq_counts = np.unique(batch, return_counts=True)
            self.assertTrue(np.alltrue(unq_counts <= 1)) # ensure each item occurs at most once in the batch
            self.assertTrue(np.alltrue(batch >= 100) and np.alltrue(batch < 205)) # ensure batch was plausibly drawn from data

    def test_subsample_batchify_fetch_batches_differ_without_replacement(self):
        data = np.arange(105) + 100
        init, fetch = subsample_batchify_data((data,), 10)
        batchifier_state = jax.random.PRNGKey(2)
        # num_batches = 10

        batch_0 = fetch(3, batchifier_state)
        batch_1 = fetch(8, batchifier_state)
        self.assertFalse(jnp.allclose(batch_0, batch_1)) # ensure batches are different


    def test_subsample_batchify_fetch_correct_shape_without_replacement(self):
        data = np.random.normal(size=(105, 3))
        init, fetch = subsample_batchify_data((data,), 10)
        batchifier_state = jax.random.PRNGKey(2)
        # num_batches = 10

        batch = fetch(6, batchifier_state)
        batch = batch[0]
        self.assertEqual((10,3), jnp.shape(batch))


    def test_subsample_batchify_fetch_with_replacement(self):
        data = np.arange(105) + 100
        init, fetch = subsample_batchify_data((data,), 10, with_replacement=True)
        batchifier_state = jax.random.PRNGKey(2)
        num_batches = 10

        for i in range(num_batches):
            batch = fetch(i, batchifier_state)
            batch = batch[0]
            self.assertTrue(np.alltrue(batch >= 100) and np.alltrue(batch < 205)) # ensure batch was plausibly drawn from data

    def test_subsample_batchify_fetch_batches_differ_with_replacement(self):
        data = np.arange(105) + 100
        init, fetch = subsample_batchify_data((data,), 10, with_replacement=True)
        batchifier_state = jax.random.PRNGKey(2)
        # num_batches = 10

        batch_0 = fetch(3, batchifier_state)
        batch_1 = fetch(8, batchifier_state)
        self.assertFalse(jnp.allclose(batch_0, batch_1)) # ensure batches are different

    def test_subsample_batchify_fetch_correct_shape_with_replacement(self):
        data = np.random.normal(size=(105, 3))
        init, fetch = subsample_batchify_data((data,), 10, with_replacement=True)
        batchifier_state = jax.random.PRNGKey(2)
        # num_batches = 10

        batch = fetch(6, batchifier_state)
        batch = batch[0]
        self.assertEqual((10,3), jnp.shape(batch))


if __name__ == '__main__':
    unittest.main()
