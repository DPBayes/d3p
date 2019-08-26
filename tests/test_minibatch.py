""" tests that the minibatch context manager leads to correct scaling of the
affected sample sites in the numpyro.log_density method
"""
import unittest

import jax.numpy as np
import jax
import numpy as onp

import numpyro.distributions as dist
from numpyro.svi import elbo
from numpyro.infer_util import log_density
from numpyro.handlers import seed
from numpyro.primitives import sample

from dppp.svi import minibatch

class MinibatchTests(unittest.TestCase):

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
        with self.assertRaises(ValueError):
            minibatch(batch_size)

    def test_minibatch_rejects_tuple_batch_size_argument(self):
        batch_size = (2, 3)
        with self.assertRaises(ValueError):
            minibatch(batch_size)

    def test_minibatch_rejects_float_batch_size_argument(self):
        batch_size = 10.
        with self.assertRaises(ValueError):
            minibatch(batch_size)

    def test_minibatch_batch_size_deduced_from_array(self):
        batch_size = 20
        num_obs_total = 100
        expected_scale = num_obs_total / batch_size

        X = np.ones((20, 3))
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

        X = np.ones((batch_size, 3))
        result = minibatch(X)

        self.assertAlmostEqual(expected_scale, result.scale)

    def test_minibatch_reject_fixed_batch_size_under_jit(self):
        minibatch_jitted = jax.jit(minibatch, static_argnums=1)

        batch_size = 10
        num_obs_total = 100

        with self.assertRaises(ValueError):
            minibatch_jitted(batch_size, num_obs_total)

    def test_minibatch_fixed_batch_size_scale_correct_with_static_argnums_under_jit(self):
        minibatch_jitted = jax.jit(minibatch, static_argnums=(0, 1))

        batch_size = 10
        num_obs_total = 100
        expected_scale = num_obs_total / batch_size

        result = minibatch_jitted(batch_size, num_obs_total)
        self.assertAlmostEqual(expected_scale, result.scale)

    def test_minibatch_batch_size_deduced_from_batch_under_jit(self):
        minibatch_jitted = jax.jit(minibatch, static_argnums=1)

        batch_size = 10
        num_obs_total = 100
        expected_scale = num_obs_total / batch_size

        X = np.ones((batch_size, 3))
        result = minibatch_jitted(X, num_obs_total)

        self.assertAlmostEqual(expected_scale, result.scale)

class MinibatchIntegrationTests(unittest.TestCase):

    def setUp(self):
        def model_fn(X, N=None, num_obs_total=None):
            if N is None:
                N = np.shape(X)[0]
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
        self.assertEqual(batch_size, np.shape(batch)[0])

        prior_log_prob = dist.Normal(1.).log_prob(self.mu)
        data_log_prob = np.sum(dist.Normal(self.mu).log_prob(batch))
        expected_log_joint = prior_log_prob + (self.num_samples/batch_size) * data_log_prob

        log_joint, _ = log_density(
            self.model, (batch,),
            {'num_obs_total': self.num_samples}, {"theta": self.mu}
        )
        self.assertTrue(np.allclose(expected_log_joint, log_joint))

    def test_minibatch_scale_correct_over_full_data(self):
        batch_size = self.num_samples
        self.run_minibatch_test_for_batch_size(batch_size)

    def test_minibatch_scale_correct_over_single_sample(self):
        batch_size = 1
        self.run_minibatch_test_for_batch_size(batch_size)

    def test_minibatch_scale_correct_for_true_minibatch(self):
        batch_size = int(0.1 * self.num_samples)
        self.run_minibatch_test_for_batch_size(batch_size)


if __name__ == '__main__':
    unittest.main()
