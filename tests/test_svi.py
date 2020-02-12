import unittest

import jax.numpy as np
import jax

import numpyro.distributions as dist
from numpyro.primitives import sample, param

from dppp.svi import sample_prior_predictive, sample_multi_prior_predictive, \
    sample_posterior_predictive, sample_multi_posterior_predictive

class ModelSamplingTests(unittest.TestCase):

    def test_sample_prior_predictive(self):
        def model(N, d):
            mu = sample("mu", dist.Normal(np.zeros(d)))
            x = sample("x", dist.Normal(mu), sample_shape=(N,))

        N, d = 100, 2
        rng_key = jax.random.PRNGKey(1836)
        samples = sample_prior_predictive(rng_key, model, (N, d))
        self.assertEqual((d,), np.shape(samples['mu']))
        self.assertEqual((N, d), np.shape(samples['x']))
        # crude test that samples are from model distribution (mean is within 3 times stddev)
        self.assertTrue(np.allclose(np.mean(samples['x'], axis=0), samples['mu'], atol=3/np.sqrt(N)))
        self.assertTrue(np.allclose(samples['mu'], 0, atol=3.))

    def test_sample_prior_predictive_with_substitute(self):
        def model(N, d):
            mu = sample("mu", dist.Normal(np.zeros(d)))
            x = sample("x", dist.Normal(mu), sample_shape=(N,))

        N, d = 100, 2
        mu_fixed = np.array([1., -.5])
        rng_key = jax.random.PRNGKey(235)
        samples = sample_prior_predictive(rng_key, model, (N, d), substitutes={'mu': mu_fixed})
        self.assertEqual((d,), np.shape(samples['mu']))
        self.assertEqual((N, d), np.shape(samples['x']))
        # crude test that samples are from model distribution (mean is within 3 times stddev)
        self.assertTrue(np.allclose(np.mean(samples['x'], axis=0), mu_fixed, atol=3/np.sqrt(N)))
        self.assertTrue(np.allclose(samples['mu'], mu_fixed))

    def test_sample_multi_prior_predictive(self):
        def model(N, d):
            mu = sample("mu", dist.Normal(np.zeros(d)))
            x = sample("x", dist.Normal(mu), sample_shape=(N,))

        N, d = 1, 3
        rng_key = jax.random.PRNGKey(2876)
        N_total = 100
        samples = sample_multi_prior_predictive(rng_key, N_total, model, (N, d))
        self.assertEqual((N_total, d), np.shape(samples['mu']))
        self.assertEqual((N_total, N, d), np.shape(samples['x']))
        # crude test that samples are from model distribution (mean is within 3 times stddev)
        self.assertTrue(np.allclose(np.mean(samples['x'], axis=0), 0., atol=3/np.sqrt(N_total)))
        self.assertTrue(np.allclose(np.mean(samples['mu'], axis=0), 0., atol=3/np.sqrt(N_total)))


    def test_sample_posterior_predictive(self):
        def model(N, d):
            mu = sample("mu", dist.Normal(np.zeros(d)))
            x = sample("x", dist.Normal(mu), sample_shape=(N,))

        def guide(d):
            mu_loc = param('mu_loc', np.zeros(d))
            mu = sample('mu', dist.Normal(mu_loc))

        N, d = 100, 2
        mu_loc = np.array([7., 2.12])
        rng_key = jax.random.PRNGKey(98347)
        samples = sample_posterior_predictive(rng_key, model, (N, d), guide, (d,), params={'mu_loc': mu_loc})
        self.assertEqual((d,), np.shape(samples['mu']))
        self.assertEqual((N, d), np.shape(samples['x']))
        # crude test that samples are from model distribution (mean is within 3 times stddev)
        self.assertTrue(np.allclose(np.mean(samples['x'], axis=0), samples['mu'], atol=3/np.sqrt(N)))
        self.assertTrue(np.allclose(samples['mu'], mu_loc, atol=3.))

    def test_sample_multi_posterior_predictive(self):
        def model(N, d):
            mu = sample("mu", dist.Normal(np.zeros(d)))
            x = sample("x", dist.Normal(mu), sample_shape=(N,))

        def guide(d):
            mu_loc = param('mu_loc', np.zeros(d))
            mu = sample('mu', dist.Normal(mu_loc))

        N, d = 1, 2
        N_total = 100
        mu_loc = np.array([7., 2.12])
        rng_key = jax.random.PRNGKey(98347)
        samples = sample_multi_posterior_predictive(rng_key, N_total, model, (N, d), guide, (d,), params={'mu_loc': mu_loc})
        self.assertEqual((N_total, d), np.shape(samples['mu']))
        self.assertEqual((N_total, N, d), np.shape(samples['x']))
        # crude test that samples are from model distribution (mean is within 3 times stddev)
        self.assertTrue(np.allclose(np.mean(samples['x'], axis=0), mu_loc, atol=3/np.sqrt(N_total)))
        self.assertTrue(np.allclose(np.mean(samples['mu'], axis=0), mu_loc, atol=3/np.sqrt(N_total)))

if __name__ == '__main__':
    unittest.main()




