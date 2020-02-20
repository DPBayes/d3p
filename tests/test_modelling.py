import unittest

import jax.numpy as np
import jax

import numpyro.distributions as dist
from numpyro.primitives import sample, param

from dppp.modelling import sample_prior_predictive, sample_multi_prior_predictive, \
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

    class DistWithIntermediate(dist.Distribution):

        def sample_with_intermediates(self, rng_key, sample_shape=()):
            interm = np.ones(sample_shape + (2,))
            samples = jax.random.randint(rng_key, sample_shape, 0, 100)
            return samples, [interm]

        def sample(self, rng_key, sample_shape=()):
            raise NotImplementedError()

    def test_sample_prior_predictive_without_intermediates(self):

        def model(N, d):
            x = sample("x", self.DistWithIntermediate(), sample_shape=(N, d))

        N, d = 100, 2
        rng_key = jax.random.PRNGKey(3781)
        samples = sample_prior_predictive(rng_key, model, (N, d))
        self.assertEqual((N, d), np.shape(samples['x']))

    def test_sample_prior_predictive_with_intermediates(self):

        def model(N, d):
            x = sample("x", self.DistWithIntermediate(), sample_shape=(N, d))

        N, d = 100, 2
        rng_key = jax.random.PRNGKey(2)
        samples = sample_prior_predictive(rng_key, model, (N, d), with_intermediates=True)
        self.assertEqual((N, d), np.shape(samples['x'][0]))
        self.assertEqual(2, len(samples['x']))
        self.assertEqual(1, len(samples['x'][1]))
        self.assertEqual((N, d, 2), np.shape(samples['x'][1][0]))

    def test_sample_multi_prior_predictive(self):
        def model(N, d):
            mu = sample("mu", dist.Normal(np.zeros(d)))
            x = sample("x", dist.Normal(mu), sample_shape=(N,))

        N, d = 1, 3
        rng_key = jax.random.PRNGKey(375)
        N_total = 100
        samples = sample_multi_prior_predictive(rng_key, N_total, model, (N, d))
        self.assertEqual((N_total, d), np.shape(samples['mu']))
        self.assertEqual((N_total, N, d), np.shape(samples['x']))
        # crude test that samples are from model distribution (mean is within 3 times stddev)
        self.assertTrue(np.allclose(np.mean(samples['x'], axis=0), 0., atol=3/np.sqrt(N_total)))
        self.assertTrue(np.allclose(np.mean(samples['mu'], axis=0), 0., atol=3/np.sqrt(N_total)))

    def test_sample_multi_prior_predictive_without_intermediates(self):
        def model(N, d):
            x = sample("x", self.DistWithIntermediate(), sample_shape=(N, d))

        N, d = 1, 2
        rng_key = jax.random.PRNGKey(43)
        N_total = 100
        samples = sample_multi_prior_predictive(rng_key, N_total, model, (N, d))
        self.assertEqual((N_total, N, d), np.shape(samples['x']))

    def test_sample_multi_prior_predictive_with_intermediates(self):
        def model(N, d):
            x = sample("x", self.DistWithIntermediate(), sample_shape=(N, d))

        N, d = 1, 2
        rng_key = jax.random.PRNGKey(46875)
        N_total = 100
        samples = sample_multi_prior_predictive(rng_key, N_total, model, (N, d), with_intermediates=True)
        self.assertEqual((N_total, N, d), np.shape(samples['x'][0]))
        self.assertEqual(2, len(samples['x']))
        self.assertEqual(1, len(samples['x'][1]))
        self.assertEqual((N_total, N, d, 2), np.shape(samples['x'][1][0]))

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


    def test_sample_posterior_predictive_without_intermediates(self):

        def model(N, d):
            x = sample("x", self.DistWithIntermediate(), sample_shape=(N, d))

        def guide(d):
            mu_loc = param('mu_loc', np.zeros(1))
            mu = sample('mu', self.DistWithIntermediate(), sample_shape=(1, d))

        N, d = 100, 2
        rng_key = jax.random.PRNGKey(275)
        samples = sample_posterior_predictive(rng_key, model, (N, d), guide, (d,), params={'mu_loc': 1.})
        self.assertEqual((N, d), np.shape(samples['x']))
        self.assertEqual((1, d), np.shape(samples['mu']))

    def test_sample_posterior_predictive_with_intermediates(self):

        def model(N, d):
            x = sample("x", self.DistWithIntermediate(), sample_shape=(N, d))

        def guide(d):
            mu_loc = param('mu_loc', np.zeros(1))
            mu = sample('mu', self.DistWithIntermediate(), sample_shape=(1, d))

        N, d = 100, 2
        rng_key = jax.random.PRNGKey(7234)
        samples = sample_posterior_predictive(rng_key, model, (N, d), guide, (d,), params={'mu_loc': 1.}, with_intermediates=True)
        self.assertEqual((N, d), np.shape(samples['x'][0]))
        self.assertEqual(2, len(samples['x']))
        self.assertEqual(1, len(samples['x'][1]))
        self.assertEqual((N, d, 2), np.shape(samples['x'][1][0]))

        self.assertEqual((1, d), np.shape(samples['mu'][0]))
        self.assertEqual(2, len(samples['mu']))
        self.assertEqual(1, len(samples['mu'][1]))
        self.assertEqual((1, d, 2), np.shape(samples['mu'][1][0]))

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

    def test_sample_multi_posterior_predictive_without_intermediates(self):

        def model(N, d):
            x = sample("x", self.DistWithIntermediate(), sample_shape=(N, d))

        def guide(d):
            mu_loc = param('mu_loc', np.zeros(1))
            mu = sample('mu', self.DistWithIntermediate(), sample_shape=(1, d))

        N, d = 10, 2
        rng_key = jax.random.PRNGKey(35786)
        N_total = 10
        samples = sample_multi_posterior_predictive(rng_key, N_total, model, (N, d), guide, (d,), params={'mu_loc': 1.})
        self.assertEqual((N_total, N, d), np.shape(samples['x']))
        self.assertEqual((N_total, 1, d), np.shape(samples['mu']))

    def test_sample_multi_posterior_predictive_with_intermediates(self):

        def model(N, d):
            x = sample("x", self.DistWithIntermediate(), sample_shape=(N, d))

        def guide(d):
            mu_loc = param('mu_loc', np.zeros(1))
            mu = sample('mu', self.DistWithIntermediate(), sample_shape=(1, d))

        N, d = 10, 2
        rng_key = jax.random.PRNGKey(35786)
        N_total = 10
        samples = sample_multi_posterior_predictive(rng_key, N_total, model, (N, d), guide, (d,), params={'mu_loc': 1.}, with_intermediates=True)
        self.assertEqual((N_total, N, d), np.shape(samples['x'][0]))
        self.assertEqual(2, len(samples['x']))
        self.assertEqual(1, len(samples['x'][1]))
        self.assertEqual((N_total, N, d, 2), np.shape(samples['x'][1][0]))

        self.assertEqual((N_total, 1, d), np.shape(samples['mu'][0]))
        self.assertEqual(2, len(samples['mu']))
        self.assertEqual(1, len(samples['mu'][1]))
        self.assertEqual((N_total, 1, d, 2), np.shape(samples['mu'][1][0]))

if __name__ == '__main__':
    unittest.main()




