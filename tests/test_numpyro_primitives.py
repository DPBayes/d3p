import unittest
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.primitives
import numpyro.handlers
import numpyro.distributions as dists

from d3p.numpyro_primitives import mask_observed

from numpyro.infer.elbo import log_density

class MaskObservedTests(unittest.TestCase):

    def test_mask_observed_simple(self):
        def model(x):
            with numpyro.primitives.plate("plate", len(x)):
                numpyro.primitives.sample("x", dists.Normal(0, 1).expand((2,)).to_event(1), obs=x)

        x = np.random.randn(10, 2)
        mask = np.arange(len(x)) < 6
        masked_model = mask_observed(model, mask)

        t = numpyro.handlers.trace(numpyro.handlers.seed(masked_model, jax.random.PRNGKey(0))).get_trace(x)
        assert "x" in t
        assert t['x']['type'] == "sample"
        assert t['x']['is_observed']

        expected_scale = len(x) / np.sum(mask)
        self.assertAlmostEqual(expected_scale, t['x']['scale'])

        expected_logprob = dists.Normal(0, 1).expand((2,)).to_event(1).log_prob(x) * mask
        logprob = t['x']['fn'].log_prob(t['x']['value'])
        self.assertTrue(np.allclose(expected_logprob, logprob))

        log_dens, _ = log_density(numpyro.handlers.seed(masked_model, jax.random.PRNGKey(0)), (x,), dict(), dict())
        expected_log_dens = expected_logprob.sum() * expected_scale
        self.assertAlmostEqual(expected_log_dens, log_dens, places=5)

    def test_mask_observed_pre_scaled(self):
        subsample_size = 4
        def model(x):
            with numpyro.primitives.plate("plate", len(x), subsample_size=subsample_size) as ind:
                numpyro.primitives.sample("x", dists.Normal(0, 1).expand((2,)).to_event(1), obs=x[ind])

        x = np.random.randn(10, 2)
        pre_scale = len(x)/subsample_size
        mask = np.arange(subsample_size) >= subsample_size - 1
        masked_model = mask_observed(model, mask)

        t = numpyro.handlers.trace(numpyro.handlers.seed(masked_model, jax.random.PRNGKey(0))).get_trace(x)
        assert "x" in t
        assert t['x']['type'] == "sample"
        assert t['x']['is_observed']

        expected_scale = pre_scale * (subsample_size / np.sum(mask))
        self.assertAlmostEqual(expected_scale, t['x']['scale'])

        expected_logprob = dists.Normal(0, 1).expand((2,)).to_event(1).log_prob(t['x']['value']) * mask
        logprob = t['x']['fn'].log_prob(t['x']['value'])
        self.assertTrue(np.allclose(expected_logprob, logprob))

        log_dens, _ = log_density(numpyro.handlers.seed(masked_model, jax.random.PRNGKey(0)), (x,), dict(), dict())
        expected_log_dens = expected_logprob.sum() * expected_scale
        self.assertAlmostEqual(expected_log_dens, log_dens, places=5)

    def test_mask_observed_with_non_affected(self):
        # NOTE: this test confirms current expected behaviour, in which mask_observed cannot properly handle per-sample latent variables
        subsample_size = 4
        def model(x):
            mu = numpyro.primitives.sample("mu", dists.Normal(0, 1))
            with numpyro.primitives.plate("plate", len(x), subsample_size=subsample_size) as ind:
                z = numpyro.primitives.sample("z", dists.Normal(mu, 1))
                z_expanded = numpyro.primitives.deterministic("z_expanded", jnp.repeat(z, 2).reshape(-1, 2))
                numpyro.primitives.sample("x", dists.Normal(z_expanded, 1).to_event(1), obs=x[ind])

        x = np.random.randn(10, 2)
        pre_scale = len(x)/subsample_size
        mask = np.arange(subsample_size) >= subsample_size - 1
        masked_model = mask_observed(model, mask)

        t = numpyro.handlers.trace(numpyro.handlers.seed(masked_model, jax.random.PRNGKey(0))).get_trace(x)

        assert "mu" in t
        assert t['mu']['type'] == "sample"
        assert not t['mu']['is_observed']
        self.assertTrue(t['mu']['scale'] is None or np.isclose(1., t['mu']['scale']))
        
        mu = t['mu']['value']
        expected_logprob_mu = dists.Normal(0, 1).log_prob(mu)
        self.assertAlmostEqual(expected_logprob_mu, t['mu']['fn'].log_prob(mu))

        assert "z" in t
        assert t['z']['type'] == "sample"
        assert not t['z']['is_observed']

        self.assertAlmostEqual(pre_scale, t['z']['scale'])

        z = t['z']['value']
        expected_logprob_z = dists.Normal(mu, 1).expand((subsample_size,)).log_prob(z)
        self.assertTrue(np.allclose(expected_logprob_z, t['z']['fn'].log_prob(z)))

        assert "z_expanded" in t
        z = t['z_expanded']['value']

        assert "x" in t
        assert t['x']['type'] == "sample"
        assert t['x']['is_observed']

        expected_scale = pre_scale * (subsample_size / np.sum(mask))
        self.assertAlmostEqual(expected_scale, t['x']['scale'])

        expected_logprob_x = dists.Normal(z, 1).to_event(1).log_prob(t['x']['value']) * mask
        logprob = t['x']['fn'].log_prob(t['x']['value'])
        self.assertTrue(np.allclose(expected_logprob_x, logprob))

        log_dens, _ = log_density(numpyro.handlers.seed(masked_model, jax.random.PRNGKey(0)), (x,), dict(), dict())
        expected_log_dens = expected_logprob_x.sum() * expected_scale + expected_logprob_z.sum() * pre_scale + expected_logprob_mu.sum()
        self.assertAlmostEqual(expected_log_dens, log_dens, places=5)

    @unittest.expectedFailure
    def test_mask_observed_with_non_affected_desired(self):
        # NOTE: this test would confirm desired behaviour in which mask_observed correctly handles per-sample latent variables
        subsample_size = 4
        def model(x):
            mu = numpyro.primitives.sample("mu", dists.Normal(0, 1))
            with numpyro.primitives.plate("plate", len(x), subsample_size=subsample_size) as ind:
                z = numpyro.primitives.sample("z", dists.Normal(mu, 1))
                z_expanded = numpyro.primitives.deterministic("z_expanded", jnp.repeat(z, 2).reshape(-1, 2))
                numpyro.primitives.sample("x", dists.Normal(z_expanded, 1).to_event(1), obs=x[ind])

        x = np.random.randn(10, 2)
        pre_scale = len(x)/subsample_size
        mask = np.arange(subsample_size) >= subsample_size - 1
        masked_model = mask_observed(model, mask)

        t = numpyro.handlers.trace(numpyro.handlers.seed(masked_model, jax.random.PRNGKey(0))).get_trace(x)

        assert "mu" in t
        assert t['mu']['type'] == "sample"
        assert not t['mu']['is_observed']
        self.assertTrue(t['mu']['scale'] is None or np.isclose(1., t['mu']['scale']))
        mu = t['mu']['value']
        expected_logprob_mu = dists.Normal(0, 1).log_prob(mu)
        self.assertAlmostEqual(expected_logprob_mu, t['mu']['fn'].log_prob(mu))

        assert "z" in t
        assert t['z']['type'] == "sample"
        assert not t['z']['is_observed']

        expected_scale = pre_scale * (subsample_size / np.sum(mask))
        self.assertAlmostEqual(expected_scale, t['z']['scale'])

        z = t['z']['value']        
        expected_logprob_z = dists.Normal(mu, 1).expand((subsample_size,)).log_prob(z) * mask
        self.assertTrue(np.allclose(expected_logprob_z, t['z']['fn'].log_prob(z)))

        assert "z_expanded" in t
        z = t['z_expanded']['value']

        assert "x" in t
        assert t['x']['type'] == "sample"
        assert t['x']['is_observed']

        self.assertAlmostEqual(expected_scale, t['x']['scale'])

        expected_logprob_x = dists.Normal(z, 1).to_event(1).log_prob(t['x']['value']) * mask
        logprob = t['x']['fn'].log_prob(t['x']['value'])
        self.assertTrue(np.allclose(expected_logprob_x, logprob))

        log_dens, _ = log_density(numpyro.handlers.seed(masked_model, jax.random.PRNGKey(0)), (x,), dict(), dict())
        expected_log_dens = expected_logprob_x.sum() * expected_scale + expected_logprob_z.sum() * expected_scale + expected_logprob_mu.sum()
        self.assertAlmostEqual(expected_log_dens, log_dens, places=5)

    def test_mask_observed_all_masked(self):
        def model(x):
            with numpyro.primitives.plate("plate", len(x)):
                numpyro.primitives.sample("x", dists.Normal(0, 1).expand((2,)).to_event(1), obs=x)

        x = np.random.randn(10, 2)
        mask = np.zeros(len(x), dtype=bool)
        masked_model = mask_observed(model, mask)

        t = numpyro.handlers.trace(numpyro.handlers.seed(masked_model, jax.random.PRNGKey(0))).get_trace(x)
        assert "x" in t
        assert t['x']['type'] == "sample"
        assert t['x']['is_observed']

        expected_logprob = np.zeros(len(x))
        logprob = t['x']['fn'].log_prob(t['x']['value'])
        self.assertTrue(np.allclose(expected_logprob, logprob))

        log_dens, _ = log_density(numpyro.handlers.seed(masked_model, jax.random.PRNGKey(0)), (x,), dict(), dict())
        self.assertEqual(0., log_dens)
