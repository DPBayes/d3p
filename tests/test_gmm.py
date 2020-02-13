import unittest

import jax
import jax.numpy as np
from jax.scipy.special import logsumexp
import numpy as onp
from numpyro.distributions import Normal

from dppp.gmm import GaussianMixture

class GaussianMixtureTests(unittest.TestCase):

    def test_rejects_non_simplex_pis(self):
        locs = np.array([[-5., -5.], [0., 0.], [5., 5.]])
        scales = np.ones_like(locs) * 0.1
        pis = np.ones(3)
        with self.assertRaises(ValueError):
            GaussianMixture(locs, scales, pis, validate_args=True)

    def test_sample_shape_correct(self):
        locs = np.array([[-5., -5.], [0., 0.], [5., 5.]])
        scales = np.ones_like(locs) * 0.1
        pis = np.ones(3)/3
        mix = GaussianMixture(locs, scales, pis)

        rng = jax.random.PRNGKey(2963)
        vals = mix.sample(rng)
        self.assertEqual((2,), np.shape(vals))

    def test_sample_batch_shape_correct(self):
        locs = np.array([[-5., -5.], [0., 0.], [5., 5.]])
        scales = np.ones_like(locs) * 0.1
        pis = np.ones(3)/3
        mix = GaussianMixture(locs, scales, pis)

        rng = jax.random.PRNGKey(2963)
        vals = mix.sample(rng, sample_shape=(5, 4))
        self.assertEqual((5, 4, 2), np.shape(vals))

    def test_sample_with_intermediates(self):
        locs = np.array([[-5., -5.], [0., 0.], [5., 5.]])
        scales = np.ones_like(locs) * 0.1
        pis = np.array([.5, .3, .2])
        mix = GaussianMixture(locs, scales, pis)

        rng = jax.random.PRNGKey(2963)
        n_total = 1000
        vals, interm = mix.sample_with_intermediates(rng, sample_shape=(10, n_total//10))
        zs = interm[0]

        self.assertEqual((10, n_total//10), np.shape(zs))
        self.assertEqual((10, n_total//10, 2), np.shape(vals))

        self.assertTrue(np.alltrue(zs >= 0) and np.alltrue(zs < 3))

        unq_vals, unq_counts = onp.unique(zs, return_counts=True)
        unq_counts = unq_counts / n_total
        # crude test that samples are from mixture (ratio of samples per component is plausible)
        pis_stddev = np.sqrt(pis*(1-pis)/n_total)
        self.assertTrue(np.allclose(unq_counts, pis, atol=3*pis_stddev))

        for i in range(3):
            # crude test that samples are from mixture (mean is within 3 times stddev per component)
            self.assertTrue(np.allclose(locs[i], np.mean(vals[zs == i], axis=0), atol=3*scales[i]/np.sqrt(n_total)))

    def test_log_prob(self):
        locs = np.array([[-5., -5.], [0., 0.], [5., 5.]])
        scales = np.ones_like(locs) * 0.1
        pis = np.array([.5, .3, .2])
        mix = GaussianMixture(locs, scales, pis)

        x = onp.array([[-4, -3], [1, .5]])

        log_pis = np.reshape(np.log(pis), (3,1))

        log_phis = np.array([Normal(locs[0], scales[0]).log_prob(x),
                   Normal(locs[1], scales[1]).log_prob(x),
                   Normal(locs[2], scales[2]).log_prob(x)])
        log_phis = np.sum(log_phis, axis=-1)

        expected = logsumexp(log_pis + log_phis, axis=0)

        actual = mix.log_prob(x)

        self.assertEqual((2,), np.shape(expected))
        self.assertTrue(np.allclose(expected, actual), "expected {}, actual {}".format(expected, actual))


if __name__ == '__main__':
    unittest.main()
