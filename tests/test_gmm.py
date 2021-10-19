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

import unittest

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np
from numpyro.distributions import Normal

from d3p.gmm import GaussianMixture

class GaussianMixtureTests(unittest.TestCase):

    def test_rejects_non_simplex_pis(self):
        locs = jnp.array([[-5., -5.], [0., 0.], [5., 5.]])
        scales = jnp.ones_like(locs) * 0.1
        pis = jnp.ones(3)
        with self.assertRaises(ValueError):
            GaussianMixture(locs, scales, pis, validate_args=True)

    def test_sample_shape_correct(self):
        locs = jnp.array([[-5., -5.], [0., 0.], [5., 5.]])
        scales = jnp.ones_like(locs) * 0.1
        pis = jnp.ones(3)/3
        mix = GaussianMixture(locs, scales, pis)

        rng = jax.random.PRNGKey(2963)
        vals = mix.sample(rng)
        self.assertEqual((2,), jnp.shape(vals))

    def test_sample_batch_shape_correct(self):
        locs = jnp.array([[-5., -5.], [0., 0.], [5., 5.]])
        scales = jnp.ones_like(locs) * 0.1
        pis = jnp.ones(3)/3
        mix = GaussianMixture(locs, scales, pis)

        rng = jax.random.PRNGKey(2963)
        vals = mix.sample(rng, sample_shape=(5, 4))
        self.assertEqual((5, 4, 2), jnp.shape(vals))

    def test_sample_with_intermediates(self):
        locs = jnp.array([[-5., -5.], [0., 0.], [5., 5.]])
        scales = jnp.ones_like(locs) * 0.1
        pis = jnp.array([.5, .3, .2])
        mix = GaussianMixture(locs, scales, pis)

        rng = jax.random.PRNGKey(2963)
        n_total = 1000
        vals, interm = mix.sample_with_intermediates(rng, sample_shape=(10, n_total//10))
        zs = interm[0]

        self.assertEqual((10, n_total//10), jnp.shape(zs))
        self.assertEqual((10, n_total//10, 2), jnp.shape(vals))

        self.assertTrue(jnp.alltrue(zs >= 0) and jnp.alltrue(zs < 3))

        unq_vals, unq_counts = np.unique(zs, return_counts=True)
        unq_counts = unq_counts / n_total
        # crude test that samples are from mixture (ratio of samples per component is plausible)
        pis_stddev = jnp.sqrt(pis*(1-pis)/n_total)
        self.assertTrue(np.allclose(unq_counts, pis, atol=3*pis_stddev))

        for i in range(3):
            # crude test that samples are from mixture (mean is within 3 times stddev per component)
            self.assertTrue(np.allclose(locs[i], jnp.mean(vals[zs == i], axis=0), atol=3*scales[i]/jnp.sqrt(n_total)))

    def test_log_prob(self):
        locs = jnp.array([[-5., -5.], [0., 0.], [5., 5.]])
        scales = jnp.ones_like(locs) * 0.1
        pis = jnp.array([.5, .3, .2])
        mix = GaussianMixture(locs, scales, pis)

        x = np.array([[-4, -3], [1, .5]])

        log_pis = jnp.reshape(jnp.log(pis), (3,1))

        log_phis = jnp.array([Normal(locs[0], scales[0]).log_prob(x),
                   Normal(locs[1], scales[1]).log_prob(x),
                   Normal(locs[2], scales[2]).log_prob(x)])
        log_phis = jnp.sum(log_phis, axis=-1)

        expected = logsumexp(log_pis + log_phis, axis=0)

        actual = mix.log_prob(x)

        self.assertEqual((2,), jnp.shape(expected))
        self.assertTrue(np.allclose(expected, actual), "expected {}, actual {}".format(expected, actual))


if __name__ == '__main__':
    unittest.main()
