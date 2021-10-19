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

""" tests that the components of DPSVI class work as expected
"""
import unittest

from functools import reduce

import jax.numpy as jnp
import jax
from numpyro.optim import SGD
import numpy as np

from d3p.svi import DPSVI, DPSVIState

class DPSVITest(unittest.TestCase):

    def setUp(self):
        self.rng = jax.random.PRNGKey(9782346)
        self.batch_size = 10
        self.num_obs_total = 100
        self.px_grads = ((
            jnp.zeros((self.batch_size, 10000)),
            jnp.zeros((self.batch_size, 10000))
        ))
        self.px_grads_list, self.tree_def = jax.tree_flatten(self.px_grads)
        self.px_loss = jnp.arange(self.batch_size, dtype=jnp.float32)
        self.dp_scale = 1.
        self.clipping_threshold = 2.
        optim = SGD(1.)
        self.svi = DPSVI(None, None, optim, None, self.clipping_threshold,
            self.dp_scale, num_obs_total=self.num_obs_total
        )

    def test_px_gradient_aggregation(self):
        svi_state = DPSVIState(None, self.rng, .3)

        np.random.seed(0)
        px_grads_list, _testMethodDoc = jax.tree_flatten((
            np.random.normal(1, 1, size=(self.batch_size, 10000)),
            np.random.normal(1, 1, size=(self.batch_size, 10000))
        ))

        expected_grads_list = [jnp.mean(px_grads, axis=0) for px_grads in px_grads_list]
        expected_loss = jnp.mean(self.px_loss)

        loss, grads_list = self.svi._combine_gradients(px_grads_list, self.px_loss)

        self.assertTrue(np.allclose(expected_loss, loss), f"expected loss {expected_loss} but was {loss}")
        self.assertTrue(np.allclose(expected_grads_list, grads_list), f"expected gradients {expected_grads_list} but was {grads_list}")

    def test_dp_noise_perturbation(self):
        svi_state = DPSVIState(None, self.rng, .3)

        grads_list = [jnp.mean(px_grads, axis=0) for px_grads in self.px_grads_list]

        new_svi_state, grads = \
            self.svi._perturb_and_reassemble_gradients(
                svi_state, grads_list, self.batch_size, self.tree_def
            )

        self.assertIs(svi_state.optim_state, new_svi_state.optim_state)
        self.assertFalse(np.allclose(svi_state.rng_key, new_svi_state.rng_key))
        self.assertEqual(self.tree_def, jax.tree_structure(grads))

        expected_std = (self.dp_scale * self.clipping_threshold / self.batch_size) * svi_state.observation_scale
        for site, px_site in zip(jax.tree_leaves(grads), jax.tree_leaves(self.px_grads_list)):
            self.assertEqual(px_site.shape[1:], site.shape)
            self.assertTrue(
                np.allclose(expected_std, jnp.std(site), atol=1e-2), f"expected stdev {expected_std} but was {jnp.std(site)}"
            )
            self.assertTrue(np.allclose(0., jnp.mean(site), atol=1e-2))

    def test_dp_noise_perturbation_not_deterministic_over_calls(self):
        """ verifies that different randomness is used in subsequent calls """
        svi_state = DPSVIState(None, self.rng, .3)

        grads_list = [jnp.mean(px_grads, axis=0) for px_grads in self.px_grads_list]

        new_svi_state, first_grads = \
            self.svi._perturb_and_reassemble_gradients(
                svi_state, grads_list, self.batch_size, self.tree_def
            )

        _, second_grads = \
            self.svi._perturb_and_reassemble_gradients(
                new_svi_state, grads_list, self.batch_size, self.tree_def
            )

        some_gradient_noise_is_equal = reduce(lambda are_equal, acc: are_equal or acc,
            jax.tree_leaves(
                jax.tree_multimap(
                    lambda x, y: jnp.allclose(x, y), first_grads, second_grads
                )
            )
        )
        self.assertFalse(some_gradient_noise_is_equal)

    def test_dp_noise_perturbation_not_deterministic_over_sites(self):
        """ verifies that different randomness is used for different gradient sites """
        svi_state = DPSVIState(None, self.rng, .3)

        grads_list = [jnp.mean(px_grads, axis=0) for px_grads in self.px_grads_list]

        _, grads = \
            self.svi._perturb_and_reassemble_gradients(
                svi_state, grads_list, self.batch_size, self.tree_def
            )

        noise_sites = jax.tree_leaves(grads)

        self.assertFalse(np.allclose(noise_sites[0], noise_sites[1]))


if __name__ == '__main__':
    unittest.main()