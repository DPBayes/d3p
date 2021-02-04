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
from numpyro.infer.svi import SVIState

from dppp.svi import DPSVI

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
        self.svi = DPSVI(None, None, None, None, self.clipping_threshold,
            self.dp_scale, num_obs_total=self.num_obs_total
        )

    def test_dp_noise_perturbation(self):
        svi_state = SVIState(None, self.rng)

        new_svi_state, loss_val, grads = \
            self.svi._combine_and_transform_gradient(
                svi_state, self.px_grads_list, self.px_loss, self.tree_def
            )

        self.assertIs(svi_state.optim_state, new_svi_state.optim_state)
        self.assertFalse(jnp.allclose(svi_state.rng_key, new_svi_state.rng_key))
        self.assertEqual(jnp.mean(self.px_loss), loss_val)
        self.assertEqual(self.tree_def, jax.tree_structure(grads))

        batch_size = len(self.px_loss)
        expected_std = self.dp_scale * self.clipping_threshold / batch_size
        for site in jax.tree_leaves(grads):
            self.assertTrue(
                jnp.allclose(expected_std, jnp.std(site)/self.num_obs_total, atol=1e-1)
            )
            self.assertTrue(jnp.allclose(0., jnp.mean(site)/self.num_obs_total, atol=1e-1))

    def test_dp_noise_perturbation_not_deterministic_over_calls(self):
        svi_state = SVIState(None, self.rng)
        new_svi_state, _, first_grads = \
            self.svi._combine_and_transform_gradient(
                svi_state, self.px_grads_list, self.px_loss, self.tree_def
            )

        _, _, second_grads = \
            self.svi._combine_and_transform_gradient(
                new_svi_state, self.px_grads_list, self.px_loss, self.tree_def
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
        svi_state = SVIState(None, self.rng)
        _, _, grads = \
            self.svi._combine_and_transform_gradient(
                svi_state, self.px_grads_list, self.px_loss, self.tree_def
            )

        noise_sites = jax.tree_leaves(grads)

        self.assertFalse(jnp.allclose(noise_sites[0], noise_sites[1]))


if __name__ == '__main__':
    unittest.main()