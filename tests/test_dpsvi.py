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

""" tests that the components of DPSVI class work as expected
"""
import unittest

from functools import reduce

import jax.numpy as jnp
import jax
from numpyro.optim import SGD
import d3p.random
import d3p.random.debug
import numpy as np

from d3p.svi import DPSVI, DPSVIState, full_norm


class DPSVITestBase:

    def get_rng_suite(self):
        raise NotImplementedError()

    def setUp(self):
        self.rng_suite = self.get_rng_suite()
        self.rng = self.rng_suite.PRNGKey(9782346)
        self.batch_size = 10
        self.num_obs_total = 100
        self.px_grads = (
            jnp.zeros((self.batch_size, 10000)),
            jnp.zeros((self.batch_size, 10000))
        )
        self.px_loss = jnp.arange(self.batch_size, dtype=jnp.float32)
        self.dp_scale = 1.
        self.clipping_threshold = 2.
        optim = SGD(1.)
        self.svi = DPSVI(
            None, None, optim, None, self.clipping_threshold,
            self.dp_scale, num_obs_total=self.num_obs_total,
            rng_suite=self.rng_suite
        )

    def test_px_gradient_clipping(self):
        svi_state = DPSVIState(None, self.rng, 0.8)

        px_grads = (
            jnp.repeat(jnp.array([1., 0]), 10).reshape(2, 10),
            jnp.repeat(jnp.array([0., 1.]), 2).reshape(2, 2)
        )

        px_norms = jax.vmap(full_norm)(px_grads)
        expected_px_norms = (np.sqrt(10), np.sqrt(2))
        self.assertTrue(np.allclose(px_norms, expected_px_norms))

        new_svi_state, clipped_px_grads = \
            self.svi._clip_gradients(svi_state, px_grads)

        self.assertEqual(new_svi_state, svi_state)
        self.assertEqual(
            jax.tree_structure(clipped_px_grads),
            jax.tree_structure(px_grads)
        )

        clipped_px_norms = jax.vmap(full_norm)(clipped_px_grads)
        expected_clipped_px_norms = np.array([2., np.sqrt(2)])
        self.assertTrue(np.allclose(clipped_px_norms, expected_clipped_px_norms))

        _, clipped_grad = self.svi._combine_gradients(clipped_px_grads, jnp.ones((2,)))
        clipped_norm = full_norm(clipped_grad)
        self.assertTrue(clipped_norm < self.clipping_threshold)

    def test_px_gradient_aggregation(self):
        np.random.seed(0)
        px_grads, _ = jax.tree_flatten((
            np.random.normal(1, 1, size=(self.batch_size, 10000)),
            np.random.normal(1, 1, size=(self.batch_size, 10000))
        ))

        expected_grads_list = [jnp.mean(px_grads, axis=0) for px_grads in jax.tree_leaves(px_grads)]
        expected_loss = jnp.mean(self.px_loss)

        loss, grads = self.svi._combine_gradients(px_grads, self.px_loss)

        self.assertTrue(np.allclose(expected_loss, loss), f"expected loss {expected_loss} but was {loss}")
        self.assertTrue(
            np.allclose(expected_grads_list, jax.tree_leaves(grads)),
            f"expected gradients {expected_grads_list} but was {grads}"
        )

    def test_dp_noise_perturbation(self):
        svi_state = DPSVIState(None, self.rng, .3)

        grads = jax.tree_map(lambda x: jnp.mean(x, axis=0), self.px_grads)

        new_svi_state, perturbed_grads = \
            self.svi._perturb_and_reassemble_gradients(
                svi_state, self.rng, grads, self.batch_size
            )

        self.assertIs(svi_state.optim_state, new_svi_state.optim_state)
        self.assertEqual(jax.tree_structure(grads), jax.tree_structure(perturbed_grads))

        expected_std = self.dp_scale * (self.clipping_threshold / self.batch_size) * svi_state.observation_scale
        for perturbed_site, site in zip(jax.tree_leaves(perturbed_grads), jax.tree_leaves(grads)):
            self.assertEqual(perturbed_site.shape, site.shape)
            self.assertTrue(
                np.allclose(expected_std, jnp.std(perturbed_site), atol=1e-2),
                f"expected stdev {expected_std} but was {jnp.std(perturbed_site)}"
            )
            self.assertTrue(np.allclose(jnp.mean(site), jnp.mean(perturbed_site), atol=1e-2))

    def test_dp_noise_perturbation_not_deterministic_over_rngs(self):
        """ verifies that different randomness is used in subsequent calls """
        svi_state = DPSVIState(None, self.rng, .3)
        first_rng, second_rng = self.rng_suite.split(self.rng)

        grads = jax.tree_map(lambda x: jnp.mean(x, axis=0), self.px_grads)

        new_svi_state, first_grads = \
            self.svi._perturb_and_reassemble_gradients(
                svi_state, first_rng, grads, self.batch_size
            )

        _, second_grads = \
            self.svi._perturb_and_reassemble_gradients(
                new_svi_state, second_rng, grads, self.batch_size
            )

        some_gradient_noise_is_equal = reduce(
            lambda are_equal, acc: are_equal or acc,
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

        grads = jax.tree_map(lambda x: jnp.mean(x, axis=0), self.px_grads)

        _, grads = \
            self.svi._perturb_and_reassemble_gradients(
                svi_state, self.rng, grads, self.batch_size
            )

        noise_sites = jax.tree_leaves(grads)

        self.assertFalse(np.allclose(noise_sites[0], noise_sites[1]))


class DPSVIStrongRNGTests(DPSVITestBase, unittest.TestCase):

    def get_rng_suite(self):
        return d3p.random


class DPSVIDebugRNGTests(DPSVITestBase, unittest.TestCase):

    def get_rng_suite(self):
        return d3p.random.debug


if __name__ == '__main__':
    unittest.main()
