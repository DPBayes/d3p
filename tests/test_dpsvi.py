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
import numpyro
from numpyro.optim import SGD
import d3p.random
import d3p.random.debug
import numpy as np

from d3p.svi import DPSVI, DPSVIState, full_norm
from tests.util import are_trees_close


class DPSVITestBase:

    def get_rng_suite(self):
        raise NotImplementedError()

    def setUp(self):
        self.rng_suite = self.get_rng_suite()
        self.rng = self.rng_suite.PRNGKey(9782346)
        self.batch_size = 10
        self.num_elements = 8
        self.mask = jnp.arange(self.batch_size) < self.num_elements
        self.rescale_factor = self.batch_size / self.num_elements
        self.num_obs_total = 100
        self.px_grads = (
            jnp.ones((self.batch_size, 10000)),
            jnp.ones((self.batch_size, 10000))
        )
        self.masked_px_grads = tuple(
            g * self.mask.reshape(-1, 1) for g in self.px_grads
        )
        self.px_loss = jnp.arange(self.batch_size, dtype=jnp.float32) * self.mask
        self.dp_scale = 1.
        self.clipping_threshold = 2.
        self.optim = SGD(1.)
        self.svi = DPSVI(
            None, None, self.optim, None, self.clipping_threshold,
            self.dp_scale, num_obs_total=self.num_obs_total,
            rng_suite=self.rng_suite
        )

    def test_init(self):
        def model(X):
            mu = numpyro.sample("mu", numpyro.distributions.Normal(np.zeros((X.shape[1],))))
            with numpyro.plate("plate", size=self.num_obs_total, subsample_size=X.shape[0]):
                numpyro.sample("X", numpyro.distributions.Normal(mu, 1.).to_event(1), obs=X)

        guide = numpyro.infer.autoguide.AutoDiagonalNormal(model)

        dpsvi = DPSVI(
            model, guide, self.optim, numpyro.infer.Trace_ELBO(),
            self.clipping_threshold, self.dp_scale, rng_suite=self.rng_suite
        )
        svi = numpyro.infer.SVI(model, guide, self.optim, numpyro.infer.Trace_ELBO())

        batch = (jnp.zeros((self.batch_size, 3)),)

        dpsvi_state = dpsvi.init(self.rng, *batch)
        svi_state = svi.init(self.rng_suite.convert_to_jax_rng_key(self.rng), *batch)

        self.assertEqual(self.num_obs_total, dpsvi_state.observation_scale)
        self.assertTrue(np.allclose(self.rng, dpsvi_state.rng_key))

        self.assertEqual(jax.tree_util.tree_structure(svi_state.optim_state), jax.tree_util.tree_structure(dpsvi_state.optim_state))

    def test_init_no_unscaling(self):
        def model(X):
            mu = numpyro.sample("mu", numpyro.distributions.Normal(np.zeros((X.shape[1],))))
            with numpyro.plate("plate", size=self.num_obs_total, subsample_size=X.shape[0]):
                numpyro.sample("X", numpyro.distributions.Normal(mu, 1.).to_event(1), obs=X)

        guide = numpyro.infer.autoguide.AutoDiagonalNormal(model)

        dpsvi = DPSVI(
            model, guide, self.optim, numpyro.infer.Trace_ELBO(),
            self.clipping_threshold, self.dp_scale, rng_suite=self.rng_suite, clip_unscaled_observations=False
        )
        svi = numpyro.infer.SVI(model, guide, self.optim, numpyro.infer.Trace_ELBO())

        batch = (jnp.zeros((self.batch_size, 3)),)

        dpsvi_state = dpsvi.init(self.rng, *batch)
        svi_state = svi.init(self.rng_suite.convert_to_jax_rng_key(self.rng), *batch)

        self.assertEqual(1., dpsvi_state.observation_scale)
        self.assertTrue(np.allclose(self.rng, dpsvi_state.rng_key))

        self.assertEqual(jax.tree_util.tree_structure(svi_state.optim_state), jax.tree_util.tree_structure(dpsvi_state.optim_state))

    def test_compute_px_gradients_masking(self):
        def model(X):
            mu = numpyro.sample("mu", numpyro.distributions.Normal(np.zeros((X.shape[1],))))
            with numpyro.plate("plate", size=self.num_obs_total, subsample_size=X.shape[0]):
                numpyro.sample("X", numpyro.distributions.Normal(mu, 1.).to_event(1), obs=X)

        guide = numpyro.infer.autoguide.AutoDiagonalNormal(model)

        batch = (jnp.ones((self.batch_size, 3)),)

        svi = DPSVI(
            model, guide, self.optim, numpyro.infer.Trace_ELBO(),
            self.clipping_threshold, self.dp_scale, rng_suite=self.rng_suite
        )
        svi_state = svi.init(self.rng, *batch)

        new_svi_state, px_losses, px_grads, num_elements, batch_mask_scaling_factor = \
            svi._compute_per_example_gradients(svi_state, svi_state.rng_key, *batch, mask=self.mask)
        
        self.assertTrue(are_trees_close(svi_state.optim_state, new_svi_state.optim_state))
        self.assertEqual(svi_state.observation_scale, new_svi_state.observation_scale)
        
        self.assertEqual(self.num_elements, num_elements)
        self.assertAlmostEqual(self.batch_size / self.num_elements, batch_mask_scaling_factor)

        self.assertFalse(np.allclose(px_losses[:self.num_elements], 0.))
        self.assertTrue(np.allclose(px_losses[self.num_elements:], 0.))

        self.assertFalse(np.allclose(px_grads["auto_loc"][:self.num_elements], 0.))
        self.assertTrue(np.allclose(px_grads["auto_loc"][self.num_elements:], 0.))

        self.assertFalse(np.allclose(px_grads["auto_scale"][:self.num_elements], 0.))
        self.assertTrue(np.allclose(px_grads["auto_scale"][self.num_elements:], 0.))

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
            jax.tree_util.tree_structure(clipped_px_grads),
            jax.tree_util.tree_structure(px_grads)
        )

        clipped_px_norms = jax.vmap(full_norm)(clipped_px_grads)
        expected_clipped_px_norms = np.array([2., np.sqrt(2)])
        self.assertTrue(np.allclose(clipped_px_norms, expected_clipped_px_norms))

        _, clipped_grad = self.svi._combine_gradients(clipped_px_grads, jnp.ones((2,)))
        clipped_norm = full_norm(clipped_grad)
        self.assertTrue(clipped_norm < self.clipping_threshold)

    def test_px_gradient_aggregation(self):
        np.random.seed(0)
        px_grads, _ = jax.tree_util.tree_flatten((
            np.random.normal(1, 1, size=(self.batch_size, 10000)),
            np.random.normal(1, 1, size=(self.batch_size, 10000))
        ))

        expected_grads_list = [jnp.mean(px_grads, axis=0) for px_grads in jax.tree_util.tree_leaves(px_grads)]
        expected_loss = jnp.mean(self.px_loss)

        loss, grads = self.svi._combine_gradients(px_grads, self.px_loss)

        self.assertTrue(np.allclose(expected_loss, loss), f"expected loss {expected_loss} but was {loss}")
        self.assertTrue(
            np.allclose(expected_grads_list, jax.tree_util.tree_leaves(grads)),
            f"expected gradients {expected_grads_list} but was {grads}"
        )

    def test_dp_noise_perturbation(self):
        svi_state = DPSVIState(None, self.rng, .3)

        grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), self.px_grads)
        masked_grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), self.masked_px_grads)

        new_svi_state, perturbed_grads = \
            self.svi._perturb_and_reassemble_gradients(
                svi_state, self.rng, masked_grads, self.num_elements, self.rescale_factor
            )

        self.assertIs(svi_state.optim_state, new_svi_state.optim_state)
        self.assertEqual(jax.tree_util.tree_structure(grads), jax.tree_util.tree_structure(perturbed_grads))

        corrected_observation_scale = svi_state.observation_scale * self.rescale_factor
        expected_std = self.dp_scale * (self.clipping_threshold / self.num_elements) * corrected_observation_scale
        for perturbed_site, site in zip(jax.tree_util.tree_leaves(perturbed_grads), jax.tree_util.tree_leaves(grads)):
            self.assertEqual(perturbed_site.shape, site.shape)
            self.assertTrue(
                np.allclose(expected_std, jnp.std(perturbed_site), atol=1e-2),
                f"expected stdev {expected_std} but was {jnp.std(perturbed_site)}"
            )
            # self.assertTrue(np.allclose(jnp.mean(site), jnp.mean(perturbed_site), atol=1e-2))
            self.assertAlmostEqual(jnp.mean(site) * svi_state.observation_scale, jnp.mean(perturbed_site), places=2)

    def test_dp_noise_perturbation_not_deterministic_over_rngs(self):
        """ verifies that different randomness is used in subsequent calls """
        svi_state = DPSVIState(None, self.rng, .3)
        first_rng, second_rng = self.rng_suite.split(self.rng)

        grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), self.px_grads)

        new_svi_state, first_grads = \
            self.svi._perturb_and_reassemble_gradients(
                svi_state, first_rng, grads, self.num_elements, self.rescale_factor
            )

        _, second_grads = \
            self.svi._perturb_and_reassemble_gradients(
                new_svi_state, second_rng, grads, self.num_elements, self.rescale_factor
            )

        some_gradient_noise_is_equal = reduce(
            lambda are_equal, acc: are_equal or acc,
            jax.tree_util.tree_leaves(
                jax.tree_util.tree_map(
                    lambda x, y: jnp.allclose(x, y), first_grads, second_grads
                )
            )
        )
        self.assertFalse(some_gradient_noise_is_equal)

    def test_dp_noise_perturbation_not_deterministic_over_sites(self):
        """ verifies that different randomness is used for different gradient sites """
        svi_state = DPSVIState(None, self.rng, .3)

        grads = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), self.px_grads)

        _, grads = \
            self.svi._perturb_and_reassemble_gradients(
                svi_state, self.rng, grads, self.num_elements, self.rescale_factor
            )

        noise_sites = jax.tree_util.tree_leaves(grads)

        self.assertFalse(np.allclose(noise_sites[0], noise_sites[1]))


class DPSVIStrongRNGTests(DPSVITestBase, unittest.TestCase):

    def get_rng_suite(self):
        return d3p.random


class DPSVIDebugRNGTests(DPSVITestBase, unittest.TestCase):

    def get_rng_suite(self):
        return d3p.random.debug


if __name__ == '__main__':
    unittest.main()
