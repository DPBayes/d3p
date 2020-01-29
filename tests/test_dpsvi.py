""" tests that the AdaDP optimizer works correctly
"""
import unittest

import jax.numpy as np
import jax
import numpy as onp
from functools import reduce

from dppp.svi import DPSVI, TunableSVI

class DPSVITest(unittest.TestCase):

    def test_dp_noise_perturbation(self):
        rng = jax.random.PRNGKey(0)

        px_grads = ((np.zeros((10, 100)), np.zeros((10, 100))))
        px_grads_list, tree_def = jax.tree_flatten(px_grads)
        px_loss = np.arange(10, dtype=np.float32)

        dp_scale = 1.
        svi = DPSVI(None, None, None, None, 1., dp_scale, rng)
        svi_state = object()
        new_svi_state, loss_val, grads = \
            svi._combine_and_transform_gradient(
                svi_state, px_grads_list, px_loss, tree_def
            )

        self.assertIs(new_svi_state, svi_state)
        self.assertEqual(np.mean(px_loss), loss_val)
        self.assertEqual(tree_def, jax.tree_structure(grads))
        for site in jax.tree_leaves(grads):
            self.assertTrue(np.allclose(dp_scale/10., np.std(site), atol=1e-2))

    def test_dp_noise_perturbation_not_deterministic_over_calls(self):
        rng = jax.random.PRNGKey(0)

        px_grads = ((np.zeros((10, 100)), np.zeros((10, 100))))
        px_grads_list, tree_def = jax.tree_flatten(px_grads)
        px_loss = np.arange(10, dtype=np.float32)

        dp_scale = 1.
        svi = DPSVI(None, None, None, None, 1., dp_scale, rng)
        svi_state = object()
        new_svi_state, _, first_grads = \
            svi._combine_and_transform_gradient(
                svi_state, px_grads_list, px_loss, tree_def
            )

        _, _, second_grads = \
            svi._combine_and_transform_gradient(
                new_svi_state, px_grads_list, px_loss, tree_def
            )

        some_gradient_noise_is_equal = reduce(lambda are_equal, acc: are_equal or acc, 
            jax.tree_leaves(
                jax.tree_multimap(
                    lambda x, y: np.allclose(x, y), first_grads, second_grads
                )
            )
        )
        self.assertFalse(some_gradient_noise_is_equal)

    def test_dp_noise_perturbation_not_deterministic_over_sites(self):
        rng = jax.random.PRNGKey(0)

        px_grads = ((np.zeros((10, 100)), np.zeros((10, 100))))
        px_grads_list, tree_def = jax.tree_flatten(px_grads)
        px_loss = np.arange(10, dtype=np.float32)

        dp_scale = 1.
        svi = DPSVI(None, None, None, None, 1., dp_scale, rng)
        svi_state = object()
        _, _, grads = \
            svi._combine_and_transform_gradient(
                svi_state, px_grads_list, px_loss, tree_def
            )

        noise_sites = jax.tree_leaves(grads)

        self.assertFalse(np.allclose(noise_sites[0], noise_sites[1]))


if __name__ == '__main__':
    unittest.main()