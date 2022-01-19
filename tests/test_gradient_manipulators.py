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

""" tests that the d3p.clip_gradient method works correctly
"""
import unittest

import jax.numpy as jnp
import numpy as np

from d3p.svi import clip_gradient, full_norm, normalize_gradient


class GradientManipulatorsTests(unittest.TestCase):

    def assert_array_tuple_close(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for e, a in zip(expected, actual):
            self.assertEqual(e.shape, a.shape)
            self.assertTrue(np.allclose(a, e))

    def assert_gradient_direction(self, expected_gradient_parts, actual_gradient_parts):
        self.assertEqual(len(expected_gradient_parts), len(actual_gradient_parts))
        norm_expected = full_norm(expected_gradient_parts)
        norm_actual = full_norm(actual_gradient_parts)
        for g_expected, g_actual in zip(expected_gradient_parts, actual_gradient_parts):
            self.assertEqual(g_expected.shape, g_actual.shape)
            g_expected_normalized = g_expected / norm_expected
            g_actual_normalized = g_actual / norm_actual
            self.assertTrue(np.allclose(g_expected_normalized, g_actual_normalized))

    def assert_clipping_results(self, gradient_parts, clipped_gradient_parts, clip_threshold):
        self.assert_gradient_direction(gradient_parts, clipped_gradient_parts)
        norm_clipped = full_norm(clipped_gradient_parts)
        norm = full_norm(gradient_parts)
        self.assertLessEqual(norm_clipped, clip_threshold + 1e-6)
        self.assertLessEqual(norm_clipped, norm + 1e-6)

    def setUp(self):
        np.random.seed(0)
        self.gradient_parts = (np.random.randn(2, 5), np.random.randn(6, 4, 2))

    def test_full_norm_is_correct(self):
        norm = full_norm(self.gradient_parts)
        expectedNorm = jnp.sqrt(
            jnp.sum(jnp.array(tuple(jnp.sum(jnp.square(x)) for x in self.gradient_parts)))
        )
        self.assertTrue(np.allclose(expectedNorm, norm))

    def test_full_norm_deals_with_empty_input_gracefully(self):
        norm = full_norm(None)
        self.assertEqual(0, norm)
        norm = full_norm([])
        self.assertEqual(0, norm)
        norm = full_norm(())
        self.assertEqual(0, norm)

    def test_full_norm_on_jax_tree(self):
        gradient_tree = (
            jnp.ones(shape=(17, 2, 3)),
            jnp.ones(shape=(2, 54)),
            (jnp.ones(shape=(2, 3)), jnp.ones(shape=(3, 4, 5))),
            ()
        )
        norm = full_norm(gradient_tree)
        expectedNorm = 16.613247
        self.assertTrue(np.allclose(expectedNorm, norm))

    def test_clip_gradient_gives_input_when_threshold_equals_norm(self):
        clip_threshold = full_norm(self.gradient_parts)
        clipped_gradient_parts = clip_gradient(self.gradient_parts, clip_threshold)
        self.assert_array_tuple_close(self.gradient_parts, clipped_gradient_parts)

    def test_clip_gradient_clips_when_threshold_is_less_than_norm(self):
        clip_threshold = 0.1 * full_norm(self.gradient_parts)
        clipped_gradient_parts = clip_gradient(self.gradient_parts, clip_threshold)
        self.assert_clipping_results(self.gradient_parts, clipped_gradient_parts, clip_threshold)

    def test_clip_gradient_gives_input_when_threshold_exceeds_norm(self):
        clip_threshold = 2 * full_norm(self.gradient_parts)
        clipped_gradient_parts = clip_gradient(self.gradient_parts, clip_threshold)
        self.assert_array_tuple_close(self.gradient_parts, clipped_gradient_parts)

    def test_clip_gradient_gives_input_when_threshold_is_infinite(self):
        clip_threshold = jnp.inf
        clipped_gradient_parts = clip_gradient(self.gradient_parts, clip_threshold)
        self.assert_array_tuple_close(self.gradient_parts, clipped_gradient_parts)

    def test_clip_gradient_rejects_zero_threshold(self):
        with self.assertRaises(ValueError):
            clip_gradient(self.gradient_parts, 0.)

    def test_normalize_gradient(self):
        normalized_gradient_parts = normalize_gradient(self.gradient_parts)
        self.assert_gradient_direction(self.gradient_parts, normalized_gradient_parts)
        normalized_norm = full_norm(normalized_gradient_parts)
        self.assertTrue(np.allclose(1., normalized_norm))


if __name__ == '__main__':
    unittest.main()
