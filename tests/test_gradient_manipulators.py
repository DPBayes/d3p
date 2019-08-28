""" tests that the dppp.clip_gradient method works correctly
"""
import unittest

import jax.numpy as np
import jax
import numpy as onp

from dppp.svi import clip_gradient, full_norm

class GradientManipulatorsTests(unittest.TestCase):

    def assert_array_tuple_close(self, expected, actual):
        self.assertEqual(len(expected), len(actual))
        for e, a in zip(expected, actual):
            self.assertEqual(e.shape, a.shape)
            self.assertTrue(np.allclose(a, e))

    def assert_clipping_results(self, gradient_parts, clipped_gradient_parts, clip_threshold):
        self.assertEqual(len(clipped_gradient_parts), len(gradient_parts))
        norm = full_norm(gradient_parts)
        norm_clipped = full_norm(clipped_gradient_parts)
        for g, g_clipped in zip(gradient_parts, clipped_gradient_parts):
            self.assertEqual(g.shape, g_clipped.shape)
            g_normalized = g / norm
            g_clipped_normalized = g_clipped / norm_clipped
            self.assertTrue(np.allclose(g_normalized, g_clipped_normalized))

    def setUp(self):
        onp.random.seed(0)
        self.gradient_parts = (onp.random.randn(2, 5), onp.random.randn(6, 4, 2))

    def test_full_norm_is_correct(self):
        norm = full_norm(self.gradient_parts)
        expectedNorm = np.sqrt(
            np.sum(tuple(np.sum(np.square(x)) for x in self.gradient_parts))
        )
        self.assertTrue(np.allclose(expectedNorm, norm))

    def test_full_norm_deals_with_empty_input_gracefully(self):
        norm = full_norm(None)
        self.assertEqual(0, norm)
        norm = full_norm([])
        self.assertEqual(0, norm)

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
        clip_threshold = np.inf
        clipped_gradient_parts = clip_gradient(self.gradient_parts, clip_threshold)
        self.assert_array_tuple_close(self.gradient_parts, clipped_gradient_parts)

    def test_clip_gradient_rejects_zero_threshold(self):
        with self.assertRaises(ValueError):
            clip_gradient(self.gradient_parts, 0.)


if __name__ == '__main__':
    unittest.main()