""" tests that the dppp.clip_gradient method works correctly
"""
import os

# allow example to find dppp without installing
import sys
sys.path.append(os.path.dirname(sys.path[0]))
#### 


import jax.numpy as np
import jax
import numpy as onp

from dppp.svi import clip_gradient, full_norm

def assert_clipping_results(gradient_parts, clipped_gradient_parts, clip_threshold):
    assert(len(clipped_gradient_parts) == len(gradient_parts))
    norm = full_norm(gradient_parts)
    norm_clipped = full_norm(clipped_gradient_parts)
    for g, g_clipped in zip(gradient_parts, clipped_gradient_parts):
        assert(g.shape == g_clipped.shape)
        g_normalized = g / norm
        g_clipped_normalized = g_clipped / norm_clipped
        assert(np.allclose(g_normalized, g_clipped_normalized))
    

def assert_array_tuple_close(expected, actual):
    assert(len(expected) == len(actual))
    for e, a in zip(expected, actual):
        assert(e.shape == a.shape)
        assert(np.allclose(a, e))


onp.random.seed(0)
gradient_parts = (onp.random.randn(2, 5), onp.random.randn(6, 4, 2))
norm = full_norm(gradient_parts)
expectedNorm = np.sqrt(np.sum(tuple(np.sum(np.square(x)) for x in gradient_parts)))
assert(np.allclose(expectedNorm, norm))


clip_threshold = norm
clipped_gradient_parts = clip_gradient(gradient_parts, clip_threshold)
assert_array_tuple_close(gradient_parts, clipped_gradient_parts)

clip_threshold = 0.1 * norm
clipped_gradient_parts = clip_gradient(gradient_parts, clip_threshold)
assert_clipping_results(gradient_parts, clipped_gradient_parts, clip_threshold)

clip_threshold = 2 * norm
clipped_gradient_parts = clip_gradient(gradient_parts, clip_threshold)
assert_array_tuple_close(gradient_parts, clipped_gradient_parts)

clip_threshold = np.inf
clipped_gradient_parts = clip_gradient(gradient_parts, clip_threshold)
assert_array_tuple_close(gradient_parts, clipped_gradient_parts)

print("success")