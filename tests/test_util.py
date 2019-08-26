"""tests the implementations in the dppp.svi utils package
"""
import os

# allow example to find dppp without installing
import sys
sys.path.append(os.path.dirname(sys.path[0]))
#### 


import jax.numpy as np
import jax
import numpy as onp

from dppp import util

# test map_over_secondary_dims

x = np.array([
    [
        [.3, .4],
        [2., 1.],
        [6., 1.5]
    ],
    [
        [1., 2.],
        [2.4, -1],
        [3.2, 1.]
    ]
])
expected = np.array([
    [1.3, 2.4],
    [4.4, 0],
    [9.2, 2.5]
])

mapped_sum = util.map_over_secondary_dims(np.sum)
result = mapped_sum(x)
assert(np.allclose(expected, result))

# test is_int_scalar

assert(util.is_int_scalar(5) == True)
assert(util.is_int_scalar(0) == True)
assert(util.is_int_scalar(-3.4) == False)

assert(util.is_int_scalar(onp.asscalar(onp.array(1))) == True)
assert(util.is_int_scalar(onp.asscalar(onp.array(1.))) == False)
assert(util.is_int_scalar(onp.array([1,2])) == False)

is_int_scalar_jit = jax.jit(util.is_int_scalar)
assert(is_int_scalar_jit(5) == True)
assert(is_int_scalar_jit(0) == True)
assert(is_int_scalar_jit(-3.4) == False)

assert(is_int_scalar_jit(onp.asscalar(onp.array(1))) == True)
assert(is_int_scalar_jit(onp.asscalar(onp.array(1.))) == False)
assert(is_int_scalar_jit(onp.array([1,2])) == False)

print("success")
