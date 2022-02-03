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

import jax
import jax.numpy as jnp
import d3p.random as strong_rng
from functools import reduce, wraps, partial

__all__ = [
    "map_over_secondary_dims", "has_shape", "is_array", "is_scalar",
    "is_integer", "is_int_scalar", "example_count",
    "unvectorize_shape", "unvectorize_shape_1d", "unvectorize_shape_2d",
    "unvectorize_shape_3d"
]


def map_over_secondary_dims(f):
    """ Maps a function taking a over all secondary axes of an array.

    f is assumed to take a vector of shape (a,) and output a scalar.
    Returns a function f_mapped that for an input T with shape
    (a, b_1, ..., b_k) applies f on the entries along the first axis
    over all other axes, i.e., to all T[:,i_1, ..., i_k].

    The output of f_mapped will be an array of shape [b_1, ...., b_k] where
    each entry is the corresponding output of f applied as described above.

    :param f: a function from vector to scalar
    :return: function f_mapped applying f to each first axis vector in an
        arbitrary shaped input

    Example:
    >>> T = [
    >>>      [ [ a_1, a_2 ], [ a_3, a_4 ] ],
    >>>      [ [ b_1, b_2 ], [ b_3, b_4 ] ],
    >>>      [ [ c_1, c_2 ], [ c_3, c_4 ] ]
    >>> ]
    >>> print(T.shape)
    (3, 2, 2)
    >>> R = map_over_secondary_dims(f)(T)
    >>> print(R)
    [
        [f(a_1, b_1, c_1), f(a_2, b_2, c_2)],
        [f(a_3, b_3, c_3), f(a_4, b_4, c_4)]
    ]
    """
    @wraps(f)
    def map_over_secondary_dims_f(T):
        assert(jnp.ndim(T) >= 1)
        T_ = T.reshape((T.shape[0], -1))
        Z_ = jax.vmap(f, in_axes=1)(T_)
        return Z_.reshape(T.shape[1:])
    return map_over_secondary_dims_f


def example_count(a):
    """ Returns the amount of examples/observations in an array interpreted
    as multi-example data set.

    :param a: The data set from which to extract the example count.
    """
    try:
        return jnp.shape(a)[0]
    except IndexError:
        return 1


def has_shape(a):
    """ Returns True if the input has the shape attribute, indicating that it is
    of a numpy array type.

    Note that this also applies to scalars in jax.jit decorated functions.
    """
    try:
        a.shape
        return True
    except AttributeError:
        return False


def is_array(a):
    """ Returns True if the input has is determined to be an array, i.e.,
    has more than 0 dimensions and a shape attribute.

    Note that this does not apply to scalars in jax.jit decorated functions.

    :param a: Anything that might be an array.
    """
    return has_shape(a) and jnp.ndim(a) > 0


def is_scalar(x):
    """ Returns True if the input can be interpreted as a scalar.

    This fits actual scalars as well as arrays that contain only one element
    (regardless of their number of dimensions). I.e., a (jax.)numpy array
    with shape (1,1,1,1) would be considered a scalar.

    Works with jax.jit.
    """
    # note(lumip): a call to jax.jit(is_scalar)(s), where x is a scalar,
    #   results in an x that is a jax.numpy array without any dimensions but
    #   which has a shape attribute. therefore, jnp.isscalar(x) as well as
    #   is_array(x) are False -> we have to use has_shape(x) to detect this
    return jnp.isscalar(x) or (has_shape(x) and reduce(lambda x, a: x*a, jnp.shape(x), 1) == 1)


def is_integer(x):
    """ Returns True if the input value(s) (a scalar or (jax.)numpy array) have integer type.

    Works with jax.jit.

    :param x: Scalar or (jax.)numpy array that could have integer values.
    """
    return (has_shape(x) and jnp.issubdtype(x.dtype, jnp.integer)) or jnp.issubdtype(type(x), jnp.integer)


def is_int_scalar(x):
    """ Returns True if the input can be interepreted as a scalar integer value.

    Works with jax.jit.

    :param x: Anything that might be an integer scalar.
    """
    return is_scalar(x) and is_integer(x)


def normalize(x):
    """ Normalizes a vector, i.e., returns a vector with unit lengths pointing
    in the same direction.

    :param x: The vector to normalize.
    """
    return x / jnp.linalg.norm(x)


def unvectorize_shape(a, d):
    """ Undoes the stripping of leading dimensions in the shape of a vectorized/
    vmapped array.

    Accepts a target number of dimensions and returns a shape of at least that
    length. If the shape of the vectorized array is smaller than that, it will
    be filled with dimensions of size 1 from the front. If the input
    array has as least as many dimensions as specified, its shape is returned
    unmodified.

    Similar to `jnp.shape(jnp.atleast_xd(a))` but guaranteed to fill the shape
    from the front.

    :param a: The vectorized/vmapped array.
    :param d: The minimum number of dimensions included in the output shape.
    """
    shape = jnp.shape(a)
    ndim = len(shape)
    if ndim < d:
        return (1,) * (d - ndim) + shape
    else:
        return shape


def unvectorize_shape_1d(a):
    """ Undoes the stripping of leading dimensions in the shape of a vectorized/
    vmapped 1-dimensional array.

    If the shape of the vectorized array is smaller than 1, it will
    be filled with dimensions of size 1 from the front. If the input
    array has dimensionality 1 or greater, its shape is returned unmodified.

    :param a: The vectorized/vmapped array.
    """
    return unvectorize_shape(a, 1)


def unvectorize_shape_2d(a):
    """ Undoes the stripping of leading dimensions in the shape of a vectorized/
    vmapped 2-dimensional array.

    If the shape of the vectorized array is smaller than 2, it will
    be filled with dimensions of size 1 from the front. If the input
    array has dimensionality 2 or greater, its shape is returned unmodified.

    Similar to `jnp.shape(jnp.atleast_2d(a))`.

    :param a: The vectorized/vmapped array.
    """
    return unvectorize_shape(a, 2)


def unvectorize_shape_3d(a):
    """ Undoes the stripping of leading dimensions in the shape of a vectorized/
    vmapped 3-dimensional array.

    If the shape of the vectorized array is smaller than 3, it will
    be filled with dimensions of size 1 from the front. If the input
    array has dimensionality 3 or greater, its shape is returned unmodified.

    Similar to `jnp.shape(jnp.atleast_3d(a))` but fills the shape from the front.

    :param a: The vectorized/vmapped array.
    """
    return unvectorize_shape(a, 3)


@partial(jax.jit, static_argnums=(2, 3, 4))
def sample_from_array(rng_key, x, n, axis, rng_suite=strong_rng):
    """ Samples `n` elements from array `x` along given `axis` without replacement.

    Implementation notes:
      Internally implements the CUDA/Feistel shuffle: We apply a pseudo-random
      permutation based on a Feistel network for each position index in the
      output sample to obtain the index of the sampled element.

      The diffusion functions for the Feistel network follow the Philox
      construction of Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3",
      but using randomly sampled multiplication constants.
    """
    capacity = jnp.shape(x)[axis]
    bits = (capacity - 1).bit_length()
    idxs = jnp.arange(n, dtype=jnp.uint32)

    num_feistel_rounds = 10

    # Philox multiplication constants and xor'ed key for each Feistel round.
    # We use separate multiplication constants for each multiplication in a round
    # to increase the amount of random bits we use in the process;
    # Using the same multiplicative constant in Philox seems to be mostly motivated
    # from a performance perspective.
    round_constants = rng_suite.random_bits(
        rng_key, 32, (num_feistel_rounds, 3)
    )

    # need to ensure that multiplication constants are odd so that scramble_lower is a bijection
    make_first_odd = jnp.array([[1, 0, 0]], dtype=jnp.uint32)
    round_constants = jnp.bitwise_or(round_constants, make_first_odd)

    # masks to separate high and low bit regions
    bits_lower = jnp.right_shift(bits, 1)
    bits_upper = bits - bits_lower
    mask_lower = jnp.left_shift(jnp.uint32(1), bits_lower) - 1
    mask_upper = jnp.left_shift(jnp.uint32(1), bits_upper) - 1

    def scramble_for_upper(x, key):
        """ Scrambles/diffues bits on the diffusion path (lower to upper bits) of the Feistel network.

        F_k in Salmon et al."""
        return jnp.right_shift(x * key[1], bits_upper) ^ key[2]

    def scramble_for_lower(x, key):
        """ Scrambles/diffuses bits on the bijective path (upper to lower bits) of the Feistel network.

        B_k in Salmon et al. """
        return x * key[0]

    def permute_idx_power_of_two_capacity(position):
        """ Evaluates a random permutation (keyed by `rng_key`) for integers less
            than the next larger power-of-two from `capacity`, i.e., `2**(bits)`.

        Essentially a Feistel network to build a permutation from some non-bijective scrambling/diffusion function
        (`scramble_for_upper`).
         """
        def loop_body(j, x):
            x_upper = jnp.right_shift(x, bits_lower)
            x_lower = jnp.bitwise_and(x, mask_lower)

            y_upper = jnp.bitwise_and(scramble_for_upper(x_upper, round_constants[j]), mask_lower)
            y_upper = jnp.bitwise_xor(x_lower, y_upper)

            y_lower = jnp.bitwise_and(scramble_for_lower(x_upper, round_constants[j]), mask_upper)

            y = jnp.bitwise_or(jnp.left_shift(y_upper, bits_upper), y_lower)
            return y

        position = jax.lax.fori_loop(0, num_feistel_rounds, loop_body, position)
        return position

    def permute_idx(position):
        """ Evaluates a random permutation (keyed by `rng_key`) over {0, ..., `capacity`-1}. """

        position = permute_idx_power_of_two_capacity(position)
        position = jax.lax.while_loop(
            lambda position: position >= capacity,
            permute_idx_power_of_two_capacity,
            position
        )

        return position

    idxs = jax.vmap(permute_idx)(idxs)  # randomly permute array indices {1,...,n}
    return jnp.take(x, idxs, axis)  # select the array values corresponding to permuted indices
