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
""" Cryptographically secure PRNG interface used by d3p to sample DP perturbation noise.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Union, Sequence
from functools import partial
import secrets

import chacha.random as ccrng
from chacha.defs import ChaChaKeySizeInBytes

PRNGState = ccrng.RNGState
split = ccrng.split
fold_in = ccrng.fold_in
random_bits = ccrng.random_bits
uniform = ccrng.uniform


def PRNGKey(seed: Optional[Union[jnp.ndarray, int, bytes]] = None) -> PRNGState:
    """Initializes a PRNGKey for the d3p.random secure random number generator.

    :param seed: Optional. A seed to derive randomness from, equivalent to a cryptographic
        key. Its length in bits determines the cryptographic strength of the generated random numbers.
        It can be up to 256 bit long. Default: None. In this case, a full length seed is randomly sampled
        from the `secrets` module.
    """
    if seed is None:
        nonopt_seed = secrets.token_bytes(ChaChaKeySizeInBytes)
    else:
        nonopt_seed = seed
    return ccrng.PRNGKey(nonopt_seed)


def normal(key: ccrng.RNGState,
           shape: Sequence[int] = (),
           dtype: np.dtype = jnp.float_) -> jnp.ndarray:
    """Samples standard normal random values with given shape and float dtype
        derived from a cryptographically-secure pseudo-random number generator.

    The sampling follows `jax.random.normal` exactly but uses `jax-chacha-prng`
    as underlying generator instead of JAX's default generator.

    Note that this currently leaves this vulnerable to attacks on imprecise sampling
    as described in Mironov, "On Significance of the Least Significant Bits For Differential Privacy".

    :param key: A PRNGKey used as the random key.
    :param shape: Optional. A tuple of nonnegative integers representing the result
        shape. Default ().
    :param dtype: Optional. A float dtype for the returned values (default `float64` if
        `jax_enable_x64` is `true`, otherwise `float32`).

    :return: A random array with the specified shape and dtype.
    """
    if not jax.dtypes.issubdtype(dtype, np.floating):
        raise ValueError(f"dtype argument to `normal` must be a float dtype, got {dtype}")
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    return _normal(key, shape, dtype)


@partial(jax.jit, static_argnums=(1, 2))
def _normal(key, shape, dtype) -> jnp.ndarray:
    lo = np.nextafter(np.array(-1., dtype), 0., dtype=dtype)
    hi = np.array(1., dtype)
    u = ccrng.uniform(key, shape, dtype, lo, hi)
    return np.array(np.sqrt(2), dtype) * jax.lax.erf_inv(u)


def randint(key: ccrng.RNGState,
            shape: Sequence[int],
            minval: Union[jnp.int16, jnp.int32, jnp.int64],
            maxval: Union[jnp.int16, jnp.int32, jnp.int64],
            dtype: np.dtype = jnp.int32) -> jnp.ndarray:
    """Uniformly samples integer values in given bounds with given shape
        derived from a cryptographically-secure pseudo-random number generator.

    :param key: A PRNGKey used as the random key.
    :param shape: A tuple of nonnegative integers representing the result
        shape.
    :param minval: Minimum possible value of samples..
    :param maxval: Maximum possible value of samples.
    :param dtype: Optional. An integer dtype for the returned values (default `int32`).

    :return: A random array with the specified shape and dtype.
    """
    if not jax.dtypes.issubdtype(dtype, np.integer):
        raise TypeError(f"dtype argument to `randint` must be an integer dtype, got {dtype}")
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    return _randint(key, minval, maxval, shape, dtype)



@partial(jax.jit, static_argnums=(3, 4))
def _randint(key, minval,  maxval, shape, dtype) -> jnp.ndarray:
    if not jax.dtypes.issubdtype(dtype, np.integer):
        raise TypeError(f"dtype argument to `randint` must be an integer dtype, got {dtype}")
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    iinfo = jnp.iinfo(dtype)
    nbits = iinfo.bits
    assert nbits in (8, 16, 32, 64)

    vdtype, udtype = {
         8: (jnp.int8,  jnp.uint8 ),
        16: (jnp.int16, jnp.uint16),
        32: (jnp.int32, jnp.uint32),
        64: (jnp.int64, jnp.uint64)
    }[nbits]

    delta = udtype(maxval - 1 - minval)
    log_delta_next_po2 = jnp.minimum(udtype(jnp.log2(jnp.float32(delta)) + 1), nbits)

    delta_next_po2 = udtype(1) << log_delta_next_po2
    bitmask = delta_next_po2 - udtype(1)

    def sampling_condition(args):
        _, uvals = args
        return jnp.any((uvals > delta)) # & (uvals > 0))

    def sampling_body(args):
        key, uvals = args
        key, round_key = ccrng.split(key, 2)
        new_uvals = jax.lax.bitwise_and(udtype(ccrng.random_bits(round_key, nbits, shape)), bitmask)
        uvals = jnp.where(uvals > delta, new_uvals, uvals)
        return key, uvals
    
    key, round_key = ccrng.split(key, 2)
    uvals = jax.lax.bitwise_and(udtype(ccrng.random_bits(round_key, nbits, shape)), bitmask)
    _, uvals = jax.lax.while_loop(sampling_condition, sampling_body, (key, uvals))

    vals = vdtype(uvals) + minval
    return vals


def convert_to_jax_rng_key(rng_key: ccrng.RNGState) -> jnp.array:
    """Converts a give d3p.random RNG key to a jax.random RNG key.

    :param rng_key: d3p.random RNG key.
    :return: jax.random RNG key derived from `rng_key`.
    """
    return ccrng.random_bits(rng_key, 32, (2,))
