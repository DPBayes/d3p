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


def PRNGKey(seed: Optional[Union[int, jnp.ndarray, int]] = None) -> PRNGState:
    if seed is None:
        nonopt_seed = secrets.token_bytes(ChaChaKeySizeInBytes)
    else:
        nonopt_seed = seed
    return ccrng.PRNGKey(nonopt_seed)


def normal(key: ccrng.RNGState,
           shape: Sequence[int] = (),
           dtype: np.dtype = jnp.float_) -> jnp.ndarray:
    """Sample standard normal random values with given shape and float dtype
        derived from a cryptographically-secure pseudo-random number generator.

    The sampling follows `jax.random.normal` exactly but uses jax-chacha-prng
    as underlying generator instead of JAX's default generator.

    Note that this currently leaves this vulnerable to attacks on imprecise sampling
    as described in Mironov, "On Significance of the Least Significant Bits For Differential Privacy".

    Args:
        key: a PRNGKey used as the random key.
        shape: optional, a tuple of nonnegative integers representing the result
        shape. Default ().
        dtype: optional, a float dtype for the returned values (default float64 if
        jax_enable_x64 is true, otherwise float32).

    Returns:
        A random array with the specified shape and dtype.
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


def convert_to_jax_rng_key(chacha_rng_key: ccrng.RNGState) -> jnp.array:
    return ccrng.random_bits(chacha_rng_key, 32, (2,))
