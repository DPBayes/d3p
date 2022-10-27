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
""" PRNG interface for use in d3p to sample DP perturbation noise when debugging.

This is a simple wrapper around the standard jax.random PRNG. It is not
cryptographically secure and should not be used in production settings, but
runs faster than the secure variant and thus can be used to speed up debugging runs.
"""

import jax.numpy as jnp
from typing import Optional
import secrets
import warnings

import jax.random as jrng

try:
    from jax.random import KeyArray as PRNGState
except (AttributeError, ImportError):
    from jax._src.random import IntegerArray as PRNGState

split = jrng.split
fold_in = jrng.fold_in
from jax._src.random import _random_bits as _random_bits
uniform = jrng.uniform
normal = jrng.normal

try:
    from jax._src.random import _check_prng_key as _check_prng_key
except (AttributeError, ImportError):
    def _check_prng_key(x): return x, False

KeyRandomnessInBytes = 4

warnings.warn(
    "d3p is currently using a non-cryptographic random number generator!\n"
    "This is intended for debugging only! Please make sure to switch to using d3p.random to"
    " ensure privacy guarantees hold!",
    stacklevel=2
)


def PRNGKey(seed: Optional[int] = None) -> PRNGState:
    """Initializes a PRNGKey for the d3p debug random number generator.

    :param seed: Optional. A seed to derive randomness from.  Default: None. In this case,
        a seed is randomly sampled from the `secrets` module.
    """
    if seed is None:
        nonopt_seed = int.from_bytes(secrets.token_bytes(KeyRandomnessInBytes), 'big', signed=False)
    else:
        nonopt_seed = seed
    return jrng.PRNGKey(nonopt_seed)


def random_bits(key, bit_width, shape):
    key, _ = _check_prng_key(key)
    return _random_bits(key, bit_width, shape)


def convert_to_jax_rng_key(rng_key: PRNGState) -> jnp.array:
    """Converts a give d3p.random.debug RNG key to a jax.random RNG key.

    :param rng_key: d3p.random.debug RNG key.
    :return: jax.random RNG key derived from `rng_key`.
    """
    return rng_key
