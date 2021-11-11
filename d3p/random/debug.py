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

import jax.numpy as jnp
from typing import Optional
import secrets
import warnings

import jax.random as jrng
import jax

PRNGState = jrng.KeyArray
split = jrng.split
fold_in = jrng.fold_in
random_bits = jax._src.random._random_bits
uniform = jrng.uniform
normal = jrng.normal

KeyRandomnessInBytes = 4

warnings.warn(
    "d3p is currently using a non-cryptographic random number generator!\n"
    "This is intended for debugging only! Please make sure to switch to using d3p.random to"
    " ensure privacy guarantees hold!",
    stacklevel=2
)


def PRNGKey(seed: Optional[int] = None) -> PRNGState:
    if seed is None:
        nonopt_seed = int.from_bytes(secrets.token_bytes(KeyRandomnessInBytes), 'big', signed=False)
    else:
        nonopt_seed = seed
    return jrng.PRNGKey(nonopt_seed)


def convert_to_jax_rng_key(chacha_rng_key: PRNGState) -> jnp.array:
    return chacha_rng_key
