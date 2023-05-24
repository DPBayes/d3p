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
from functools import reduce
import unittest

def _and_reduce(iterable):
    return reduce(lambda x, y: x and y, iterable, True)

def _tree_and_reduce(tree):
    return jax.tree_util.tree_reduce(lambda x, y: x and y, tree, True)

def do_trees_have_same_structure(a, b):
    """Returns True if two jax trees have the same structure.
    """
    return jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)


def do_trees_have_same_shape(a, b):
    """Returns True if two jax trees have the same structure and the shapes of
        all corresponding leaves are identical.
    """
    return do_trees_have_same_structure(a, b)  and _tree_and_reduce(
        jax.tree_util.tree_map(lambda x, y: jnp.shape(x) == jnp.shape(y), a, b)
    )


def are_trees_close(a, b):
    """Returns True if two jax trees have the same structure and all corresponding
    leaf values are close.
    """
    return do_trees_have_same_shape(a, b) and _tree_and_reduce(
        jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, y), a, b)
    )


def are_trees_equal(a, b):
    """Returns True if two jax trees have the same structure and all corresponding
    leaf values are equal."""
    return do_trees_have_same_shape(a, b) and _tree_and_reduce(
        jax.tree_util.tree_map(lambda x, y: jnp.all(x == y), a, b)
    )
