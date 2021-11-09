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

"""tests the implementations in the d3p.utils package
"""
import unittest

import jax.numpy as jnp
import jax
import numpy as np

from tests import util


class TestsUtilityTests(unittest.TestCase):

    #### do_trees_have_same_structure ####

    def test_do_trees_have_same_structure(self):
        a = ((jnp.ones((3, 4)), jnp.ones((7, 6, 3))), jnp.ones(7))
        b = ((jnp.ones((3, 4)), jnp.ones((7, 6, 3))), jnp.ones(7))
        self.assertTrue(util.do_trees_have_same_structure(a, b))

    def test_do_trees_have_same_structure_empty(self):
        a = ()
        b = ()
        self.assertTrue(util.do_trees_have_same_structure(a, b))

    def test_do_trees_have_same_structure_different_shape_dimensions(self):
        a = ((jnp.ones((3, 4)), jnp.ones((7, 6, 3))), jnp.ones(7))
        b = ((jnp.ones((8, 4)), jnp.ones((7, 6, 3))), jnp.ones(7))
        self.assertTrue(util.do_trees_have_same_structure(a, b))

    def test_do_trees_have_same_structure_different_shape_lengths(self):
        a = ((jnp.ones((3, 4)), jnp.ones((7, 6, 3))), jnp.ones(7))
        b = ((jnp.ones((3)), jnp.ones((7, 6))), jnp.ones(7))
        self.assertTrue(util.do_trees_have_same_structure(a, b))

    def test_do_trees_have_same_structure_different_values(self):
        a = ((jnp.ones((3, 4)), jnp.ones((7, 6, 3))), jnp.ones(7))
        b = ((jnp.zeros((3, 4)), jnp.ones((7, 6, 3))), jnp.ones(7))
        self.assertTrue(util.do_trees_have_same_structure(a, b))

    #### do_trees_have_same_shape ####

    def test_do_trees_have_same_shape(self):
        a = ((jnp.ones((3, 4)), jnp.ones((7, 6, 3))), jnp.ones(7))
        b = ((jnp.ones((3, 4)), jnp.ones((7, 6, 3))), jnp.ones(7))
        self.assertTrue(util.do_trees_have_same_shape(a, b))

    def test_do_trees_have_same_shape_empty(self):
        a = ()
        b = ()
        self.assertTrue(util.do_trees_have_same_shape(a, b))

    def test_do_trees_have_same_shape_different_shape_dimensions(self):
        a = ((jnp.ones((3, 4)), jnp.ones((7, 6, 3))), jnp.ones(7))
        b = ((jnp.ones((8, 4)), jnp.ones((7, 6, 3))), jnp.ones(7))
        self.assertFalse(util.do_trees_have_same_shape(a, b))

    def test_do_trees_have_same_shape_different_shape_lengths(self):
        a = ((jnp.ones((3, 4)), jnp.ones((7, 6, 3))), jnp.ones(7))
        b = ((jnp.ones((3)), jnp.ones((7, 6))), jnp.ones(7))
        self.assertFalse(util.do_trees_have_same_shape(a, b))

    def test_do_trees_have_same_shape_different_values(self):
        a = ((jnp.ones((3, 4)), jnp.ones((7, 6, 3))), jnp.ones(7))
        b = ((jnp.zeros((3, 4)), jnp.ones((7, 6, 3))), jnp.ones(7))
        self.assertTrue(util.do_trees_have_same_shape(a, b))

    #### are_trees_close ####

    def test_are_trees_close(self):
        a = ((jnp.ones((3, 4)), jnp.ones((7, 6, 3))), jnp.ones(7))
        b = ((jnp.ones((3, 4)), jnp.ones((7, 6, 3))), jnp.ones(7))
        self.assertTrue(util.are_trees_close(a, b))

    def test_are_trees_close_empty(self):
        a = ()
        b = ()
        self.assertTrue(util.are_trees_close(a, b))

    def test_are_trees_close_different_shape_dimensions(self):
        a = ((jnp.ones((3, 4)), jnp.ones((7, 6, 3))), jnp.ones(7))
        b = ((jnp.ones((8, 4)), jnp.ones((7, 6, 3))), jnp.ones(7))
        self.assertFalse(util.are_trees_close(a, b))

    def test_are_trees_close_different_shape_lengths(self):
        a = ((jnp.ones((3, 4)), jnp.ones((7, 6, 3))), jnp.ones(7))
        b = ((jnp.ones((3)), jnp.ones((7, 6))), jnp.ones(7))
        self.assertFalse(util.are_trees_close(a, b))


if __name__ == '__main__':
    unittest.main()
