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

from d3p import util


class UtilityTests(unittest.TestCase):

    def test_map_over_secondary_dims_with_sum(self):
        x = jnp.array([
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
        expected = jnp.array([
            [1.3, 2.4],
            [4.4, 0],
            [9.2, 2.5]
        ])

        mapped_sum = util.map_over_secondary_dims(jnp.sum)
        result = mapped_sum(x)
        self.assertTrue(np.allclose(expected, result))

    #### has_shape ####

    def test_has_shape_true_for_single_element_jax_array(self):
        a = jnp.array([1])
        self.assertTrue(util.has_shape(a))

    def test_has_shape_true_for_single_element_numpy_array(self):
        a = np.array([1])
        self.assertTrue(util.has_shape(a))

    def test_has_shape_false_for_scalar(self):
        a = 1.
        self.assertFalse(util.has_shape(a))

    def test_has_shape_true_for_complex_array(self):
        a = jnp.array([[1, 2], [3., 4.]])
        self.assertTrue(util.has_shape(a))

    def test_has_shape_true_for_array_under_jit(self):
        has_shape_jitted = jax.jit(util.has_shape)
        a = jnp.array([[1, 2], [3., 4.]])
        self.assertTrue(has_shape_jitted(a))

    def test_has_shape_true_for_scalar_under_jit(self):
        has_shape_jitted = jax.jit(util.has_shape)
        a = 2.4
        self.assertTrue(has_shape_jitted(a))

    def test_has_shape_deals_gracefully_with_none(self):
        self.assertFalse(util.has_shape(None))

    def test_has_shape_false_for_tuple(self):
        a = (2, 3)
        self.assertFalse(util.has_shape(a))

    #### is_array ####

    def test_is_array_true_for_single_element_jax_array(self):
        a = jnp.array([1])
        self.assertTrue(util.is_array(a))

    def test_is_array_true_for_single_element_numpy_array(self):
        a = np.array([1])
        self.assertTrue(util.is_array(a))

    def test_is_array_false_for_scalar(self):
        a = 1.
        self.assertFalse(util.is_array(a))

    def test_is_array_true_for_complex_array(self):
        a = jnp.array([[1, 2], [3., 4.]])
        self.assertTrue(util.is_array(a))

    def test_is_array_true_for_array_under_jit(self):
        is_array_jitted = jax.jit(util.is_array)
        a = jnp.array([[1, 2], [3., 4.]])
        self.assertTrue(is_array_jitted(a))

    def test_is_array_false_for_scalar_under_jit(self):
        is_array_jitted = jax.jit(util.is_array)
        a = 2.4
        self.assertFalse(is_array_jitted(a))

    def test_is_array_deals_gracefully_with_none(self):
        self.assertFalse(util.is_array(None))

    def test_is_array_false_for_tuple(self):
        a = (2, 3)
        self.assertFalse(util.is_array(a))

    #### is_scalar ####

    def test_is_scalar_true_for_single_element_array(self):
        a = jnp.array([1])
        self.assertTrue(util.is_scalar(a))

    def test_is_scalar_true_for_single_element_array_with_many_dims(self):
        a = jnp.array([1])
        a.reshape((1, 1, 1, 1))
        self.assertTrue(util.is_scalar(a))

    def test_is_scalar_true_for_scalar(self):
        a = 1.
        self.assertTrue(util.is_scalar(a))

    def test_is_scalar_false_for_complex_array(self):
        a = jnp.array([[1, 2], [3., 4.]])
        self.assertFalse(util.is_scalar(a))

    def test_is_scalar_false_for_array_under_jit(self):
        is_scalar_jitted = jax.jit(util.is_scalar)
        a = jnp.array([[1, 2], [3., 4.]])
        self.assertFalse(is_scalar_jitted(a))

    def test_is_scalar_true_for_single_element_array_with_many_dims_under_jit(self):
        is_scalar_jitted = jax.jit(util.is_scalar)
        a = jnp.array([1])
        a.reshape((1, 1, 1, 1))
        self.assertTrue(is_scalar_jitted(a))

    def test_is_scalar_true_for_scalar_under_jit(self):
        is_scalar_jitted = jax.jit(util.is_scalar)
        a = 2.4
        self.assertTrue(is_scalar_jitted(a))

    def test_is_scalar_false_for_tuple(self):
        a = (2, 3)
        self.assertFalse(util.is_scalar(a))

    #### is_integer ####

    def test_is_integer_true_for_int(self):
        a = 4
        self.assertTrue(util.is_integer(a))

    def test_is_integer_false_for_float(self):
        a = 4.
        self.assertFalse(util.is_integer(a))

    def test_is_integer_true_for_int_array(self):
        a = jnp.array([2, 3])
        self.assertTrue(util.is_integer(a))

    def test_is_integer_false_for_float_array(self):
        a = jnp.array([2, 3.])
        self.assertFalse(util.is_integer(a))

    def test_is_integer_true_for_int_array_under_jit(self):
        is_integer_jitted = jax.jit(util.is_integer)
        a = jnp.array([2, 3])
        self.assertTrue(is_integer_jitted(a))

    def test_is_integer_false_for_float_array_under_jit(self):
        is_integer_jitted = jax.jit(util.is_integer)
        a = jnp.array([2, 3.])
        self.assertFalse(is_integer_jitted(a))

    #### is_int_scalar ####

    def test_is_int_scalar_true_for_int_scalar(self):
        a = 5
        self.assertTrue(util.is_int_scalar(a))

    def test_is_int_scalar_false_for_float_scalar(self):
        a = 5.
        self.assertFalse(util.is_int_scalar(a))

    def test_is_int_scalar_true_for_single_element_int_array(self):
        a = jnp.array([2])
        self.assertTrue(util.is_int_scalar(a))

    def test_is_int_scalar_false_for_single_element_float_array(self):
        a = jnp.array([2.])
        self.assertFalse(util.is_int_scalar(a))

    def test_is_int_scalar_false_for_array(self):
        a = jnp.array([5, 4])
        self.assertFalse(util.is_int_scalar(a))

    def test_is_int_scalar_true_for_int_scalar_under_jit(self):
        is_int_scalar_jitted = jax.jit(util.is_int_scalar)
        a = 5
        result = is_int_scalar_jitted(a)
        self.assertTrue(result)

    def test_is_int_scalar_false_for_float_scalar_under_jit(self):
        is_int_scalar_jitted = jax.jit(util.is_int_scalar)
        a = 5.
        self.assertFalse(is_int_scalar_jitted(a))

    def test_is_int_scalar_true_for_single_element_int_array_under_jit(self):
        is_int_scalar_jitted = jax.jit(util.is_int_scalar)
        a = jnp.array([2])
        self.assertTrue(is_int_scalar_jitted(a))

    def test_is_int_scalar_false_for_single_element_float_array_under_jit(self):
        is_int_scalar_jitted = jax.jit(util.is_int_scalar)
        a = jnp.array([2.])
        self.assertFalse(is_int_scalar_jitted(a))

    def test_is_int_scalar_false_for_array_under_jit(self):
        is_int_scalar_jitted = jax.jit(util.is_int_scalar)
        a = jnp.array([5, 4])
        self.assertFalse(is_int_scalar_jitted(a))

    #### example_count ####

    def test_example_count_is_correct(self):
        a = jnp.array([[1., 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
        expected = 5
        result = util.example_count(a)
        self.assertEqual(expected, result)

    def test_example_count_is_correct_for_single_element(self):
        a = jnp.array(1)
        expected = 1
        result = util.example_count(a)
        self.assertEqual(expected, result)

    def test_example_count_deals_gracefully_with_scalar(self):
        a = 1.
        expected = 1
        result = util.example_count(a)
        self.assertEqual(expected, result)

    def test_example_count_is_correct_under_jit(self):
        example_count_jitted = jax.jit(util.example_count)
        a = jnp.array([[1., 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
        expected = 5
        result = example_count_jitted(a)
        self.assertEqual(expected, result)

    def test_example_count_is_correct_for_single_element_under_jit(self):
        example_count_jitted = jax.jit(util.example_count)
        a = jnp.array(1)
        expected = 1
        result = example_count_jitted(a)
        self.assertEqual(expected, result)

    def test_example_count_deals_gracefully_with_scalar_under_jit(self):
        example_count_jitted = jax.jit(util.example_count)
        a = 1.
        expected = 1
        result = example_count_jitted(a)
        self.assertEqual(expected, result)

    #### normalize ####

    def test_normalize_correct(self):
        x = jnp.arange(7)
        res = util.normalize(x)
        self.assertTrue(np.allclose(1., jnp.linalg.norm(res)))

    def test_normalize_correct_for_scalar(self):
        x = 8.
        res = util.normalize(x)
        self.assertAlmostEqual(1., res)

    #### unvectorize_shape ####

    def test_unvectorize_shape_input_has_expected_amount_of_dims(self):
        a = jnp.array([[3, 4], [5, 6], [7, 8]])
        expected = (3, 2)
        res = util.unvectorize_shape(a, 2)
        self.assertEqual(expected, res)

    def test_unvectorize_shape_input_has_more_dims(self):
        a = jnp.array([[3, 4], [5, 6], [7, 8]])
        expected = (3, 2)
        res = util.unvectorize_shape(a, 1)
        self.assertEqual(expected, res)

    def test_unvectorize_shape_input_has_one_missing_dim(self):
        a = jnp.array([[3, 4], [5, 6], [7, 8]])
        expected = (1, 3, 2)
        res = util.unvectorize_shape(a, 3)
        self.assertEqual(expected, res)

    def test_unvectorize_shape_input_has_two_missing_dims(self):
        a = jnp.array([[3, 4], [5, 6], [7, 8]])
        expected = (1, 1, 3, 2)
        res = util.unvectorize_shape(a, 4)
        self.assertEqual(expected, res)

    def test_unvectorize_shape_input_is_scalar_and_one_dim_expected(self):
        a = 3
        expected = (1,)
        res = util.unvectorize_shape(a, 1)
        self.assertEqual(expected, res)

    def test_unvectorize_shape_input_is_scalar_and_two_dims_expected(self):
        a = 3
        expected = (1, 1)
        res = util.unvectorize_shape(a, 2)
        self.assertEqual(expected, res)

    ### shuffle

    def test_sample_from_array(self):
        x = jnp.arange(0, 1000000) + 100
        rng_key = jax.random.PRNGKey(0)
        n_vals = 978
        shuffled = util.sample_from_array(rng_key, x, n_vals, 0)
        unq_vals = np.unique(shuffled)
        self.assertEqual(n_vals, np.size(unq_vals))
        self.assertTrue(jnp.alltrue(shuffled >= 100))

    def test_sample_from_array_correct_shape(self):
        x = jax.random.uniform(jax.random.PRNGKey(124), shape=(1000, 200))
        rng_key = jax.random.PRNGKey(0)
        n_vals = 38
        shuffled = util.sample_from_array(rng_key, x, n_vals, 0)
        self.assertEqual((n_vals, 200), jnp.shape(shuffled))

        shuffled = util.sample_from_array(rng_key, x, n_vals, 1)
        self.assertEqual((1000, n_vals), jnp.shape(shuffled))

    def test_sample_from_array_full_shuffle(self):
        x = jnp.arange(0, 100) + 100
        rng_key = jax.random.PRNGKey(0)
        n_vals = 100
        shuffled = util.sample_from_array(rng_key, x, n_vals, 0)
        unq_vals = np.unique(shuffled)
        self.assertEqual(n_vals, np.size(unq_vals))
        self.assertTrue(jnp.alltrue(shuffled >= 100))

    def test_sample_from_array_almost_full_shuffle(self):
        x = jnp.arange(0, 100) + 100
        rng_key = jax.random.PRNGKey(0)
        n_vals = 99
        shuffled = util.sample_from_array(rng_key, x, n_vals, 0)
        unq_vals = np.unique(shuffled)
        self.assertEqual(n_vals, np.size(unq_vals))
        self.assertTrue(jnp.alltrue(shuffled >= 100))

    def test_sample_from_array_single_sample(self):
        x = jnp.arange(0, 100) + 100
        rng_key = jax.random.PRNGKey(0)
        n_vals = 1
        shuffled = util.sample_from_array(rng_key, x, n_vals, 0)
        self.assertTrue(jnp.alltrue(shuffled >= 100))


if __name__ == '__main__':
    unittest.main()
