""" tests that the AdaDP optimizer works correctly
"""
import unittest

import jax.numpy as np
import jax
import numpy as onp

from functools import reduce

from dppp.optimizers import ADADP
import dppp.util

class ADADPTests(unittest.TestCase):

    def assertTreeStructure(self, expected, actual):
        self.assertTrue(dppp.util.do_trees_have_same_structure(expected, actual))

    def assertTreeAllClose(self, expected, actual):
        self.assertTrue(dppp.util.are_trees_close(expected, actual))

    def same_tree_with_value(self, tree, value):
        return jax.tree_map(
            lambda x: np.ones_like(x) * value, tree
        )

    def setUp(self):
        self.template = (
            np.ones(shape=(7, 10)),
            np.ones(shape=(7,)),
            (
                np.ones(shape=(2, 7)),
                np.ones(shape=(2,)),
            )
        )

    def test_init(self):
        learning_rate = 1.
        adadp = ADADP(learning_rate, 1.)
        value = self.template
        i, (x, lr, x_stepped, x_prev) = adadp.init(value)

        self.assertEqual(0, i)
        self.assertTreeAllClose(value, x)
        self.assertEqual(learning_rate, lr)
        self.assertTreeAllClose(
            self.same_tree_with_value(self.template, 0.), x_stepped
        )
        self.assertTreeStructure(self.template, x_prev)

    def test_update_step_1(self):
        learning_rate = 1.
        adadp = ADADP(learning_rate, 1.)
        value = self.same_tree_with_value(self.template, 0.)
        gradient = self.same_tree_with_value(self.template, 1.)

        opt_state = (
            0, (value, learning_rate, value, value)
        )
        i, (x, lr, x_stepped, x_prev) = adadp.update(gradient, opt_state)

        step_result = self.same_tree_with_value(self.template, -1.)
        half_step_result = self.same_tree_with_value(self.template, -0.5)

        self.assertEqual(1, i)
        self.assertTreeAllClose(half_step_result, x)
        self.assertEqual(learning_rate, lr)
        self.assertTreeAllClose(step_result, x_stepped)
        self.assertTreeAllClose(value, x_prev)

    def test_update_step_2_no_stability_check(self):
        learning_rate = 1.
        adadp = ADADP(learning_rate, tol=5., stability_check=False)
        value = self.same_tree_with_value(self.template, 0.)
        gradient = self.same_tree_with_value(self.template, 2.)

        opt_state = (
            1, 
            (
                self.same_tree_with_value(value, -0.5),
                learning_rate,
                self.same_tree_with_value(value, -1.),
                value
            )
        )        
        
        i, (x, lr, x_stepped, x_prev) = adadp.update(gradient, opt_state)

        step_result = self.same_tree_with_value(self.template, -1.)
        two_half_step_results = self.same_tree_with_value(self.template, -1.5)

        expected_lr = 1.018308251

        self.assertEqual(2, i)
        self.assertTreeAllClose(two_half_step_results, x)
        self.assertTrue(np.allclose(expected_lr, lr))

    def test_update_step_2_with_stability_check(self):
        learning_rate = 1.
        adadp = ADADP(learning_rate, tol=5., stability_check=True)
        value = self.same_tree_with_value(self.template, 0.)
        gradient = self.same_tree_with_value(self.template, 3.)

        opt_state = (
            1, 
            (
                self.same_tree_with_value(value, -0.5),
                learning_rate,
                self.same_tree_with_value(value, -1.),
                value
            )
        )        
        
        i, (x, lr, x_stepped, x_prev) = adadp.update(gradient, opt_state)

        step_result = self.same_tree_with_value(self.template, -1.)
        two_half_step_results = self.same_tree_with_value(self.template, -2.)

        expected_lr = .9 # 0.72005267 clipped by alpha_min

        self.assertEqual(2, i)
        self.assertTreeAllClose(value, x) # update rejected
        self.assertTrue(np.allclose(expected_lr, lr))

    def test_init_with_jit(self):
        learning_rate = 1.
        adadp = ADADP(learning_rate, 1.)
        value = self.template
        i, (x, lr, x_stepped, x_prev) = jax.jit(adadp.init)(value)

        self.assertEqual(0, i)
        self.assertTreeAllClose(value, x)
        self.assertEqual(learning_rate, lr)
        self.assertTreeAllClose(
            self.same_tree_with_value(self.template, 0.), x_stepped
        )
        self.assertTreeStructure(self.template, x_prev)

    def test_update_step_1_with_jit(self):
        learning_rate = 1.
        adadp = ADADP(learning_rate, 1.)
        value = self.same_tree_with_value(self.template, 0.)
        gradient = self.same_tree_with_value(self.template, 1.)

        opt_state = (
            0, (value, learning_rate, value, value)
        )
        i, (x, lr, x_stepped, x_prev) = jax.jit(adadp.update)(gradient, opt_state)

        step_result = self.same_tree_with_value(self.template, -1.)
        half_step_result = self.same_tree_with_value(self.template, -0.5)

        self.assertEqual(1, i)
        self.assertTreeAllClose(half_step_result, x)
        self.assertEqual(learning_rate, lr)
        self.assertTreeAllClose(step_result, x_stepped)
        self.assertTreeAllClose(value, x_prev)

    def test_update_step_2_no_stability_check_with_jit(self):
        learning_rate = 1.
        adadp = ADADP(learning_rate, tol=5., stability_check=False)
        value = self.same_tree_with_value(self.template, 0.)
        gradient = self.same_tree_with_value(self.template, 2.)

        opt_state = (
            1, 
            (
                self.same_tree_with_value(value, -0.5),
                learning_rate,
                self.same_tree_with_value(value, -1.),
                value
            )
        )        
        
        i, (x, lr, x_stepped, x_prev) = jax.jit(adadp.update)(gradient, opt_state)

        step_result = self.same_tree_with_value(self.template, -1.)
        two_half_step_results = self.same_tree_with_value(self.template, -1.5)

        expected_lr = 1.018308251

        self.assertEqual(2, i)
        self.assertTreeAllClose(two_half_step_results, x)
        self.assertTrue(np.allclose(expected_lr, lr))

    def test_update_step_2_with_stability_check_with_jit(self):
        learning_rate = 1.
        adadp = ADADP(learning_rate, tol=5., stability_check=True)
        value = self.same_tree_with_value(self.template, 0.)
        gradient = self.same_tree_with_value(self.template, 3.)

        opt_state = (
            1, 
            (
                self.same_tree_with_value(value, -0.5),
                learning_rate,
                self.same_tree_with_value(value, -1.),
                value
            )
        )        
        
        i, (x, lr, x_stepped, x_prev) = jax.jit(adadp.update)(gradient, opt_state)

        step_result = self.same_tree_with_value(self.template, -1.)
        two_half_step_results = self.same_tree_with_value(self.template, -2.)

        expected_lr = .9 # 0.72005267 clipped by alpha_min

        self.assertEqual(2, i)
        self.assertTreeAllClose(value, x) # update rejected
        self.assertTrue(np.allclose(expected_lr, lr))

if __name__ == '__main__':
    unittest.main()