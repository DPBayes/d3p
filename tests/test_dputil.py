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

""" tests that the interface code for the Fourier Accountant works correctly
"""
import unittest

import numpy as np

from d3p.dputil import approximate_sigma_remove_relation, get_epsilon_R


class ApproximateSigmaTests(unittest.TestCase):

    def test_approximate_sigma_recovers_accountant_failure(self):
        delta = 1e-5
        epsilon = 1.0
        q = 0.001
        num_iter = 100000
        tol = 1e-4

        failing_sig = 0.625

        # ensure that get_epsilon_R fails for at least one query during approximation procedure
        with self.assertRaises(ValueError):
            get_epsilon_R(delta, failing_sig, q, num_iter, nx=int(1e6))

        sigma, reached_eps, num_evals = approximate_sigma_remove_relation(
            epsilon, delta, q, 100000, maxeval=40, tol=tol
        )
        self.assertTrue(np.allclose(reached_eps, epsilon, atol=tol))
        self.assertTrue(np.isfinite(sigma))
        self.assertLessEqual(num_evals, 40)


if __name__ == '__main__':
    unittest.main()
