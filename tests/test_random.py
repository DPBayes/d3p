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

"""tests the implementations in the d3p.random package
"""
import unittest
import numpy as np

import d3p.random
import d3p.random.debug
import scipy.stats


class RNGSuiteTests:

    def test_PRNGKey(self) -> None:
        self.assertTrue(np.any(self.rng_suite.PRNGKey() != 0))

    def test_random_bits(self) -> None:
        key = self.rng_suite.PRNGKey(98734)
        shape = (3, 8, 9)

        result = self.rng_suite.random_bits(key, 32, shape)
        self.assertEqual(result.shape, shape)
        self.assertEqual(result.dtype, np.uint32)
        self.assertTrue(np.any(result != 0))

    def test_uniform(self) -> None:
        key = self.rng_suite.PRNGKey(98734)
        shape = (1000, 8, 9)
        total = np.prod(shape)

        result = self.rng_suite.uniform(key, shape)
        self.assertEqual(result.shape, shape)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.any(result != 0))

        self.assertTrue(np.abs(np.mean(result) - .5) <= 5/(12*np.sqrt(total)))
        res = scipy.stats.kstest(np.ravel(result), "uniform")
        self.assertTrue(res.pvalue > 0.05)

    def test_normal(self) -> None:
        key = self.rng_suite.PRNGKey(98734)
        shape = (1000, 8, 9)
        total = np.prod(shape)

        result = self.rng_suite.normal(key, shape)
        self.assertEqual(result.shape, shape)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.any(result != 0))

        self.assertTrue(np.abs(np.mean(result)) <= 5/np.sqrt(total))
        res = scipy.stats.kstest(np.ravel(result), "norm")
        self.assertTrue(res.pvalue > 0.05)


class StrongRNGSuiteTests(RNGSuiteTests, unittest.TestCase):

    def setUp(self) -> None:
        self.rng_suite = d3p.random


class DebugRNGSuiteTests(RNGSuiteTests, unittest.TestCase):

    def setUp(self) -> None:
        self.rng_suite = d3p.random.debug


if __name__ == '__main__':
    unittest.main()
