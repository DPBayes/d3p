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
import scipy.stats

import d3p.random
import d3p.random.debug


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

        res = scipy.stats.kstest(
            np.ravel(result), scipy.stats.uniform.cdf
        )
        self.assertTrue(res.pvalue >= 0.05)

    def test_normal(self) -> None:
        key = self.rng_suite.PRNGKey(98734)
        shape = (1000, 8, 9)
        total = np.prod(shape)

        result = self.rng_suite.normal(key, shape)
        self.assertEqual(result.shape, shape)
        self.assertEqual(result.dtype, np.float32)
        self.assertTrue(np.any(result != 0))

        self.assertTrue(np.abs(np.mean(result)) <= 5/np.sqrt(total))

        res = scipy.stats.kstest(
            np.ravel(result), scipy.stats.norm.cdf
        )
        self.assertTrue(res.pvalue >= 0.05)

    def test_randint(self) -> None:
        key = self.rng_suite.PRNGKey(8025111)
        shape = (1000, 8, 9)
        minval = 8
        maxval = minval + 2**10 + 1
        num_values = maxval - minval

        result = self.rng_suite.randint(key, shape, minval, maxval, np.int32)
        self.assertTrue(result.shape, shape)
        self.assertTrue(result.dtype, np.int32)
        self.assertTrue(np.max(result) == maxval - 1)
        self.assertTrue(np.min(result) == minval)

        vals, valfreqs = np.unique(np.ravel(result), return_counts=True)
        freqs = np.zeros(num_values)
        freqs[vals - minval] = valfreqs
        res = scipy.stats.chisquare(
            freqs, 
        )
        self.assertTrue(res.pvalue >= 0.05)

    def test_randint_full_range(self) -> None:
        key = self.rng_suite.PRNGKey(802511)
        shape = (1000, 8, 9)
        minval = -2**7
        maxval = np.uint16(2**7)
        num_values = maxval - minval

        result = self.rng_suite.randint(key, shape, minval, maxval, np.int8)
        self.assertTrue(result.shape, shape)
        self.assertTrue(result.dtype, np.int16)
        self.assertTrue(np.any(result >= minval))
        self.assertTrue(np.any(result < maxval))

        vals, valfreqs = np.unique(np.ravel(result), return_counts=True)
        freqs = np.zeros(num_values)
        freqs[vals - minval] = valfreqs
        res = scipy.stats.chisquare(
            freqs, 
        )
        self.assertTrue(res.pvalue >= 0.05)

    def test_randint_to_upper_bound(self) -> None:
        key = self.rng_suite.PRNGKey(8025111)
        shape = (1000, 8, 9)
        minval = 0
        maxval = np.uint16(2**15) #2**15-1
        num_values = maxval - minval

        result = self.rng_suite.randint(key, shape, minval, maxval, np.int16)
        self.assertTrue(result.shape, shape)
        self.assertTrue(result.dtype, np.int16)
        self.assertTrue(np.any(result >= minval))
        self.assertTrue(np.any(result < maxval))

        vals, valfreqs = np.unique(np.ravel(result), return_counts=True)
        freqs = np.zeros(num_values)
        freqs[vals - minval] = valfreqs
        res = scipy.stats.chisquare(
            freqs, 
        )
        self.assertTrue(res.pvalue >= 0.05)

    def test_randint_single_support_value(self) -> None:
        key = self.rng_suite.PRNGKey(8025111)
        shape = (100,)
        minval = -4
        maxval = -3

        result = self.rng_suite.randint(key, shape, minval, maxval, np.int32)
        self.assertTrue(result.shape, shape)
        self.assertTrue(result.dtype, np.int32)
        self.assertTrue(np.all(result == -4))


class StrongRNGSuiteTests(RNGSuiteTests, unittest.TestCase):

    def setUp(self) -> None:
        self.rng_suite = d3p.random


class DebugRNGSuiteTests(RNGSuiteTests, unittest.TestCase):

    def setUp(self) -> None:
        self.rng_suite = d3p.random.debug


if __name__ == '__main__':
    unittest.main()
