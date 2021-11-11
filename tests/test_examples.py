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

import os.path
import unittest
import subprocess
from typing import Tuple


class ExampleTests(unittest.TestCase):

    @staticmethod
    def run_example(example_file: str) -> Tuple[int, str]:
        run_result = subprocess.run(
            ['python', os.path.join('.', 'examples', example_file)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        output = run_result.stdout.decode('utf8')
        return run_result.returncode, output

    def test_example_simple_gaussian_posterior(self) -> None:
        ret, output = self.run_example('simple_gaussian_posterior.py')
        if ret != 0:
            self.fail(f"Error in example:\n{output}")

    def test_example_logistic_regression(self) -> None:
        ret, output = self.run_example('logistic_regression.py')
        if ret != 0:
            self.fail(f"Error in example:\n{output}")

    def test_example_gaussian_mixture_model(self) -> None:
        ret, output = self.run_example('gaussian_mixture_model.py')
        if ret != 0:
            self.fail(f"Error in example:\n{output}")
