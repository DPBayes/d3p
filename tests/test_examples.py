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
