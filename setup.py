# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2019- d3p Developers and their Assignees

from __future__ import absolute_import, division, print_function
import os

from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

# read version number
import importlib
spec = importlib.util.spec_from_file_location("version_module", "d3p/version.py")
version_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(version_module)

_version = version_module.VERSION

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

_numpyro_version_lower_constraint = '>=0.8.0'
_numpyro_version_optimistic_upper_constraint = ', < 2.0.0'

setup(
    name='d3p',
    python_requires='>=3.7',
    version=_version,
    description='Differentially-Private Probabilistic Programming using NumPyro and the differentially-private variational inference algorithm',
    packages=find_packages(include=['d3p', 'd3p.*']),
    author='FCAI R4 @ Helsinki University and Aalto University',
    install_requires=[
        f'numpyro[cpu] {_numpyro_version_lower_constraint}{_numpyro_version_optimistic_upper_constraint}',
        'jax >= 0.2.20',
        'fourier-accountant >= 0.12.0, < 1.0.0',
        'jax-chacha-prng >= 1, < 2',
    ],
    extras_require={
        'examples': ['matplotlib'],
        'compatible-dependencies': [
            "numpyro==0.10.1",
            "jax[cpu]==0.3.23",
        ],
        'tpu': "numpyro[tpu]",
        'cpu': "",
        'cuda': "numpyro[cuda]"
    },
    long_description=long_description,
    long_description_content_type='text/markdown',
    tests_require=[],
    test_suite='tests',
    keywords='probabilistic machine learning bayesian statistics differential-privacy',
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3'
    ],
)
