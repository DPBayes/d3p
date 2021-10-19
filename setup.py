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


_available_cuda_versions = ['101', '102', '110', '111']

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

_numpyro_version_lower_constraint = '>=0.6.0'
_numpyro_version_optimistic_upper_constraint = ', < 2.0.0'

setup(
    name='d3p',
    python_requires='>=3.6',
    version=_version,
    description='Differentially-Private Probabilistic Programming',
    packages=find_packages(include=['d3p', 'd3p.*']),
    author='PADS @ Helsinki University and Aalto University',
    install_requires=[
        f'numpyro {_numpyro_version_lower_constraint}{_numpyro_version_optimistic_upper_constraint}',
        'fourier-accountant >= 0.12.0, < 1.0.0'
    ],
    extras_require={
        'examples': ['matplotlib'],
        'compatible-dependencies': [
            "numpyro == 0.8.0",
            "jax[minimum-jaxlib]==0.2.22"
        ],
        'tpu': f"numpyro[tpu]",
        'cpu': f"numpyro[cpu]",
        **{
            f'cuda{version}': [f'numpyro[cuda{version}]']
            for version in _available_cuda_versions
        }
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
