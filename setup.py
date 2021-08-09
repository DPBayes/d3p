from __future__ import absolute_import, division, print_function

import os
import sys

from setuptools import find_packages, setup

_available_cuda_versions = ['101', '102', '110', '111']

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

_numpyro_version_constraints = '>= 0.6.0, < 0.8.0'

setup(
    name='d3p',
    python_requires='>=3.6',
    version='0.2.0-alpha',
    description='Differentially-Private Probabilistic Programming',
    packages=find_packages(include=['d3p', 'd3p.*']),
    author='PADS @ Helsinki University and Aalto University',
    install_requires=[
        f'numpyro {_numpyro_version_constraints}',
        'fourier-accountant >= 0.12.0, < 1.0.0'
    ],
    extras_require={
        'examples': ['matplotlib'],
        'tpu': f"numpyro[tpu] {_numpyro_version_constraints}",
        'cpu': f"numpyro[cpu] {_numpyro_version_constraints}",
        **{
            f'cuda{version}': [f'numpyro[cuda{version}] {_numpyro_version_constraints}']
            for version in _available_cuda_versions
        }
    },
    long_description="",
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
