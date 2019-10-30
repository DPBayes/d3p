from __future__ import absolute_import, division, print_function

import os
import sys

from setuptools import find_packages, setup

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))

numpyro_commit = '101753fe69f00e8200c27789c570ff607c93cbd3'

setup(
    name='dppp',
    version='0.1.0',
    description='Differentially-Private Probabilistic Programming',
    packages=find_packages(include=['dppp', 'dppp.*']),
    author='PADS @ Helsinki University and Aalto University',
    install_requires=[
        'numpyro @ git+https://github.com/pyro-ppl/numpyro.git@{}#egg=numpyro'.format(numpyro_commit)
    ],
    extras_require={
        'examples': ['matplotlib'],
    },
    dependency_links=[
        'https://github.com/pyro-ppl/numpyro/tarball/{}#egg=numpyro'.format(numpyro_commit)
    ],
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
        'Programming Language :: Python :: 3.7',
    ],
)
