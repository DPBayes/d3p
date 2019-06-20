"""Logistic regression example from numpyro.

"""

import os

# allow example to find dppp without installing
import sys
sys.path.append(os.path.dirname(sys.path[0]))
#### 

import argparse
import time

import matplotlib.pyplot as plt

import jax.numpy as np
from jax import jit, lax, random
from jax.experimental import optimizers, stax
from jax.random import PRNGKey

import numpyro.distributions as dist
from numpyro.handlers import param, sample

from dppp.svi import per_sample_elbo, svi


def model(batch_X, batch_y, z_dim, **kwargs):
	"""Defines the generative probabilistic model: p(x|z)p(z)

	The model is conditioned on the observed data
	:param batch_X: a batch of predictors
	:parm batch_y: a batch of observations
	:param other keyword arguments' are accepted but ignored
	"""
	z_w = sample('w', dist.Normal(np.zeros((z_dim,)), np.ones((z_dim,)))) # prior is N(0,I)
	z_intercept = sample('intercept', dist.Normal(0,1)) # prior is N(0,I)
	logits = batch_X.dot(z_w)+z_intercept
	sample('obs', dist.Bernoulli(logits=logits), obs=batch_y)
	

def guide(z_dim):
	"""Defines the probabilistic guide for z (variational approximation to posterior): q(z) ~ p(z|q)
	"""
	z_w = sample('w', dist.Normal(np.zeros((z_dim,)), np.ones((z_dim,)))) # prior is N(0,I)
	z_intercept = sample('intercept', dist.Normal(0,1)) # prior is N(0,I)
	

## Create some toy data
import numpy as onp
onp.random.seed(123)
d = 20
w_true = onp.random.randn(d)
intercept_true = onp.random.randn()
N = 1000
X = onp.random.randn(N, d)
logits_true = X.dot(w_true)+intercept_true
sigmoid = lambda x : 1/(1+onp.exp(-x))
y = 1*(sigmoid(logits_true)>onp.random.rand(N))

## Train model
