""" tests that the minibatch context manager leads to correct scaling of the
affected sample sites in the numpyro.log_density method
"""

import os

# allow example to find dppp without installing
import sys
sys.path.append(os.path.dirname(sys.path[0]))
#### 


import jax.numpy as np
import jax
import numpy as onp

import numpyro.distributions as dist
from numpyro.svi import elbo
from numpyro.infer_util import log_density
from numpyro.handlers import seed
from numpyro.primitives import sample

from dppp.svi import minibatch

def model_fn(X, N=None, num_obs_total=None):
    if N is None:
        N = np.shape(X)[0]
    if num_obs_total is None:
        num_obs_total = N

    mu = sample("theta", dist.Normal(1.))
    with minibatch(N, num_obs_total=num_obs_total):
        X = sample("X", dist.Normal(mu), obs=X, sample_shape=(N,))
    return X, mu

model = seed(model_fn, jax.random.PRNGKey(0))
num_samples = 100
X, mu = model(None, num_samples)

for batch_size in range(10, 100, 10):
    batch = X[:batch_size]
    assert(np.shape(batch)[0] == batch_size)
    prior_log_prob = dist.Normal(1.).log_prob(mu)
    data_log_prob = np.sum(dist.Normal(mu).log_prob(batch))
    expected_log_joint = prior_log_prob + (num_samples/batch_size) * data_log_prob

    log_joint, _ = log_density(model, (batch,), {'num_obs_total': num_samples}, {"theta": mu})
    assert(np.allclose(expected_log_joint, log_joint))

print("success")
