import os

# allow example to find dppp without installing
import sys
sys.path.append(os.path.dirname(sys.path[0]))
#### 

from dppp.svi import per_example_log_density
from numpyro.svi import log_density

from numpyro.handlers import sample, seed, trace
import numpyro.distributions as dist
from numpyro.svi import log_density
import jax.numpy as np
import jax
import numpy as onp

onp.random.seed(0)

################################################################################
# a simple model
def model(obs):
    z_dist = sample('z', dist.Bernoulli(np.ones((10,))*0.5))
    x_dist = sample('x', dist.Normal(z_dist, np.ones((10,))), obs=obs)
    return x_dist

model = seed(model, jax.random.PRNGKey(0))

# generating toy data for model
obs = onp.random.rand(2, 10)

### all the following variants need to work
# variant 1. single z, broadcasted to all obs
z1 = np.array([[True, True, False, True, False, False, False, False, False, True]])
z1 = np.broadcast_to(z1, (2, z1.shape[1]))
pev1 = {'x', 'z'}

# variant 2. single z
z2 = np.array([True, True, False, True, False, False, False, False, False, True])
pev2 = {'x'}

# variant 2.1. single z, column
z21 = np.array([[True, True, False, True, False, False, False, False, False, True]])
pev21 = {'x'}

# variant 3. separate z for all obs
z3 = np.array([[True, True, False, True, False, False, False, False, False, True],
              [False, False, True, True, False, False, True, True, False, True]])
pev3 = {'x', 'z'}

zs = (z1, z2, z21, z3)
pevs = (pev1, pev2, pev21, pev3)

for z, pev in zip(zs, pevs):
    # computing expected per-sample log probability
    log_prob_z = dist.Bernoulli(np.ones((10,))*0.5).log_prob(z)
    log_prob_obs = dist.Normal(z, np.ones((10,))).log_prob(obs)
    per_ex_log_prob_z = np.sum(log_prob_z) / obs.shape[0]
    per_ex_log_prob_obs = np.sum(log_prob_obs, axis=1)
    per_ex_log_prob = per_ex_log_prob_z + per_ex_log_prob_obs

    # invoking per_example_log_density and comparing to expected
    d, _ = per_example_log_density(model, (obs,), {}, {'z': z}, pev)
    assert(np.allclose(per_ex_log_prob, d))

    # summing per-sample results and compare to numpyro's log_density function
    numpyro_d, _ = log_density(model, (obs,), {}, {'z': z})
    assert(np.allclose(np.sum(d), numpyro_d))

################################################################################
# a simple mixture model
def model2(N, d):
    mus = sample('mus', dist.Normal(np.zeros((2, d)), 1))
    z = sample('z', dist.Bernoulli(np.ones((N))*0.5))*1
    # z = sample('z', dist.Bernoulli(0.5))
    x = sample('x', dist.Normal(mus[z], 0.5))
    return x

# generating toy data for model2
d = 4
N = 10
# z = (1*(onp.random.rand(N)>0.5))
# mus = onp.random.randn(2*d).reshape(2, -1)
# x0 = onp.random.multivariate_normal(mus[0], onp.eye(d)*0.5, size=(N-onp.sum(z)))
# x1 = onp.random.multivariate_normal(mus[1], onp.eye(d)*0.5, size=onp.sum(z))
# x = onp.zeros((N, d))
# x[z==0,:] = x0
# x[z==1,:] = x1
## using the model to generate some data instead of specifying it manually:
tr = trace(seed(model2, jax.random.PRNGKey(6))).get_trace(N, d)
mus = tr['mus']['value']
z = tr['z']['value']*1
x = tr['x']['value']

# computing expected per-sample log probability
per_ex_log_prob_x = np.sum(dist.Normal(mus[z], 0.5).log_prob(x), axis=1)
per_ex_log_prob_z = dist.Bernoulli(0.5).log_prob(z)
log_prob_mus = np.sum(dist.Normal(np.zeros((2,d)),1).log_prob(mus))
per_ex_log_probs = per_ex_log_prob_x + per_ex_log_prob_z + (log_prob_mus / N)

# invoking per_example_log_density and comparing to expected
model2 = seed(model2, jax.random.PRNGKey(1))
d2, _ = per_example_log_density(model2, (N, d), {}, {'z': z, 'x': x, 'mus': mus}, {'x'})
assert(np.allclose(per_ex_log_probs, d2))

# summing per-sample results and compare to numpyro's log_density function
numpyro_d2, _ = log_density(model2, (N, d), {}, {'z': z, 'x': x, 'mus': mus})
assert(np.allclose(np.sum(d2), numpyro_d2))

################################################################################
# model for testing scalar
def model3(N):
    b = sample('b', dist.Normal(0., 1.))
    x = sample('x', dist.Normal(np.ones((N,1))*b, 1.))
    return x

# generating toy data for model3
N = 10
tr = trace(seed(model3, jax.random.PRNGKey(5))).get_trace(N)
x = tr['x']['value']
b_true = tr['b']['value']

# computing expected per-sample log probability
per_ex_log_prob_x = np.sum(dist.Normal(np.ones((N,1))*b_true, 1.).log_prob(x), axis=1)
log_prob_b = dist.Normal(0., 1.).log_prob(b_true)
per_ex_log_probs = per_ex_log_prob_x + (log_prob_b/N)

# invoking per_example_log_density and comparing to expected
model3 = seed(model3, jax.random.PRNGKey(2))
d3, _ = per_example_log_density(model3, (N,), {}, {'b': b_true, 'x': x}, {'x'})
assert(np.allclose(per_ex_log_probs, d3))

# summing per-sample results and compare to numpyro's log_density function
numpyro_d3, _ = log_density(model3, (N,), {}, {'b': b_true, 'x': x})
assert(np.allclose(np.sum(d3), numpyro_d3))

################################################################################
# testing scalar output
per_example_variables_sweep = [{}, None, {'someUnseenVar'}]
for pev in per_example_variables_sweep:
    d4, _ = per_example_log_density(model3, (N,), {}, {'b': b_true, 'x': x}, pev)
    assert(d4.shape==(1,))
    numpyro_d4, _ = log_density(model3, (N,), {}, {'b': b_true, 'x': x})
    assert(np.allclose(d4, numpyro_d4))

print("success")