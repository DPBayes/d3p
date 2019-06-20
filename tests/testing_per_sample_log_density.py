import os

# allow example to find dppp without installing
import sys
sys.path.append(os.path.dirname(sys.path[0]))
#### 

from dppp.svi import per_sample_log_density

from numpyro.handlers import sample, seed, trace
import numpyro.distributions as dist
from numpyro.svi import log_density
import jax.numpy as np
import jax
import numpy as onp

def model(obs):
    z_dist = sample('z', dist.Bernoulli(np.ones((10,))*0.5))
    x_dist = sample('x', dist.Normal(z_dist, np.ones((10,))), obs=obs)
    return x_dist

model = seed(model, jax.random.PRNGKey(0))

z = np.array([True, True, False, True, False, False, False, False, False, True])
z = z.reshape(1, -1)
z = np.broadcast_to(z, (2, z.shape[1]))

obs = onp.random.rand(2, 10)
tr = trace(model).get_trace(obs)
d, t = per_sample_log_density(model, (obs,), {}, {'z': z})

print(d)
