import jax
import jax.numpy as np
from jax.scipy.special import logsumexp
import numpyro.distributions as dist

class GaussianMixture(dist.Distribution):
    arg_constraints = {
        '_locs': dist.constraints.real, 
        '_scales': dist.constraints.positive, 
        '_pis' : dist.constraints.simplex
    }
    support = dist.constraints.real

    def __init__(self, locs=0., scales=1., pis=1.0, validate_args=None):
        self._locs, self._scales, self._pis = locs, scales, pis
        event_shape = np.shape(locs[0])
        super(GaussianMixture, self).__init__(
            event_shape=event_shape, validate_args=validate_args
        )

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        log_pis = np.log(self._pis)
        log_phis = np.array([
            dist.Normal(loc, scale).log_prob(value).sum(-1)
            for loc, scale
            in zip(self._locs, self._scales)
        ]).T
        return logsumexp(log_pis + log_phis, axis=-1)

    def sample(self, key, sample_shape=()):
        return self.sample_with_intermediates(key, sample_shape)[0]

    def sample_with_intermediates(self, key, sample_shape=()):
        vals_rng_key, pis_rng_key = jax.random.split(key, 2)
        z = dist.Categorical(self._pis).sample(pis_rng_key, sample_shape)
        vals = dist.Normal(self._locs[z], self._scales[z]).sample(vals_rng_key)
        return vals, [z]
