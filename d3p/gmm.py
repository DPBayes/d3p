# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2019- d3p Developers and their Assignees

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpyro.distributions as dists


class GaussianMixture(dists.Distribution):
    r""" A distribution class for the (one-dimensional) Gaussian mixture (or mixture of Gaussians) model.

    The probability density function is defined as
    .. math::

        p(x; \mathbf{\pi}, \mathbf{\mu}, \mathbf{\sigma}) = \sum_{k=1}^K \pi_k \mathcal{N}(x; \mu_k, \sigma_k^2)

    where :math:`\mathbf{\mu}`, :math:`\mathbf{\scale}` are vectors containing the location and scale parameters
    for each of the mixture components and :math:`\mathbf{\pi}` contains the weights/probabilities for each component.

    The generative process of the Gaussian mixture model can be intuitively understood
    by randomly selecting a component `k` according to probabilities :math:`\mathbf{\pi}`
    and then sampling :math:`x \sim \mathcal{N}(x; \mu_k, \sigma_k^2)`.
    """

    arg_constraints = {
        "mixture_probabilities": dists.constraints.simplex,
        "locs": dists.constraints.real,
        "scales": dists.constraints.positive,
    }
    support = dists.constraints.real
    reparametrized_params = ["mixture_probabilities", "locs", "scales"]

    def __init__(self, locs, scales, mixture_probabilities, validate_args=None):
        """ Initializes the Gaussian mixture model with given probabilities/weights, locations
        and scales for all components. Parameters are given in form of arrays and must
        have identical length along the first dimension, which corresponds to the
        individual mixture components.

        Parameters `locs` and `scales` may be multidimensional. In that case, all
        dimensions after the first are be treated as containing independent one-dimensional
        mixture models that share the same weights.

        :param locs: Array of shape (k, *) containing the location for each mixture component.
        :param scales: Array of shape (k, *) containing the scale for each mixture component.
        :param mixture_probabilities: Array of shape (k,) containing the probabilities/weights for mixture component.
            Must sum up to 1.
        :param validate_args: Whether to enable validation of distribution
            parameters and arguments to `.log_prob` method.
        """
        self.mixture_probabilities = jnp.array(mixture_probabilities)
        self.locs = jnp.array(locs)
        self.scales = jnp.array(scales)

        batch_shape = ()
        event_shape = self.locs.shape[1:]
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        per_component_log_prob = jax.vmap(
            lambda loc, scale: dists.Normal(loc, scale).log_prob(value),
            out_axes=-1
        )(self.locs, self.scales)

        log_pis = jnp.log(self.mixture_probabilities)

        # sum log-likelihood contributions from event dimensions
        per_component_log_prob = per_component_log_prob.sum(axis=-2)

        # "sum" over components
        loglik = logsumexp(per_component_log_prob + log_pis, axis=-1)
        return loglik

    def sample(self, key, sample_shape=()):
        return self.sample_with_intermediates(key, sample_shape)[0]

    def sample_with_intermediates(self, key, sample_shape=()):
        component_key, samples_key = jax.random.split(key)
        zs = dists.CategoricalProbs(self.mixture_probabilities).sample(component_key, sample_shape)
        xs = dists.Normal(self.locs[zs], self.scales[zs]).sample(samples_key)
        return xs, (zs,)

    @property
    def mean(self):
        return (self.mixture_probabilities * self.locs).sum()

    @property
    def variance(self):
        return (self.mixture_probabilities * (self.scales**2 + self.locs**2)) - self.mean()**2

    @property
    def num_components(self):
        return self.mixture_probabilities.shape[-1]
