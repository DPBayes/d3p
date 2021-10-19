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
from numpyro.handlers import seed, trace, substitute, condition
from d3p.util import unvectorize_shape_2d


def get_samples_from_trace(trace, with_intermediates=False):
    """ Extracts all sample values from a numpyro trace.

    :param trace: trace object obtained from `numpyro.handlers.trace().get_trace()`
    :param with_intermediates: If True, intermediate(/latent) samples from
        sample site distributions are included in the result.
    :return: Dictionary of sampled values associated with the names given
        via `sample()` in the model. If with_intermediates is True,
        dictionary values are tuples where the first element is the final
        sample values and the second element is a list of intermediate values.
    """
    samples = {
        k: (v['value'], v['intermediates']) if with_intermediates else v['value']
        for k, v in trace.items() if v['type'] == 'sample'
    }
    return samples


def sample_prior_predictive(
        rng_key,
        model,
        model_args,
        substitutes=None,
        with_intermediates=False,
        **kwargs
    ):  # noqa: E121,E125
    """ Samples once from the prior predictive distribution.

    Individual sample sites, as designated by `sample`, can be frozen to
    pre-determined values given in `substitutes`. In that case, values for these
    sites are not actually sampled but the value provided in `substitutes` is
    returned as the sample. This facilitates conditional sampling.

    Note that if the model function is written in such a way that it returns, e.g.,
    multiple observations from a single prior draw, the same is true for the
    values returned by this function.

    :param rng_key: Jax PRNG key
    :param model: Function representing the model using numpyro distributions
        and the `sample` primitive
    :param model_args: Arguments to the model function
    :param substitutes: An optional dictionary of frozen substitutes for
        sample sites.
    :param with_intermediates: If True, intermediate(/latent) samples from
        sample site distributions are included in the result.
    :param **kwargs: Keyword arguments passed to the model function.
    :return: Dictionary of sampled values associated with the names given
        via `sample()` in the model. If with_intermediates is True,
        dictionary values are tuples where the first element is the final
        sample values and the second element is a list of intermediate values.
    """
    if substitutes is None:
        substitutes = dict()
    model = seed(substitute(model, data=substitutes), rng_key)
    t = trace(model).get_trace(*model_args, **kwargs)
    return get_samples_from_trace(t, with_intermediates)


def sample_posterior_predictive(
        rng_key,
        model,
        model_args,
        guide,
        guide_args,
        params,
        with_intermediates=False,
        **kwargs
    ):  # noqa: E121, E125
    """ Samples once from the posterior predictive distribution.

    Note that if the model function is written in such a way that it returns, e.g.,
    multiple observations from a single posterior draw, the same is true for the
    values returned by this function.

    :param rng_key: Jax PRNG key
    :param model: Function representing the model using numpyro distributions
        and the `sample` primitive
    :param model_args: Arguments to the model function
    :param guide: Function representing the variational distribution (the guide)
        using numpyro distributions as well as the `sample` and `param` primitives
    :param guide_args: Arguments to the guide function
    :param params: A dictionary providing values for the parameters
        designated by call to `param` in the guide
    :param with_intermediates: If True, intermediate(/latent) samples from
        sample site distributions are included in the result.
    :param **kwargs: Keyword arguments passed to the model and guide functions.
    :return: Dictionary of sampled values associated with the names given
        via `sample()` in the model. If with_intermediates is True,
        dictionary values are tuples where the first element is the final
        sample values and the second element is a list of intermediate values.
    """
    model_rng_key, guide_rng_key = jax.random.split(rng_key)

    guide = seed(substitute(guide, data=params), guide_rng_key)
    guide_samples = get_samples_from_trace(
        trace(guide).get_trace(*guide_args, **kwargs), with_intermediates
    )

    model_params = dict(**params)
    if with_intermediates:
        model_params.update({k: v[0] for k, v in guide_samples.items()})
    else:
        model_params.update({k: v for k, v in guide_samples.items()})

    model = seed(substitute(model, data=model_params), model_rng_key)
    model_samples = get_samples_from_trace(
        trace(model).get_trace(*model_args, **kwargs), with_intermediates
    )

    guide_samples.update(model_samples)
    return guide_samples


def _sample_a_lot(rng_key, n, single_sample_fn):
    rng_keys = jax.random.split(rng_key, n)
    return jax.vmap(single_sample_fn)(rng_keys)


def sample_multi_prior_predictive(
        rng_key,
        n,
        model,
        model_args,
        substitutes=None,
        with_intermediates=False,
        **kwargs
    ):  # noqa: E121, E125
    """ Samples n times from the prior predictive distribution.

    Individual sample sites, as designated by `sample`, can be frozen to
    pre-determined values given in `substitutes`. In that case, values for these
    sites are not actually sampled but the value provided in `substitutes` is
    returned as the sample. This facilitates conditional sampling.

    Note that if the model function is written in such a way that it returns, e.g.,
    multiple observations, say n_model many, from a single prior draw, the same is
    true for the values returned by this function, i.e., this function will
    output n x n_model observations.

    :param rng_key: Jax PRNG key
    :param n: Number of draws from the prior predictive.
    :param model: Function representing the model using numpyro distributions
        and the `sample` primitive
    :param model_args: Arguments to the model function
    :param substitutes: An optional dictionary of frozen substitutes for
        sample sites.
    :param with_intermediates: If True, intermediate(/latent) samples from
        sample site distributions are included in the result.
    :param **kwargs: Keyword arguments passed to the model function.
    :return: Dictionary of sampled values associated with the names given
        via `sample()` in the model. If with_intermediates is True,
        dictionary values are tuples where the first element is the final
        sample values and the second element is a list of intermediate values.
    """
    def single_sample_fn(rng):
        return sample_prior_predictive(
            rng, model, model_args, substitutes=substitutes,
            with_intermediates=with_intermediates, **kwargs
        )
    return _sample_a_lot(rng_key, n, single_sample_fn)


def sample_multi_posterior_predictive(
        rng_key,
        n,
        model,
        model_args,
        guide,
        guide_args,
        params,
        with_intermediates=False,
        **kwargs
    ):  # noqa: E121, E125
    """ Samples n times from the posterior predictive distribution.

    Note that if the model function is written in such a way that it returns, e.g.,
    multiple observations, say n_model many, from a single posterior draw, the same is
    true for the values returned by this function, i.e., this function will
    output n x n_model observations.

    :param rng_key: Jax PRNG key
    :param model: Function representing the model using numpyro distributions
        and the `sample` primitive
    :param model_args: Arguments to the model function
    :param guide: Function representing the variational distribution (the guide)
        using numpyro distributions as well as the `sample` and `param` primitives
    :param guide_args: Arguments to the guide function
    :param params: A dictionary providing values for the parameters
        designated by call to `param` in the guide
    :param with_intermediates: If True, intermediate(/latent) samples from
        sample site distributions are included in the result.
    :param **kwargs: Keyword arguments passed to the model and guide functions.
    :return: Dictionary of sampled values associated with the names given
        via `sample()` in the model. If with_intermediates is True,
        dictionary values are tuples where the first element is the final
        sample values and the second element is a list of intermediate values.
    """
    def single_sample_fn(rng):
        return sample_posterior_predictive(
            rng, model, model_args, guide, guide_args, params,
            with_intermediates=with_intermediates, **kwargs
        )
    return _sample_a_lot(rng_key, n, single_sample_fn)


def map_args_obs_to_shape(obs, *args, **kwargs):
    return unvectorize_shape_2d(obs), kwargs, {'obs': obs}


def make_observed_model(model, obs_to_model_args_fn):
    """ Transforms a generative model function into one with fixed observations
    for likelihood evaluation in the SVI algorithm.

    :param model: Any generative model function using the numpyro `sample`
        primitive.
    :param obs_to_model_args_fn: A function mapping from an argument list compatible
        with SVI (i.e., accepting a batch of observations) to that of `model`. The
        mapping function can take arbitrary arguments and must return a tuple
        (args, kwargs, observations), where args and kwargs are passed to `model`
        as argument and keyword arguments and observations is a dictionary of
        observations for sample sites in `model` that will be fixed using the
        `observe` handler.
    """
    def transformed_model_fn(*args, **kwargs):
        mapped_args, mapped_kwargs, fixed_obs = obs_to_model_args_fn(*args, **kwargs)
        return condition(model, data=fixed_obs)(*mapped_args, **mapped_kwargs)
    return transformed_model_fn
