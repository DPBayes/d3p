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

""" Stochastic Variational Inference implementation with per-example gradient
    clipping and differnetially-private perturbation.
"""
from typing import Any, NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from numpyro.infer.svi import SVI, SVIState
from numpyro.handlers import seed, trace, substitute, block

from d3p.util import example_count
import d3p.random as strong_rng

from fourier_accountant.compute_eps import get_epsilon_R
from fourier_accountant.compute_delta import get_delta_R

PRNGState = Any


class DPSVIState(NamedTuple):
    optim_state: Any
    rng_key: PRNGState
    observation_scale: float


def get_observations_scale(model, model_args, model_kwargs, params):
    """
    Traces through a model to extract the scale applied to observation log-likelihood.
    """

    # todo(lumip): is there a way to avoid tracing through the entire model?
    #       need to experiment with effect handlers and what exactly blocking achieves
    model = substitute(seed(model, 0), data=params)
    model = block(model, lambda msg: msg['type'] != 'sample' or not msg['is_observed'])
    model_trace = trace(model).get_trace(*model_args, **model_kwargs)
    scales = np.unique(
        [msg['scale'] if msg['scale'] is not None else 1 for msg in model_trace.values()]
    )

    if len(scales) > 1:
        raise ValueError(
            "The model received several observation sites with different example counts."
            " This is not supported in DPSVI."
        )
    elif len(scales) == 0:
        return 1.

    return scales[0]


def full_norm(vector_parts, ord=2):
    """Computes the total norm over a list of values (of any shape) or a jax
    tree by treating them as a single large vector.

    :param vector_parts: A jax tree of values that make up
        the vector to compute the norm over.
    :param ord: Order of the norm. May take any value possible for
    `numpy.linalg.norm`.
    :return: The indicated norm over the full vector.
    """
    list_of_parts = jax.tree_util.tree_leaves(vector_parts)

    if list_of_parts is None or len(list_of_parts) == 0:
        return 0.

    flattened = tuple(g.ravel() for g in list_of_parts)
    gradients = jnp.concatenate(flattened)
    assert(len(gradients.shape) == 1)
    norm = jnp.linalg.norm(gradients, ord=ord)
    return norm


def normalize_gradient(gradient_parts, ord=2):
    """Normalizes a gradient by its total norm.

    The norm is computed by interpreting the given list of parts as a single
    vector (see `full_norm`).

    :param gradient_parts: A jax tree of values (of any shape) that make up
        the overall gradient vector.
    :return: Normalized gradients given in the same format/layout/shape as
        gradient_parts.
    """
    norm_inv = 1./full_norm(gradient_parts, ord=ord)
    normalized = jax.tree_util.tree_map(lambda g: norm_inv * g, gradient_parts)
    return normalized


def clip_gradient(gradient_parts, c):
    """Clips the total norm of a gradient by a given value C.

    The norm is computed by interpreting the given list of parts as a single
    vector (see `full_norm`). Each entry is then scaled by the factor
    (1/max(1, norm/C)) which effectively clips the norm to C. Additionally,
    the gradient can be scaled by a given factor before clipping.

    :param gradient_parts: A jax tree that represents the overall gradient vector.
    :param c: The clipping threshold C.
    :return: Clipped gradients given in the same format/layout/shape as
        gradient_parts.
    """
    if c == 0.:
        raise ValueError("The clipping threshold must be greater than 0.")
    norm = full_norm(gradient_parts)
    clip_scaling_factor = 1./jnp.maximum(1., norm/c)
    clipped_grads = jax.tree_util.tree_map(lambda g: clip_scaling_factor * g, gradient_parts)
    return clipped_grads


class DPSVI(SVI):
    """ Differentially-Private Stochastic Variational Inference given a per-example
    loss objective and a gradient clipping threshold.

    This is identical to numpyro's `SVI` but adds differential privacy by
    clipping gradients per example to the given clipping_threshold and
    perturbing the batch gradient with noise determined by sigma*clipping_threshold.

    To obtain the per-example gradients, the `per_example_loss_fn` is evaluated
    for (and the gradient take wrt) each example in a vectorized manner (using
    `jax.vmap`).

    For this to work `per_example_loss_fn` must be able to deal with batches
    of single examples. The leading batch dimension WILL NOT be stripped away,
    however, so a `per_example_loss_fn` that can deal with arbitrarily sized batches
    suffices. Take special care that the loss function scales the likelihood
    contribution of the data properly wrt to batch size and total example count
    (use e.g. the `numpyro.scale` or the convenience `minibatch` context managers
    in the `model` and `guide` functions where appropriate).

    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param per_example_loss_fn: ELBo loss, i.e. negative Evidence Lower Bound,
        to minimize, per example.
    :param optim: an instance of :class:`~numpyro.optim._NumPyroOptim`.
    :param clipping_threshold: The clipping threshold C to which the norm
        of each per-example gradient is clipped.
    :param dp_scale: Scale parameter for the Gaussian mechanism applied to
        each dimension of the batch gradients.
    :param rng_suite: Optional. The PRNG suite to use. Defaults to the
        cryptographically-secure d3p.random.
    :param clip_unscaled_observations: Optional. If True, the upscaling of the log-likelihood
        of individual data examples in the model is undone before clipping. This has the effect of
        changing the interpretation of the clipping threshold C to bound the effect
        of unscaled individual example log-likelihoods. This unscaling is undone after
        applying privacy perturbations so that the final gradients have the correct scale.
        Defaults to True.
    :param static_kwargs: Static arguments for the model / guide, i.e. arguments
        that remain constant during fitting.
    """

    def __init__(
            self,
            model,
            guide,
            optim,
            per_example_loss,
            clipping_threshold,
            dp_scale,
            rng_suite=strong_rng,
            clip_unscaled_observations=True,
            **static_kwargs
        ):  # noqa: E121, E125

        self._clipping_threshold = clipping_threshold
        self._dp_scale = dp_scale
        self._rng_suite = rng_suite
        self._clip_unscaled_observations = clip_unscaled_observations

        if (not np.isfinite(clipping_threshold)):
            raise ValueError("clipping_threshold must be finite!")

        super().__init__(model, guide, optim, per_example_loss, **static_kwargs)

    @staticmethod
    def _update_state_rng(dp_svi_state: DPSVIState, rng_key: PRNGState) -> DPSVIState:
        return DPSVIState(
            dp_svi_state.optim_state,
            rng_key,
            dp_svi_state.observation_scale
        )

    @staticmethod
    def _update_state_optim_state(dp_svi_state: DPSVIState, optim_state: Any) -> DPSVIState:
        return DPSVIState(
            optim_state,
            dp_svi_state.rng_key,
            dp_svi_state.observation_scale
        )

    def _split_rng_key(self, dp_svi_state: DPSVIState, count: int = 1) -> Tuple[DPSVIState, Sequence[PRNGState]]:
        rng_key = dp_svi_state.rng_key
        split_keys = self._rng_suite.split(rng_key, count+1)
        return DPSVI._update_state_rng(dp_svi_state, split_keys[0]), split_keys[1:]

    def init(self, rng_key, *args, **kwargs):
        jax_rng_key = self._rng_suite.convert_to_jax_rng_key(rng_key)
        svi_state = super().init(jax_rng_key, *args, **kwargs)

        if svi_state.mutable_state is not None:
            raise RuntimeError("Mutable state is not supported.")

        observation_scale = 1.0
        if self._clip_unscaled_observations:
            model_kwargs = dict(kwargs)
            model_kwargs.update(self.static_kwargs)

            one_element_batch = [
                jnp.expand_dims(a[0], 0) for a in args
            ]

            # note: DO use super().get_params here to get constrained/transformed params
            #  for use in get_observations_scale (svi_state.optim_state holds unconstrained params)
            params = super().get_params(svi_state)
            observation_scale = get_observations_scale(
                self.model, one_element_batch, model_kwargs, params
            )

        return DPSVIState(svi_state.optim_state, rng_key, observation_scale)

    def _compute_per_example_gradients(self, dp_svi_state, step_rng_key, *args, **kwargs):
        """ Computes the raw per-example gradients of the model.

        This is the first step in a full update iteration.

        :param dp_svi_state: The current state of the DPSVI algorithm.
        :param step_rng_key: RNG key for this step.
        :param args: Arguments to the loss function.
        :param kwargs: All keyword arguments to model or guide.
        :returns: tuple consisting of the updated DPSVI state, an array of loss
            values per example, a jax tuple tree of per-example gradients
            per parameter site (each site's gradients have shape (batch_size, *parameter_shape)),
            and an integer denoting the batch size
        """
        jax_rng_key = self._rng_suite.convert_to_jax_rng_key(step_rng_key)

        # note: do NOT use self.get_params here; that applies constraint transforms for end-consumers of the parameters
        # but internally we maintain and optimize on unconstrained params
        # (they are constrained in the loss function so that we get the correct
        # effect of the constraint transformation in the gradient)
        params = self.optim.get_params(dp_svi_state.optim_state)

        obs_scale = dp_svi_state.observation_scale

        # we wrap the per-example loss (ELBO) to make it easier "digestable"
        # for jax.vmap(jax.value_and_grad()): slighly reordering parameters; fixing kwargs, model and guide
        def wrapped_px_loss(prms, rng_key, loss_args):
            # Vmap removes leading dimensions, we re-add those in a wrapper for loss so
            # that loss/the model can be oblivious of this.
            # Additionally we scale down losses and gradients by the inverse of the
            # observation scale (= dataset size) so that the log-likelihood part for each element is unscaled.
            # This allows the coice of the clipping threshold to be agnostic of the data set size.
            new_args = (jnp.expand_dims(arg, 0) for arg in loss_args)
            return 1/obs_scale * self.loss.loss(
                rng_key, self.constrain_fn(prms), self.model, self.guide,
                *new_args, **kwargs, **self.static_kwargs
            )

        # note: the splitting of the rng key guarantees separate randomness for each example in the batch
        #       for the sample from the guide as well as all latent variables in the model.
        #       While it would arguably be better to keep the guide sample fixed
        #       for all elements in the batch (be using the same rng key for all)
        #       to more closely mirror the standard SVI implementation, this would result in all latent
        #       samples to use the same randomness over the examples, which is something we certainly don't want.
        batch_size = example_count(args[0])
        px_rng_keys = jax.random.split(jax_rng_key, batch_size)

        px_value_and_grad_fn = jax.vmap(jax.value_and_grad(wrapped_px_loss), in_axes=(None, 0, 0))
        px_losses, px_grads = px_value_and_grad_fn(params, px_rng_keys, args)

        # we will not be doing anything privacy-related with the loss value,
        # so let's revert the downscaling we applied above; we only want that for the gradients.
        px_losses *= obs_scale

        return dp_svi_state, px_losses, px_grads, batch_size

    def _clip_gradients(self, dp_svi_state, px_grads):
        """ Clips each per-example gradient.

        This is the second step in a full update iteration.

        :param dp_svi_state: The current state of the DPSVI algorithm.
        :param px_grads: Jax tree of per-example gradients as returned
            by `_compute_per_example_gradients`
        :returns: tuple consisting of the updated svi state and a jax tree of
            clipped per-example gradients per site.
        """
        px_clipped_grads = jax.vmap(
            lambda px_grad: clip_gradient(px_grad, self._clipping_threshold), in_axes=0
        )(px_grads)

        return dp_svi_state, px_clipped_grads

    def _combine_gradients(self, px_clipped_grads, px_loss):
        """ Combines the per-example gradients into the batch gradient and
            applies the batch gradient transformation given as
            `batch_grad_manipulation_fn`.

        This is the third step of a full update iteration.

        :param px_clipped_grads: Clipped per-example gradients as returned
            by `_clip_gradients`
        :param px_loss: Array of per-example loss values as output by
            `_compute_per_example_gradients`.
        :returns: tuple consisting of the updated svi state, the loss value for
            the batch and a jax tree of batch gradients per parameter site.
        """

        loss_val = jnp.mean(px_loss, axis=0)
        avg_clipped_grads = jax.tree_util.tree_map(lambda px_grads_site: jnp.mean(px_grads_site, axis=0), px_clipped_grads)

        return loss_val, avg_clipped_grads

    def _perturb_and_reassemble_gradients(
            self, dp_svi_state, step_rng_key, avg_clipped_grads, batch_size
        ):  # noqa: E121, E125
        """ Perturbs the gradients using Gaussian noise.

        This is the fourth step of a full update iteration.

        :param dp_svi_state: The current state of the DPSVI algorithm.
        :param step_rng_key: RNG key for this step.
        :param avg_clipped_grads: Jax tree of batch gradients for each parameter site
        :param batch_size: Size of the training batch.
        """
        # because avg_clipped_grads is the average of gradients clipped to norm self._clipping_threshold
        sensitivity = (self._clipping_threshold / batch_size)
        perturbation_scale = self._dp_scale * sensitivity
        perturbed_grads = self.perturbation_function(
            self._rng_suite, step_rng_key, avg_clipped_grads, perturbation_scale
        )

        # Remember that in the very beginning we scaled down the loss and gradients
        # by 1/observation_scale? Now we revert this, so that the final gradient is scaled as
        # expected by the user
        obs_scale = dp_svi_state.observation_scale
        perturbed_grads = jax.tree_util.tree_map(lambda grad_site: grad_site * obs_scale, perturbed_grads)

        return dp_svi_state, perturbed_grads

    def _apply_gradient(self, dp_svi_state, perturbed_grads):
        """ Takes a (batch) gradient step in parameter space using the specified
            optimizer.

        This is the fifth and last step of a full update iteration.
        :param dp_svi_state: The current state of the DPSVI algorithm.
        :param perturbed_grads: Jax tree of batch gradients per parameter site,
            as returned by `_perturb_and_reassemble_gradients`.
        :returns: tuple consisting of the updated svi state.
        """
        optim_state = dp_svi_state.optim_state
        new_optim_state = self.optim.update(perturbed_grads, optim_state)

        dp_svi_state = self._update_state_optim_state(dp_svi_state, new_optim_state)
        return dp_svi_state

    def update(self, svi_state, *args, **kwargs):
        """ Takes a single step of SVI (possibly on a batch / minibatch of data),
        using the optimizer.

        :param svi_state: Current state of SVI.
        :param args: Arguments to the model / guide (these can possibly vary during
            the course of fitting).
        :param kwargs: Keyword arguments to the model / guide (these can possibly vary
            during the course of fitting).
        :return: Tuple of `(svi_state, loss)`, where `svi_state` is the updated
            DPSVI state and `loss` the value of the ELBO on the training batch for
            the given `svi_state`.
        """

        svi_state, update_rng_keys = self._split_rng_key(svi_state, 2)
        gradient_rng_key, perturbation_rng_key = update_rng_keys

        svi_state, px_losses, px_grads, batch_size = \
            self._compute_per_example_gradients(svi_state, gradient_rng_key, *args, **kwargs)

        svi_state, px_clipped_grads = \
            self._clip_gradients(
                svi_state, px_grads
            )

        loss, avg_clipped_grads = self._combine_gradients(
            px_clipped_grads, px_losses
        )

        svi_state, perturbed_grads = self._perturb_and_reassemble_gradients(
            svi_state, perturbation_rng_key, avg_clipped_grads, batch_size
        )

        svi_state = self._apply_gradient(svi_state, perturbed_grads)

        return svi_state, loss

    def evaluate(self, svi_state: DPSVIState, *args, **kwargs):
        """ Evaluates the ELBO given the current parameter values / DPSVI state
        and (a minibatch of) data.

        :param svi_state: Current state of DPSVI.
        :param args: Arguments to the model / guide.
        :param kwargs: Keyword arguments to the model / guide.
        :return: ELBO loss given the current parameter values
            (held within `svi_state.optim_state`).
        """
        # we split to have the same seed as `update_fn` given an svi_state
        jax_rng_key = self._rng_suite.convert_to_jax_rng_key(self._rng_suite.split(svi_state.rng_key, 1)[0])
        numpyro_svi_state = SVIState(svi_state.optim_state, None, jax_rng_key)
        return super().evaluate(numpyro_svi_state, *args, **kwargs)

    def _validate_epochs_and_iter(self, num_epochs, num_iter, q):
        if num_epochs is not None:
            num_iter = num_epochs / q
        if num_iter is None:
            raise ValueError("A value must be supplied for either num_iter or num_epochs")
        return num_iter

    def get_epsilon(self, target_delta, q, num_epochs=None, num_iter=None):
        num_iter = self._validate_epochs_and_iter(num_epochs, num_iter, q)

        eps = get_epsilon_R(target_delta, self._dp_scale, q, ncomp=num_iter)
        return eps

    def get_delta(self, target_epsilon, q, num_epochs=None, num_iter=None):
        num_iter = self._validate_epochs_and_iter(num_epochs, num_iter, q)

        eps = get_delta_R(target_epsilon, self._dp_scale, q, ncomp=num_iter)
        return eps

    @staticmethod
    def perturbation_function(
            rng_suite, rng: PRNGState, values: Any, perturbation_scale: float
        ) -> Any:  # noqa: E121, E125
        """ Perturbs given values using Gaussian noise.

        `values` can be a list of array-like objects. Each value is independently
        perturbed by adding noise sampled from a Gaussian distribution with a
        standard deviation of `perturbation_scale`.

        :param rng: Jax PRNGKey for perturbation randomness.
        :param values: Jax tree of which each leaf will be perturbed element-wise.
        :param perturbation_scale: The scale/standard deviation of the noise
            distribution.
        """
        def perturb_one(a: jnp.ndarray, site_rng: PRNGState) -> jnp.ndarray:
            """ perturbs a single gradient site """
            noise = rng_suite.normal(site_rng, a.shape) * perturbation_scale
            return a + noise

        values, tree_def = jax.tree_util.tree_flatten(values)
        per_site_rngs = rng_suite.split(rng, len(values))
        perturbed_values = (
            perturb_one(grad, site_rng)
            for grad, site_rng in zip(values, per_site_rngs)
        )

        perturbed_values = jax.tree_util.tree_unflatten(tree_def, perturbed_values)
        return perturbed_values
