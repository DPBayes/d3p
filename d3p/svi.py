# Copyright 2019- d3p Developers and their Assignees

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
    manipulation capability.
"""
import functools

import jax
import jax.numpy as jnp
import numpy as np

from numpyro.infer.svi import SVI
import numpyro.distributions as dist
from numpyro.handlers import seed, trace, substitute, block

from d3p.util import map_over_secondary_dims, example_count

from fourier_accountant.compute_eps import get_epsilon_R
from fourier_accountant.compute_delta import get_delta_R

from collections import namedtuple

DPSVIState = namedtuple('DPSVIState', ('optim_state', 'rng_key', 'observation_scale'))


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


class CombinedLoss(object):

    def __init__(self, per_example_loss, combiner_fn=jnp.mean):
        self.px_loss = per_example_loss
        self.combiner_fn = combiner_fn

    def loss(self, rng_key, param_map, model, guide, *args, **kwargs):
        return self.combiner_fn(self.px_loss.loss(
            rng_key, param_map, model, guide, *args, **kwargs
        ))


def full_norm(list_of_parts_or_tree, ord=2):
    """Computes the total norm over a list of values (of any shape) or a jax
    tree by treating them as a single large vector.

    :param list_of_parts_or_tree: The list or jax tree of values that make up
        the vector to compute the norm over.
    :param ord: Order of the norm. May take any value possible for
    `numpy.linalg.norm`.
    :return: The indicated norm over the full vector.
    """
    if isinstance(list_of_parts_or_tree, list):
        list_of_parts = list_of_parts_or_tree
    else:
        list_of_parts = jax.tree_leaves(list_of_parts_or_tree)

    if list_of_parts is None or len(list_of_parts) == 0:
        return 0.

    ravelled = [g.ravel() for g in list_of_parts]
    gradients = jnp.concatenate(ravelled)
    assert(len(gradients.shape) == 1)
    norm = jnp.linalg.norm(gradients, ord=ord)
    return norm


def normalize_gradient(list_of_gradient_parts, ord=2):
    """Normalizes a gradient by its total norm.

    The norm is computed by interpreting the given list of parts as a single
    vector (see `full_norm`).

    :param list_of_gradient_parts: A list of values (of any shape) that make up
        the overall gradient vector.
    :return: Normalized gradients given in the same format/layout/shape as
        list_of_gradient_parts.
    """
    norm_inv = 1./full_norm(list_of_gradient_parts, ord=ord)
    normalized = [norm_inv * g for g in list_of_gradient_parts]
    return normalized


def clip_gradient(list_of_gradient_parts, c, rescale_factor=1.):
    """Clips the total norm of a gradient by a given value C.

    The norm is computed by interpreting the given list of parts as a single
    vector (see `full_norm`). Each entry is then scaled by the factor
    (1/max(1, norm/C)) which effectively clips the norm to C. Additionally,
    the gradient can be scaled by a given factor before clipping.

    :param list_of_gradient_parts: A list of values (of any shape) that make up
        the overall gradient vector.
    :param c: The clipping threshold C.
    :param rescale_factor: Factor to scale the gradient by before clipping.
    :return: Clipped gradients given in the same format/layout/shape as
        list_of_gradient_parts.
    """
    if c == 0.:
        raise ValueError("The clipping threshold must be greater than 0.")
    norm = full_norm(list_of_gradient_parts) * rescale_factor  # norm of rescale_factor * grad
    normalization_constant = 1./jnp.maximum(1., norm/c)
    f = rescale_factor * normalization_constant  # to scale grad to max(rescale_factor * grad, C)
    clipped_grads = [f * g for g in list_of_gradient_parts]
    return clipped_grads


def get_gradients_clipping_function(c, rescale_factor):
    """Factory function to obtain a gradient clipping function for a fixed
    clipping threshold C.

    :param c: The clipping threshold C.
    :param rescale_factor: Factor to scale the gradient by before clipping.
    :return: `clip_gradient` function with fixed threshold C. Only takes a
        list_of_gradient_parts as argument.
    """
    @functools.wraps(clip_gradient)
    def gradient_clipping_fn_inner(list_of_gradient_parts):
        return clip_gradient(list_of_gradient_parts, c, rescale_factor)
    return gradient_clipping_fn_inner


class DPSVI(SVI):
    """
    Differentially-Private Stochastic Variational Inference given a per-example
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
    :param static_kwargs: static arguments for the model / guide, i.e. arguments
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
            **static_kwargs
        ):  # noqa: E121, E125

        self._clipping_threshold = clipping_threshold
        self._dp_scale = dp_scale

        total_loss = CombinedLoss(per_example_loss, combiner_fn=jnp.mean)
        super().__init__(model, guide, optim, total_loss, **static_kwargs)

    @staticmethod
    def _update_state_rng(dp_svi_state: DPSVIState, rng_key) -> DPSVIState:
        return DPSVIState(
            dp_svi_state.optim_state,
            rng_key,
            dp_svi_state.observation_scale
        )

    @staticmethod
    def _update_state_optim_state(dp_svi_state: DPSVIState, optim_state) -> DPSVIState:
        return DPSVIState(
            optim_state,
            dp_svi_state.rng_key,
            dp_svi_state.observation_scale
        )

    @staticmethod
    def _split_rng_key(dp_svi_state: DPSVIState):
        rng_key = dp_svi_state.rng_key
        rng_key, split_key = jax.random.split(rng_key)
        return DPSVI._update_state_rng(dp_svi_state, rng_key), split_key

    def init(self, rng_key, *args, **kwargs):
        svi_state = super().init(rng_key, *args, **kwargs)

        params = self.optim.get_params(svi_state.optim_state)

        model_kwargs = dict(kwargs)
        model_kwargs.update(self.static_kwargs)

        one_element_batch = [
            jnp.expand_dims(a[0], 0) for a in args
        ]

        observation_scale = get_observations_scale(
            self.model, one_element_batch, model_kwargs, params
        )

        return DPSVIState(svi_state.optim_state, svi_state.rng_key, observation_scale)

    def _compute_per_example_gradients(self, dp_svi_state, *args, **kwargs):
        """ Computes the raw per-example gradients of the model.

        This is the first step in a full update iteration.

        :param dp_svi_state: The current state of the DPSVI algorithm.
        :param args: Arguments to the loss function.
        :param kwargs: All keyword arguments to model or guide.
        :returns: tuple consisting of the updated DPSVI state, an array of loss
            values per example, and a jax tuple tree of per-example gradients
            per parameter site (each site's gradients have shape (batch_size, *parameter_shape))
        """
        dp_svi_state, rng_key_step = self._split_rng_key(dp_svi_state)
        params = self.optim.get_params(dp_svi_state.optim_state)

        # we wrap the per-example loss (ELBO) to make it easier "digestable"
        # for jax.vmap(jax.value_and_grad()): slighly reordering parameters; fixing kwargs, model and guide
        def wrapped_px_loss(prms, rng_key, loss_args):
            # vmap removes leading dimensions, we re-add those in a wrapper for fun so
            # that fun can be oblivious of this
            new_args = (jnp.expand_dims(arg, 0) for arg in loss_args)
            return self.loss.px_loss.loss(
                rng_key, self.constrain_fn(prms), self.model, self.guide,
                *new_args, **kwargs, **self.static_kwargs
            )

        batch_size = jnp.shape(args[0])[0]  # todo: need checks to ensure this indexing is okay
        px_rng_keys = jax.random.split(rng_key_step, batch_size)

        px_value_and_grad = jax.vmap(jax.value_and_grad(wrapped_px_loss), in_axes=(None, 0, 0))
        per_example_loss, per_example_grads = px_value_and_grad(params, px_rng_keys, args)

        return dp_svi_state, per_example_loss, per_example_grads

    def _clip_gradients(self, dp_svi_state, px_gradients):
        """ Clips each per-example gradient.

        This is the second step in a full update iteration.

        :param dp_svi_state: The current state of the DPSVI algorithm.
        :param px_gradients: Jax tuple tree of per-example gradients as returned
            by `_compute_per_example_gradients`
        :returns: tuple consisting of the updated svi state, a list of
            transformed per-example gradients per site and the jax tree structure
            definition. The list is a flattened representation of the jax tree,
            the shape of per-example gradients per parameter is unaffected.
        """
        obs_scale = dp_svi_state.observation_scale

        # px_gradients is a jax tree of jax jnp.arrays of shape
        #   [batch_size, (param_shape)] for each parameter. flatten it out!
        px_grads_list, px_grads_tree_def = jax.tree_flatten(
            px_gradients
        )

        # scale the gradients by 1/obs_scale then clip them:
        #  in the loss, every single examples loss contribution is scaled by obs_scale
        #  but the clipping threshold assumes no scaling.
        #  we scale by the reciprocal to ensure that clipping is correct.
        clip_fn = get_gradients_clipping_function(self._clipping_threshold, 1./obs_scale)
        px_grads_list = jax.vmap(clip_fn, in_axes=0)(px_grads_list)

        return dp_svi_state, px_grads_list, px_grads_tree_def

    def _combine_gradients(self, px_grads_list, px_loss):
        """ Combines the per-example gradients into the batch gradient and
            applies the batch gradient transformation given as
            `batch_grad_manipulation_fn`.

        This is the third step of a full update iteration.

        :param px_grads_list: List of transformed per-example gradients as returned
            by `_apply_per_example_gradient_transformations`
        :param px_loss: Array of per-example loss values as output by
            `_compute_per_example_gradients`.
        :returns: tuple consisting of the updated svi state, the loss value for
            the batch and a jax tree of batch gradients per parameter site.
        """

        # get total loss and loss combiner vjp func
        loss_val, loss_combine_vjp = jax.vjp(self.loss.combiner_fn, px_loss)

        # loss_combine_vjp gives us the backward differentiation function
        #   from combined loss to per-example losses. we use it to get the
        #   (1xbatch_size) Jacobian and construct a function that takes
        #   per-example gradients and left-multiplies them with that jacobian
        #   to get the final combined gradient
        loss_jacobian = jnp.reshape(loss_combine_vjp(jnp.array(1.))[0], (1, -1))

        def loss_vjp(px_grads):
            return jnp.matmul(loss_jacobian, px_grads)

        # we map the loss combination vjp func over all secondary dimensions
        #   of gradient sites. This is necessary since some gradient
        #   sites might be matrices in itself (e.g., for NN layers), so a stack
        #   of those would be 3-dimensional and not admittable to jnp.matmul
        loss_vjp = map_over_secondary_dims(loss_vjp)

        # combine gradients for all parameters in the gradient jax tree
        #   according to the loss combination vjp func
        grads_list = tuple(map(loss_vjp, px_grads_list))

        return loss_val, grads_list

    def _perturb_and_reassemble_gradients(self, dp_svi_state, gradient_list, batch_size, px_grads_tree_def):
        """ Perturbs the gradients using Gaussian noise and reassembles the gradient tree.

        This is the fourth step of a full update iteration.

        :param dp_svi_state: The current state of the DPSVI algorithm.
        :param gradient_list: List of batch gradients for each parameter site
        :param batch_size: Size of the training batch.
        :param px_grads_tree_def: Jax tree definition for the gradient tree as
            returned by `_apply_per_example_gradient_transformations`.
        """
        dp_svi_state, perturbation_rng = self._split_rng_key(dp_svi_state)

        perturbation_scale = self._dp_scale * self._clipping_threshold / batch_size
        perturbed_grads_list = self.perturbation_function(
            perturbation_rng, gradient_list, perturbation_scale
        )

        # we multiply each parameter site by obs_scale to revert the downscaling
        # performed before clipping, so that the final gradient is scaled as
        # expected without DP
        obs_scale = dp_svi_state.observation_scale
        perturbed_grads_list = tuple(
            grad * obs_scale
            for grad in perturbed_grads_list
        )

        # reassemble the jax tree used by optimizer for the final gradients
        perturbed_grads = jax.tree_unflatten(
            px_grads_tree_def, perturbed_grads_list
        )

        return dp_svi_state, perturbed_grads

    def _apply_gradient(self, dp_svi_state, batch_gradient):
        """ Takes a (batch) gradient step in parameter space using the specified
            optimizer.

        This is the fifth and last step of a full update iteration.
        :param dp_svi_state: The current state of the DPSVI algorithm.
        :param batch_gradient: Jax tree of batch gradients per parameter site,
            as returned by `_combine_and_transform_gradient`.
        :returns: tuple consisting of the updated svi state.
        """
        optim_state = dp_svi_state.optim_state
        optim_state = self.optim.update(batch_gradient, optim_state)
        dp_svi_state = self._update_state_optim_state(dp_svi_state, optim_state)
        return dp_svi_state

    def update(self, svi_state, *args, **kwargs):
        svi_state, per_example_loss, per_example_grads = \
            self._compute_per_example_gradients(svi_state, *args, **kwargs)

        batch_size = example_count(per_example_loss)

        svi_state, per_example_grads, tree_def = \
            self._clip_gradients(
                svi_state, per_example_grads
            )

        loss, gradient = self._combine_gradients(
            per_example_grads, per_example_loss
        )

        svi_state, gradient = self._perturb_and_reassemble_gradients(
            svi_state, gradient, batch_size, tree_def
        )

        svi_state = self._apply_gradient(svi_state, gradient)

        return svi_state, loss

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
    def perturbation_function(rng, values, perturbation_scale):
        """ Perturbs given values using Gaussian noise.

        `values` can be a list of array-like objects. Each value is independently
        perturbed by adding noise sampled from a Gaussian distribution with a
        standard deviation of `perturbation_scale`.

        :param rng: Jax PRNGKey for perturbation randomness.
        :param values: Iterable of array-like where each value will be perturbed.
        :param perturbation_scale: The scale/standard deviation of the noise
            distribution.
        """
        def perturb_one(a, site_rng):
            noise = dist.Normal(0, perturbation_scale).sample(
                site_rng, sample_shape=a.shape
            )
            return a + noise

        per_site_rngs = jax.random.split(rng, len(values))
        values = tuple(
            perturb_one(grad, site_rng)
            for grad, site_rng in zip(values, per_site_rngs)
        )
        return values
