""" Stochastic Variational Inference implementation with per-example gradient
    manipulation capability.

Based on numpyro's `svi`:
    https://github.com/pyro-ppl/numpyro/blob/master/numpyro/svi.py
"""
import functools

import jax
from jax import random
import jax.numpy as np

from numpyro.infer.svi import SVI, SVIState
import numpyro.distributions as dist

from dppp.util import map_over_secondary_dims


def per_example_value_and_grad(fun, argnums=0, has_aux=False, holomorphic=False):
    value_and_grad_fun = jax.value_and_grad(fun, argnums, has_aux, holomorphic)
    return jax.vmap(value_and_grad_fun, in_axes=(None, 0))


class CombinedLoss(object):

    def __init__(self, per_example_loss, combiner_fn = np.mean):
        self.px_loss = per_example_loss
        self.combiner_fn = combiner_fn

    def loss(self, rng_key, param_map, model, guide, *args, **kwargs):
        return np.sum(self.px_loss.loss(
            rng_key, param_map, model, guide, *args, **kwargs
        ))


class TunableSVI(SVI):
    """
    Tunable Stochastic Variational Inference given a per-example loss objective
    and a loss combiner function.

    This is identical to numpyro's `SVI` but explicitely computes gradients
    per example (i.e. observed data instance) based on `per_example_loss_fn`
    before combining them to a total loss value using `loss_combiner_fn`.
    This allows manipulating the per-example gradients which has
    applications, e.g., in differentially private machine learning applications.

    To obtain the per-example gradients, the `per_example_loss_fn` is evaluated
    for (and the gradient take wrt) each example in a vectorized manner (using
    `jax.vmap`).
    
    For this to work, the following requirements are imposed upon
    `per_example_loss_fn`:
    - in per-example evaluation, the leading dimension of the batch (indicating
        the number of examples) is stripped away. The loss function must be able
        to handle this if it was originally designed to handle batched values
    - since it will be evaluated on each example, take special care that the
        loss function scales the likelihood contribution of the data properly
        wrt to batch size and total example count (use e.g. the `numpyro.scale`
        or the convenience `minibatch` context managers)


    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param per_example_loss_fn: ELBo loss, i.e. negative Evidence Lower Bound,
        to minimize, per example.
    :param optim: an instance of :class:`~numpyro.optim._NumpyroOptim`.
    :param per_example_grad_manipulation_fn: optional function that allows to
        manipulate the gradient for each sample.
    :param batch_grad_manipulation_fn: An optional function that allows to modify
        the total gradient. This gets called after applying the
        per_example_grad_manipulation_fn and loss_combiner_fn.
    :param static_kwargs: static arguments for the model / guide, i.e. arguments
        that remain constant during fitting.
    """

    def __init__(self, model, guide, optim, per_example_loss,
            per_example_grad_manipulation_fn=None,
            batch_grad_manipulation_fn=None, **static_kwargs):
        
        self.px_grad_manipulation_fn = per_example_grad_manipulation_fn
        self.batch_grad_manipulation_fn = batch_grad_manipulation_fn

        total_loss = CombinedLoss(per_example_loss)

        super().__init__(model, guide, optim, total_loss, **static_kwargs)

    def update(self, svi_state, *args, **kwargs):
        rng_key, rng_key_step = random.split(svi_state.rng_key, 2)
        params = self.optim.get_params(svi_state.optim_state)

        def wrapped_px_loss(x, loss_args):
            return self.loss.px_loss.loss(
                rng_key_step, self.constrain_fn(x), self.model, self.guide,
                *loss_args, **kwargs, **self.static_kwargs
            )

        per_example_loss, per_example_grads = per_example_value_and_grad(
            wrapped_px_loss
        )(params, args)

        # get total loss and loss combiner vjp func
        loss_val, loss_combine_vjp = jax.vjp(self.loss.combiner_fn, per_example_loss)
        
        # loss_combine_vjp gives us the backward differentiation function
        #   from combined loss to per-example losses. we use it to get the
        #   (1xbatch_size) Jacobian and construct a function that takes
        #   per-example gradients and left-multiplies them with that jacobian
        #   to get the final combined gradient
        loss_jacobian = np.reshape(loss_combine_vjp(np.array(1.))[0], (1, -1))
        # loss_vjp = lambda px_grads: np.sum(np.multiply(loss_jacobian, px_grads))
        loss_vjp = lambda px_grads: np.matmul(loss_jacobian, px_grads)

        # we map the loss combination vjp func over all secondary dimensions
        #   of gradient sites. This is necessary since some gradient
        #   sites might be matrices in itself (e.g., for NN layers), so a stack
        #   of those would be 3-dimensional and not admittable to np.matmul
        loss_vjp = map_over_secondary_dims(loss_vjp)

        # per_example_grads will be jax tree of jax np.arrays of shape
        #   [batch_size, (param_shape)] for each parameter. flatten it out!
        px_grads_list, px_grads_tree_def = jax.tree_flatten(
            per_example_grads
        )

        # if per-sample gradient manipulation is present, we apply it to
        #   each gradient site in the tree
        if self.px_grad_manipulation_fn:
            # apply per-sample gradient manipulation, if present
            px_grads_list = jax.vmap(
                self.px_grad_manipulation_fn, in_axes=0
            )(
                px_grads_list
            )
            # todo(lumip, all): by flattening the tree before passing it into
            #   gradient manipulation, we lose all information on which value
            #   belongs to which parameter. on the other hand, we have plain and
            #   straightforward access to the values, which might be all we need.
            #   think about whether that is okay or whether ps_grad_manipulation_fn
            #   should just get the whole tree per sample to get all available
            #   information

        # combine gradients for all parameters in the gradient jax tree
        #   according to the loss combination vjp func
        grads_list = tuple(map(loss_vjp, px_grads_list))

        # apply batch gradient modification (e.g., DP noise perturbation) (if any)
        if self.batch_grad_manipulation_fn:
            grads_list = self.batch_grad_manipulation_fn(grads_list)

        # reassemble the jax tree used by optimizer for the final gradients
        grads = jax.tree_unflatten(
            px_grads_tree_def, grads_list
        )

        optim_state = self.optim.update(grads, svi_state.optim_state)
        return SVIState(optim_state, rng_key), loss_val


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
    gradients = np.concatenate(ravelled)
    assert(len(gradients.shape) == 1)
    norm = np.linalg.norm(gradients, ord=ord)
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

def clip_gradient(list_of_gradient_parts, c):
    """Clips the total norm of a gradient by a given value C.

    The norm is computed by interpreting the given list of parts as a single
    vector (see `full_norm`). Each entry is then scaled by the factor
    (1/max(1, norm/C)) which effectively clips the norm to C.

    :param list_of_gradient_parts: A list of values (of any shape) that make up
        the overall gradient vector.
    :param c: The clipping threshold C.
    :return: Clipped gradients given in the same format/layout/shape as
        list_of_gradient_parts.
    """
    if c == 0.:
        raise ValueError("The clipping threshold must be greater than 0.")
    norm = full_norm(list_of_gradient_parts)
    normalization_constant = 1./np.maximum(1., norm/c)
    clipped_grads = [g*normalization_constant for g in list_of_gradient_parts]
    # assert(np.all(full_gradient_norm(clipped_grads)<c)) # jax doesn't like this
    return clipped_grads

def get_gradients_clipping_function(c):
    """Factory function to obtain a gradient clipping function for a fixed
    clipping threshold C.

    :param c: The clipping threshold C.
    :return: `clip_gradient` function with fixed threshold C. Only takes a
        list_of_gradient_parts as argument.
    """
    @functools.wraps(clip_gradient)
    def gradient_clipping_fn_inner(list_of_gradient_parts):
        return clip_gradient(list_of_gradient_parts, c)
    return gradient_clipping_fn_inner


class DPSVI(TunableSVI):
    """
    Differentially-Private Stochastic Variational Inference given a per-example
    loss objective and a gradient clipping threshold.

    This is identical to numpyro's `svi` but adds differential privacy by
    clipping gradients per example and perturbing the batch gradient.

    To obtain the per-example gradients, the `per_example_loss_fn` is evaluated
    for (and the gradient take wrt) each example in a vectorized manner (using
    `jax.vmap`).
    
    For this to work, the following requirements are imposed upon
    `per_example_loss_fn`:
    - in per-example evaluation, the leading dimension of the batch (indicating
        the number of examples) is stripped away. The loss function must be able
        to handle this if it was originally designed to handle batched values
    - since it will be evaluated on each example, take special care that the
        loss function scales the likelihood contribution of the data properly
        wrt to batch size and total example count (use e.g. the `numpyro.scale`
        or the convenience `minibatch` context managers)

    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param per_example_loss_fn: ELBo loss, i.e. negative Evidence Lower Bound,
        to minimize, per example.
    :param optim: an instance of :class:`~numpyro.optim._NumpyroOptim`.
    :param clipping_threshold: The clipping threshold C to which the norm
        of each per-example gradient is clipped.
    :param dp_scale: Scale parameter for the Gaussian mechanism applied to
        each dimension of the batch gradients.
    :param rng: PRNG key used in sampling the Gaussian mechanism applied to
        batch gradients.
    :param static_kwargs: static arguments for the model / guide, i.e. arguments
        that remain constant during fitting.
    """

    def __init__(self, model, guide, optim, per_example_loss,
            clipping_threshold, dp_scale, rng, **static_kwargs):


        gradients_clipping_fn = get_gradients_clipping_function(
            clipping_threshold
        )

        _, perturbation_rng = jax.random.split(rng, 2)

        # todo(lumip): this might not be correct. think about how splitting
        #   over the gradient influences the noise level. currently, dp_scale
        #   noise is applied to each parameter over all gradient components
        @jax.jit
        def grad_perturbation_fn(list_of_grads):
            def perturb_one(grad):
                noise = dist.Normal(0, dp_scale).sample(
                    perturbation_rng, sample_shape=grad.shape
                )
                return grad + noise

            #list_of_grads = jax.vmap(perturb_one, in_axes=0)(list_of_grads)

            # todo(lumip): somehow parallelizing/vmapping this would be great
            #   but current vmap will instead vmap over each position in it
            list_of_grads = tuple(perturb_one(grad) for grad in list_of_grads)
            return list_of_grads

        super().__init__(
            model, guide, optim, per_example_loss,
            gradients_clipping_fn, grad_perturbation_fn, **static_kwargs
        )
