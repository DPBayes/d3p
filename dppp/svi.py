""" Stochastic Variational Inference implementation with per-example gradient
    manipulation capability.

Based on numpyro's `svi`:
    https://github.com/pyro-ppl/numpyro/blob/master/numpyro/svi.py
"""

import os

import functools
import warnings

import jax
from jax import random, vjp
import jax.numpy as np

from numpyro.handlers import replay, substitute, trace, scale
from numpyro.svi import _seed
from numpyro.distributions import constraints
from numpyro.distributions.constraints import biject_to

from dppp.util import map_over_secondary_dims, example_count, is_int_scalar, has_shape

def minibatch(batch_or_batchsize, num_obs_total=None):
    """Returns a context within which all samples are treated as being a
    minibatch of a larger data set.

    In essence, this marks the (log)likelihood of the sampled examples to be
    scaled to the total loss value over the whole data set.

    :param batch_or_batchsize: An integer indicating the batch size or an array
        indicating the shape of the batch where the length of the first axis
        is interpreted as batch size.
    :param num_obs_total: The total number of examples/observations in the
        full data set. Optional, defaults to the given batch size.
    """
    if is_int_scalar(batch_or_batchsize):
        batch_size = batch_or_batchsize
    elif has_shape(batch_or_batchsize):
        batch_size = example_count(batch_or_batchsize)
    else:
        raise ValueError("batch_or_batchsize must be an array or an integer")
    if num_obs_total is None:
        num_obs_total = batch_size
    return scale(scale_factor = num_obs_total / batch_size)

def per_example_value_and_grad(fun, argnums=0, has_aux=False, holomorphic=False):
    value_and_grad_fun = jax.jit(jax.value_and_grad(fun, argnums, has_aux, holomorphic))
    return jax.jit(jax.vmap(value_and_grad_fun, in_axes=(None, 0, 0)))

def svi(model, guide, per_example_loss_fn, optim_init, optim_update, get_params,
    per_example_grad_manipulation_fn=None, loss_combiner_fn=np.sum, **kwargs):
    """
    Stochastic Variational Inference given a per-example loss objective and a
    loss combiner function.

    This is identical to numpyro's `svi` but explicitely computes gradients
    per example (i.e. observed data instance) based on `per_example_loss_fn`
    before combining them to a total loss value using `loss_combiner_fn`.
    This will allow manipulating the per-example gradients which has
    applications, e.g., in differentially private machine learning applications.

    To obtain the per-example gradients, the `per_example_loss_fn` is evaluated
    for (and the gradient take wrt) each example in a vectorized manner (using
    `jax.vmap`).
    
    For this to work, the following requirements are imposed upon
    `per_example_loss_fn`:
    - in per-example evaluation, the leading dimension of the batch (indicating
        the number of examples) is stripped away. the loss function must be able
        to handle this (if it was originally designed to handle batched values)
    - since it will be evaluated on each example, take special care that the
        loss function scales the likelihood contribution of the data properly
        wrt to batch size and total example count (use e.g. the `numpyro.scale`
        or the convenience `minibatch` context managers)

    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param per_example_loss_fn: ELBo loss, i.e. negative Evidence Lower Bound,
        to minimize, per example.
    :param optim_init: initialization function returned by a JAX optimizer.
        see: :mod:`jax.experimental.optimizers`.
    :param optim_update: update function for the optimizer
    :param get_params: function to get current parameters values given the
        optimizer state.
    :param per_example_grad_manipulation_fn: optional function that allows to
        manipulate the gradient for each sample.
    :param loss_combiner_fn: Function to combine the per-example loss values.
        Defaults to np.sum.
    :param `**kwargs`: static arguments for the model / guide, i.e. arguments
        that remain constant during fitting.
    :return: tuple of `(init_fn, update_fn, evaluate)`.
    """

    constrain_fn = None

    def loss_fn(*args, **kwargs):
        return loss_combiner_fn(per_example_loss_fn(*args, **kwargs))

    def init_fn(rng, model_args=(), guide_args=(), params=None):
        """

        :param jax.random.PRNGKey rng: random number generator seed.
        :param tuple model_args: arguments to the model (these can possibly vary during
            the course of fitting).
        :param tuple guide_args: arguments to the guide (these can possibly vary during
            the course of fitting).
        :param dict params: initial parameter values to condition on. This can be
            useful for initializing neural networks using more specialized methods
            rather than sampling from the prior.
        :return: tuple containing initial optimizer state, and `constrain_fn`, a callable
            that transforms unconstrained parameter values from the optimizer to the
            specified constrained domain
        """
        # note(lumip): the below is unchanged from numpyro's `svi` but seems
        #   like a very inefficient/complicated way to obtain the parameters,
        #   especially since it means manual work by the user to obtain a
        #   throw-away batch just for initialization...
        # todo(lumip): is there a way to improve?
        assert isinstance(model_args, tuple)
        assert isinstance(guide_args, tuple)
        model_init, guide_init = _seed(model, guide, rng)
        if params is None:
            params = {}
        else:
            model_init = substitute(model_init, params)
            guide_init = substitute(guide_init, params)
        guide_trace = trace(guide_init).get_trace(*guide_args, **kwargs)
        model_trace = trace(model_init).get_trace(*model_args, **kwargs)
        inv_transforms = {}
        for site in list(guide_trace.values()) + list(model_trace.values()):
            if site['type'] == 'param':
                constraint = site['kwargs'].pop('constraint', constraints.real)
                transform = biject_to(constraint)
                inv_transforms[site['name']] = transform
                params[site['name']] = transform.inv(site['value'])

        def transform_constrained(inv_transforms, params):
            return {k: inv_transforms[k](v) for k, v in params.items()}

        nonlocal constrain_fn
        constrain_fn = jax.partial(transform_constrained, inv_transforms)
        return optim_init(params), constrain_fn

    def update_fn(i, rng, opt_state, model_args=(), guide_args=()):
        """
        Take a single step of SVI (possibly on a batch / minibatch of data),
        using the optimizer.

        :param int i: represents the i'th iteration over the epoch, passed as an
            argument to the optimizer's update function.
        :param opt_state: current optimizer state.
        :param jax.random.PRNGKey rng: random number generator seed.
        :param tuple model_args: dynamic arguments to the model.
        :param tuple guide_args: dynamic arguments to the guide.
        :return: tuple of `(loss_val, opt_state, rng)`.
        """
        rng, model_rng = random.split(rng, 2)
        model_init, guide_init = _seed(model, guide, model_rng)

        params = get_params(opt_state)
        @jax.jit
        def wrapped_fn(p, model_args, guide_args):
            return per_example_loss_fn(
                p, model_init, guide_init, model_args, guide_args, kwargs,
                constrain_fn=constrain_fn
            )

        per_example_loss, per_example_grads = per_example_value_and_grad(
            wrapped_fn
        )(
            params, model_args, guide_args
        )

        # get total loss and loss combiner vjp func
        loss_val, loss_combine_vjp = jax.vjp(loss_combiner_fn, per_example_loss)
        # loss_combine_vjp gives us the backward differentiation function
        #   from combined loss to per-example losses. we use it to get the
        #   (1xbatch_size) Jacobian and construct a function that takes
        #   per-example gradients and left-multiplies them with that jacobian
        #   to get the final combined gradient
        loss_jacobian = np.reshape(loss_combine_vjp(np.array(1))[0], (1, -1))
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
        if per_example_grad_manipulation_fn:

            # apply per-sample gradient manipulation, if present
            px_grads_list = jax.vmap(
                per_example_grad_manipulation_fn, in_axes=0
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
        grads = tuple(map(loss_vjp, px_grads_list))

        # reassemble the jax tree used by optimizer for the final gradients
        grads = jax.tree_unflatten(
            px_grads_tree_def, grads
        )

        # take a step in the optimizer using the gradients
        opt_state = optim_update(i, grads, opt_state)
        return loss_val, opt_state, rng

    def evaluate(rng, opt_state, model_args=(), guide_args=()):
        """
        Take a single step of SVI (possibly on a batch / minibatch of data).

        :param opt_state: current optimizer state.
        :param jax.random.PRNGKey rng: random number generator seed.
        :param tuple model_args: arguments to the model (these can possibly vary
            during the course of fitting).
        :param tuple guide_args: arguments to the guide (these can possibly vary
            during the course of fitting).
        :return: evaluate ELBo loss given the current parameter values
            (held within `opt_state`).
        """
        model_init, guide_init = _seed(model, guide, rng)
        params = get_params(opt_state)
        return loss_fn(params, model_init, guide_init, 
                       model_args, guide_args, kwargs, 
                       constrain_fn=constrain_fn
        )

    # Make local functions visible from the global scope once
    # `svi` is called for sphinx doc generation.
    if 'SPHINX_BUILD' in os.environ:
        svi.init_fn = init_fn
        svi.update_fn = update_fn
        svi.evaluate = evaluate

    return init_fn, update_fn, evaluate


def full_norm(list_of_parts, ord=2):
    """Computes the total norm over a list of values (of any shape) by treating
    them as a single large vector.

    :param list_of_parts: The list of values that make up the vector to compute
        the norm over.
    :param ord: Order of the norm. May take any value possible for
    `numpy.linalg.norm`.
    :return: The indicated norm over the full vector.
    """
    ravelled = [g.ravel() for g in list_of_parts]
    gradients = np.concatenate(ravelled)
    assert(len(gradients.shape) == 1)
    norm = np.linalg.norm(gradients, ord=ord)
    return norm

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

def dpsvi(model, guide, per_example_loss_fn, optim_init, optim_update,
    get_params, clipping_threshold, **kwargs):
    """
    Differentially-Private Stochastic Variational Inference given a per-example
    loss objective and a gradient clipping threshold.

    This is identical to numpyro's `svi` but adds differential privacy by
    clipping the gradients (and currently nothing more).

    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param per_example_loss_fn: ELBo loss function, i.e. negative Evidence Lower
        Bound, to minimize, per example.
    :param optim_init: initialization function returned by a JAX optimizer.
        see: :mod:`jax.experimental.optimizers`.
    :param optim_update: update function for the optimizer
    :param get_params: function to get current parameters values given the
        optimizer state.
    :param clipping_threshold: The clipping threshold C to which the norm
        of each per-example gradient is clipped.
    :param `**kwargs`: static arguments for the model / guide, i.e. arguments
        that remain constant during fitting.
    :return: tuple of `(init_fn, update_fn, evaluate)`.
    """

    gradients_clipping_fn = get_gradients_clipping_function(
        clipping_threshold
    )

    return svi(model, guide, per_example_loss_fn, optim_init, optim_update,
        get_params, per_example_grad_manipulation_fn=gradients_clipping_fn,
        **kwargs
    )
