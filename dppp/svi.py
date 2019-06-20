""" Stochastic Variational Inference implementation with per-sample gradient
    manipulation capability.

Based on numpyro's `svi`:
    https://github.com/pyro-ppl/numpyro/blob/master/numpyro/svi.py
"""

import os

import jax
from jax import random, vjp
import jax.numpy as np

from numpyro.handlers import replay, substitute, trace
from numpyro.svi import _seed

from dppp.util import map_over_secondary_dims


def per_sample_value_and_grad(fun, argnums=0, has_aux=False, holomorphic=False):
    """Creates a function which evaluates both `fun` and the gradient of `fun`
    per-sample.

    :param fun: Per-sample function to be differentiated. Its arguments at
        positions specified by `argnums` should be arrays, scalars, or standard
        Python containers. The first axis in the arguments indicates different
        samples. `fun` should be vectorized in the sense that it returns
        a vector where each element is the result of applying the function to
        a single input sample.
    argnums: Optional, integer or tuple of integers. Specifies which positional
      argument(s) to differentiate with respect to (default 0).
    has_aux: Optional, bool. Indicates whether `fun` returns a pair where the
     first element is considered the output of the mathematical function to be
     differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether `fun` is promised to be
      holomorphic. Default False.

  Returns:
    A function with the same arguments as `fun` that evaluates both `fun` and
    the per-sample gradients of `fun` and returns them as a pair
    (a two-element tuple). If `argnums` is an integer then the per-sample
    gradients are an array where each entry has the same shape and type as the
    positional argument indicated by that integer. If argnums is a tuple of
    integers, the pre-sample gradients result is a tuple of such arrays for each
    of the corresponding arguments.
  """

    def value_and_grad_f(*args, **kwargs):
        f = jax.linear_util.wrap_init(fun, kwargs)
        f_partial, dyn_args = jax.api._argnums_partial(f, argnums, args)

        # get the result of applying fun and the vjp (backward differentation)
        # function vjp_py
        if not has_aux:
            ans, vjp_py = vjp(f_partial, *dyn_args)
        else:
            ans, vjp_py, aux = vjp(f_partial, *dyn_args, has_aux=True)

        dtype = np.result_type(ans)
        if not (holomorphic or np.issubdtype(dtype, np.floating)):
            msg = ("Gradient only defined for real-output functions (with dtype"
                    "that is a subdtype of np.floating), but got dtype {}. For"
                    "holomorphic differentiation, pass holomorphic=True.")
            raise TypeError(msg.format(dtype))

        assert(len(ans.shape) == 1)

        # samples are aligned along the first axis
        batch_size = ans.shape[0]

        # filter the gradient contribution per sample using all possible one-hot
        # vectors as inputs to vjp_py
        one_hot_vecs = np.eye(batch_size, dtype=dtype)
        grads = jax.vmap(
            lambda v: vjp_py(v)[0] if isinstance(argnums, int) else vjp_py(v)
        )(one_hot_vecs)

        if not has_aux:
            return ans, grads
        else:
            return (ans, aux), grads

    return value_and_grad_f


def svi(model, guide, per_sample_loss_fn, loss_combiner_fn,
        optim_init, optim_update, get_params, **kwargs):
    """
    Stochastic Variational Inference given a per-sample loss objective and a
    loss combiner function.

    This is identical to numpyro's `svi` but explicitely computes gradients
    per sample based on `per_sample_loss_fn` before combining them to a total
    loss value using `loss_combiner_fn`. This will allow manipulating the
    per-sample gradients which has applications, e.g., in differentially private
    machine learning applications.

    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param per_sample_loss_fn: ELBo loss, i.e. negative Evidence Lower Bound,
        to minimize, per sample.
    :param loss_combiner_fn: Function to combine the per-sample loss values.
        For ELBo this is np.sum.
    :param optim_init: initialization function returned by a JAX optimizer.
        see: :mod:`jax.experimental.optimizers`.
    :param optim_update: update function for the optimizer
    :param get_params: function to get current parameters values given the
        optimizer state.
    :param `**kwargs`: static arguments for the model / guide, i.e. arguments
        that remain constant during fitting.
    :return: tuple of `(init_fn, update_fn, evaluate)`.
    """

    def loss_fn(*args, **kwargs):
        return loss_combiner_fn(per_sample_loss_fn(*args, **kwargs))

    def init_fn(rng, model_args=(), guide_args=(), params=None):
        """

        :param jax.random.PRNGKey rng: random number generator seed.
        :param tuple model_args: arguments to the model (these can possibly vary during
            the course of fitting).
        :param tuple guide_args: arguments to the guide (these can possibly vary during
            the course of fitting).
        :param dict params: initial parameter values to condition on. This can be
            useful forx
        :return: initial optimizer state.
        """
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
        for site in list(guide_trace.values()) + list(model_trace.values()):
            if site['type'] == 'param':
                params[site['name']] = site['value']
        return optim_init(params)

    def update_fn(i, opt_state, rng, model_args=(), guide_args=()):
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
        model_init, guide_init = _seed(model, guide, rng)
        params = get_params(opt_state)

        per_sample_loss, per_sample_grads = per_sample_value_and_grad(per_sample_loss_fn)(
            params, model_init, guide_init, model_args, guide_args, kwargs
        )
        # per_sample_grads will be jax tree of jax np.arrays of shape
        #   [batch_size, (param_shape)] for each parameter

        # todo(lumip): this is the place to perform per-sample gradient
        #   manipulation, e.g., clipping

        # get total loss and loss combiner jvp (forward differentiation) func
        loss_val, loss_jvp = jax.linearize(loss_combiner_fn, per_sample_loss)

        # mapping loss combination jvp func over all secondary dimensions
        #   of gradient collections/matrices
        loss_jvp = map_over_secondary_dims(loss_jvp)

        # combine gradients for all parameters in the gradient jax tree
        #   according to the loss combination jvp func
        grads = jax.tree_util.tree_map(loss_jvp, per_sample_grads)

        # take a step in the optimizer using the gradients
        opt_state = optim_update(i, grads, opt_state)
        rng, = random.split(rng, 1)
        return loss_val, opt_state, rng

    def evaluate(opt_state, rng, model_args=(), guide_args=()):
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
        return loss_fn(params, model_init, guide_init, model_args, guide_args, kwargs)

    # Make local functions visible from the global scope once
    # `svi` is called for sphinx doc generation.
    if 'SPHINX_BUILD' in os.environ:
        svi.init_fn = init_fn
        svi.update_fn = update_fn
        svi.evaluate = evaluate

    return init_fn, update_fn, evaluate


def per_sample_log_density(model, model_args, model_kwargs, params):
    """
    Evaluates the log_density of a model for each given sample.

    Similar to numpyro's `log_density`, the model is conditioned on `params` and
    then the logarithmic probability of the outcomes given in `model_args` and
    `model_kwargs` are computed.
    However, all sampled variables are understood to align different samples
    along the first axis (whether in model_args or params). Computing the
    per-sample log_density requires observations of all variables, including
    latent ones and the corresponding amount of samples (size of the first axis)
    must be identical for all.
    The result is a vector of corresponding length giving the log probability
    for each sample.

    :param model: The model for which to evaluate the 
    :param model_args: arguments for calling the model function
    :param model_kwargs: keyword arguments for calling the model function
    :param params: fixed parameters for the model (the corresponding model
        variables will be fixed to these, i.e., the model is conditioned on
        params)
    """
    # note(lumip): I assumed here that all sites with samples will have
    #   two-dimensional shape of size (batch_size, site_dim), where batch_size
    #   is common for all sites. is that an assumption that always holds?
    #   the way sampling works and possible jitting by jax makes this somewhat 
    #   hard to assert, so for now if that condition is not met, we will get
    #   an unspecified shape error internally.
    # todo(lumip): revisit this later and work out a nicer solution

    model = substitute(model, params)
    model_trace = trace(model).get_trace(*model_args, **model_kwargs)
    
    # note(lumip): the part below wasn't working because the second assertion
    #   was not evaulated directly when this functions context was jit'ed by jax
    #   and thus would not catch the shape errors it was intended to catch

    # # ensure all samples have the same shape and len(shape) == 2
    # sample_shapes = [site['value'].shape
    #                  for site in model_trace.values()
    #                  if site['type'] == 'sample']
    # r_shape = sample_shapes[0]

    # assert(len(r_shape) == 2)
    # assert(np.all([s_shape == r_shape for s_shape in sample_shapes]), 
    #     "samples observed for variables differ in shape. most there were"
    #     "different amounts of samples given for each variable")

    per_sample_log_joint = np.sum(
        [np.sum(site['fn'].log_prob(site['value']), axis=1)
            for site in model_trace.values()
            if site['type'] == 'sample'
        ],
        axis=0
    )

    return per_sample_log_joint, model_trace


def per_sample_elbo(param_map, model, guide, model_args, guide_args, kwargs):
    """
    Per-sample version of the most basic implementation of the Evidence
    Lower Bound, which is the fundamental objective in Variational Inference.

    Returns a vector of per-sample loss contributions instead of a total loss
    scalar. Otherwise identical to numpyro's `elbo`.

    :param dict param_map: dictionary of current parameter values keyed by site
        name.
    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param tuple model_args: arguments to the model (these can possibly vary
        during the course of fitting).
    :param tuple guide_args: arguments to the guide (these can possibly vary
        during the course of fitting).
    :param dict kwargs: static keyword arguments to the model / guide.
    :return: negative of the Evidence Lower Bound (ELBo) per sample to be
        minimized.
    """
    guide_log_density, guide_trace = per_sample_log_density(
        guide, guide_args, kwargs, param_map
    )
    model_log_density, _ = per_sample_log_density(
        replay(model, guide_trace), model_args, kwargs, param_map
    )
    elbo = model_log_density - guide_log_density
    # Return (-elbo) since by convention we do gradient descent on a loss and
    # the ELBO is a lower bound that needs to be maximized.
    return -elbo
