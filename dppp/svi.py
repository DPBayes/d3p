""" Stochastic Variational Inference implementation with per-example gradient
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


def per_example_value_and_grad(fun, argnums=0, has_aux=False, holomorphic=False):
    """Creates a function which evaluates both `fun` and the gradient of `fun`
    per example (i.e., per observed data instance).

    :param fun: Per-example function to be differentiated. Its arguments at
        positions specified by `argnums` should be arrays, scalars, or standard
        Python containers. The first axis in the arguments indicates different
        examples. `fun` should be vectorized in the sense that it returns
        a vector where each element is the result of applying the function to
        a single input example.
    argnums: Optional, integer or tuple of integers. Specifies which positional
      argument(s) to differentiate with respect to (default 0).
    has_aux: Optional, bool. Indicates whether `fun` returns a pair where the
     first element is considered the output of the mathematical function to be
     differentiated and the second element is auxiliary data. Default False.
    holomorphic: Optional, bool. Indicates whether `fun` is promised to be
      holomorphic. Default False.

  Returns:
    A function with the same arguments as `fun` that evaluates both `fun` and
    the per-example gradients of `fun` and returns them as a pair
    (a two-element tuple). If `argnums` is an integer then the per-example
    gradients are an array where each entry has the same shape and type as the
    positional argument indicated by that integer. If argnums is a tuple of
    integers, the per-example gradients result is a tuple of such arrays for
    each of the corresponding arguments.
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

        # examples are aligned along the first axis
        batch_size = ans.shape[0]

        # filter the gradient contribution per example using all possible
        # one-hot vectors as inputs to vjp_py
        one_hot_vecs = np.eye(batch_size, dtype=dtype)
        grads = jax.vmap(
            lambda v: vjp_py(v)[0] if isinstance(argnums, int) else vjp_py(v)
        )(one_hot_vecs)

        if not has_aux:
            return ans, grads
        else:
            return (ans, aux), grads

    return value_and_grad_f


def svi(model, guide, per_example_loss_fn,
        optim_init, optim_update, get_params, per_example_variables=None,
        loss_combiner_fn = np.sum, **kwargs):
    """
    Stochastic Variational Inference given a per-example loss objective and a
    loss combiner function.

    This is identical to numpyro's `svi` but explicitely computes gradients
    per example (i.e. observed data instance) based on `per_example_loss_fn`
    before combining them to a total loss value using `loss_combiner_fn`.
    This will allow manipulating the per-example gradients which has
    applications, e.g., in differentially private machine learning applications.

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
    :param per_example_variables: Names of the variables that have per-example
        contribution to the (log) probabilities.
    :param loss_combiner_fn: Function to combine the per-example loss values.
        Defaults to np.sum.
    :param `**kwargs`: static arguments for the model / guide, i.e. arguments
        that remain constant during fitting.
    :return: tuple of `(init_fn, update_fn, evaluate)`.
    """

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
            useful forx
        :return: initial optimizer state.
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

        per_example_loss, per_example_grads = per_example_value_and_grad(
            per_example_loss_fn
        )(
            params, model_init, guide_init, model_args, guide_args, kwargs,
            per_example_variables=per_example_variables
        )
        # per_example_grads will be jax tree of jax np.arrays of shape
        #   [batch_size, (param_shape)] for each parameter

        # todo(lumip): this is the place to perform per-example gradient
        #   manipulation, e.g., clipping

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

        # combine gradients for all parameters in the gradient jax tree
        #   according to the loss combination vjp func
        grads = jax.tree_util.tree_map(loss_vjp, per_example_grads)

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
        return loss_fn(params, model_init, guide_init, 
                       model_args, guide_args, kwargs, per_example_variables)

    # Make local functions visible from the global scope once
    # `svi` is called for sphinx doc generation.
    if 'SPHINX_BUILD' in os.environ:
        svi.init_fn = init_fn
        svi.update_fn = update_fn
        svi.evaluate = evaluate

    return init_fn, update_fn, evaluate

def per_example_log_density(
    model, model_args, model_kwargs, params, per_example_variables=None):
    """
    Evaluates the log_density of a model for each given example.

    Similar to numpyro's `log_density`, the model is conditioned on `params` and
    then the logarithmic probability of the outcomes given in `model_args` and
    `model_kwargs` are computed. The result is a vector giving the log
    probability for each example (i.e., observed data instance).
    Summing over the output vector gives the same result as numpyro's
    `log_density` function.

    Random variables specified by name in `per_example_variables` are
    understood to align different examples along the first axis, each of which
    will contribute only to the loss term of the corresponding item in the
    output.

    All other random variables are interepreted as 'global' and their
    probability contribution will be divivded evenly over the output cells.

    If `per_example_variables` is empty or none of the random variables of the
    model is contained in it, the output will be a scalar giving the total loss
    and thus identical to that of numpyro's `log_density`.    

    :param model: The model for which to evaluate the 
    :param model_args: arguments for calling the model function
    :param model_kwargs: keyword arguments for calling the model function
    :param params: fixed parameters for the model (the corresponding model
        variables will be fixed to these, i.e., the model is conditioned on
        params)
    :param per_example_variables: Names of the variables that have per-example
        contribution to the (log) probabilities.
    """

    model = substitute(model, params)
    model_trace = trace(model).get_trace(*model_args, **model_kwargs)

    # determine num_examples from first encountered example variable
    num_examples = 1 # 1 is default in case no site is in per_example_variables
    if per_example_variables is not None:
        for site in model_trace.values():
            if site['name'] in per_example_variables:
                num_examples = np.atleast_1d(site['value']).shape[0]
                break

    # helper function to sum a random variable according to whether it has
    # per-example contribution or not
    def axis_aware_per_example_sum(x, name):
        if len(x.shape) > 2:
            raise TypeError("invalid shape in sampled data in "
                "per_example_log_density. too many axes: {}".format(x.shape))
        if len(x.shape) < 2:
            x = x.reshape(-1, 1)
        if per_example_variables is not None and name in per_example_variables:
            assert(x.shape[0] == num_examples)
            return np.sum(x, axis=1)
        else:
            return np.ones(num_examples) * (np.sum(x) / num_examples)

    per_site_sums = \
        [axis_aware_per_example_sum(
            site['fn'].log_prob(site['value']), site['name']
         )
         for site in model_trace.values()
         if site['type'] == 'sample']

    per_example_log_joint = np.sum(per_site_sums, axis=0)

    return per_example_log_joint, model_trace


def per_example_elbo(
    param_map, model, guide, model_args, guide_args, kwargs, 
    per_example_variables=None):
    """
    Per-example version of the most basic implementation of the Evidence
    Lower Bound, which is the fundamental objective in Variational Inference.

    Returns a vector of per-example loss contributions instead of a total loss
    scalar. Otherwise identical to numpyro's `elbo`. If the
    `per_example_variables` parameter is None or none of the random variables
    occuring in model or guide are contained in it, the output will be a scalar
    holding the total elbo loss and thus identical to the output of `elbo`.

    :param dict param_map: dictionary of current parameter values keyed by site
        name.
    :param per_example_variables: Names of the variables that have per-example
        contribution to the (log) probabilities.
    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param tuple model_args: arguments to the model (these can possibly vary
        during the course of fitting).
    :param tuple guide_args: arguments to the guide (these can possibly vary
        during the course of fitting).
    :param dict kwargs: static keyword arguments to the model / guide.
    :return: negative of the Evidence Lower Bound (ELBo) per example to be
        minimized.
    """

    guide_log_density, guide_trace = per_example_log_density(
        guide, guide_args, kwargs, param_map, per_example_variables
    )
    model_log_density, _ = per_example_log_density(
        replay(model, guide_trace), model_args, kwargs, param_map, 
        per_example_variables
    )

    # note(lumip): If the guide only contains global random variables
    #   that do not directly contribute to per-example probability (i.e. are not
    #   contained in per_example_variables),
    #   guide_log_density will be a scalar holding the total loss contribution
    #   of the guide instead of a vector holding per-example contributions.
    #   In this case we have to divide it by the number of examples to get the
    #   correct per-example elbo
    assert(len(model_log_density.shape)==1)
    assert(len(guide_log_density.shape)==1)
    if guide_log_density.shape[0] == 1:
        guide_log_density /= model_log_density.shape[0]

    elbo = model_log_density - guide_log_density
    # Return (-elbo) since by convention we do gradient descent on a loss and
    # the ELBO is a lower bound that needs to be maximized.
    return -elbo
