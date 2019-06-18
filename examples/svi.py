"""from numpyro.

original: https://github.com/pyro-ppl/numpyro/blob/master/numpyro/svi.py
"""

import os

from jax import random, value_and_grad, vjp
import jax.api

import jax.numpy as np

from numpyro.handlers import replay, seed, substitute, trace
from numpyro.hmc_util import log_density


def _seed(model, guide, rng):
    model_seed, guide_seed = random.split(rng, 2)
    model_init = seed(model, model_seed)
    guide_init = seed(guide, guide_seed)
    return model_init, guide_init


def svi(model, guide, per_sample_loss_fn, combined_loss_fn, optim_init, optim_update, get_params, **kwargs):
    """
    Stochastic Variational Inference given an ELBo loss objective.

    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param per_sample_loss_fn: ELBo loss, i.e. negative Evidence Lower Bound, to minimize, per sample.
    :param combined_loss_fn: Function to combine the per-sample loss values. For ELBo this is np.sum.
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
        return combined_loss_fn(per_sample_loss_fn(*args, **kwargs))

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

        # note(lumip): using a vmapped jax.linearize(fun, *primals) could help speeding up forward differentiation
        # mode and (somewhat) negate the performance cost to incurred by per-sample differentiation,
        # if I understand the implication of its documentation correctly.
        onp = jax.api.onp
        def per_sample_value_and_grad(fun, argnums=0, has_aux=False, holomorphic=False):

            def value_and_grad_f(*args, **kwargs):
                f = jax.linear_util.wrap_init(fun, kwargs)
                f_partial, dyn_args = jax.api._argnums_partial(f, argnums, args)
                if not has_aux:
                    ans, vjp_py = vjp(f_partial, *dyn_args)
                else:
                    ans, vjp_py, aux = vjp(f_partial, *dyn_args, has_aux=True)
                
                dtype = np.result_type(ans)
                if not (holomorphic or np.issubdtype(dtype, onp.floating)):
                    msg = ("Gradient only defined for real-output functions (with dtype that "
                            "is a subdtype of np.floating), but got dtype {}. For holomorphic "
                            "differentiation, pass holomorphic=True.")
                    raise TypeError(msg.format(dtype))

                assert(len(ans.shape) == 1)
                batch_size = ans.shape[0]
                one_hot_vecs = np.eye(batch_size, dtype=dtype)
                
                grads = jax.vmap(lambda v: vjp_py(v)[0] if isinstance(argnums, int) else vjp_py(v))(one_hot_vecs)

                if not has_aux:
                    return ans, grads
                else:
                    return (ans, aux), grads

            return value_and_grad_f

        per_sample_loss, per_sample_grads = per_sample_value_and_grad(per_sample_loss_fn)(
            params, model_init, guide_init, model_args, guide_args, kwargs
        )
        # per_sample_grads will be jax tree of jax np.arrays of shape [batch_size, (param_shape)] for each parameter
        
        # get total loss and loss combination jvp (forward differentiation) function
        loss_val, loss_jvp = jax.linearize(combined_loss_fn, per_sample_loss)

        # combine gradients for all parameters according to the loss combination jvp func
        def combine_gradients(grads):
            return loss_jvp(grads)
            # note(lumip): this currently fails due to a bug(?) in jax which tries to assert that the shape
            #       of the gradients is equal to that of the primals here (only for jit'ed functions).
            #       reported to the issue tracker as https://github.com/google/jax/issues/871
        grads = jax.tree_util.tree_map(combine_gradients, per_sample_grads)

        opt_state = optim_update(i, grads, opt_state)
        rng, = random.split(rng, 1)
        return loss_val, opt_state, rng

    def evaluate(opt_state, rng, model_args=(), guide_args=()):
        """
        Take a single step of SVI (possibly on a batch / minibatch of data).

        :param opt_state: current optimizer state.
        :param jax.random.PRNGKey rng: random number generator seed.
        :param tuple model_args: arguments to the model (these can possibly vary during
            the course of fitting).
        :param tuple guide_args: arguments to the guide (these can possibly vary during
            the course of fitting).
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


def elbo(param_map, model, guide, model_args, guide_args, kwargs):
    """
    This is the most basic implementation of the Evidence Lower Bound, which is the
    fundamental objective in Variational Inference. This implementation has various
    limitations (for example it only supports random variablbes with reparameterized
    samplers) but can be used as a template to build more sophisticated loss
    objectives.

    For more details, refer to http://pyro.ai/examples/svi_part_i.html.

    :param dict param_map: dictionary of current parameter values keyed by site
        name.
    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param tuple model_args: arguments to the model (these can possibly vary during
        the course of fitting).
    :param tuple guide_args: arguments to the guide (these can possibly vary during
        the course of fitting).
    :param dict kwargs: static keyword arguments to the model / guide.
    :return: negative of the Evidence Lower Bound (ELBo) to be minimized.
    """
    guide_log_density, guide_trace = log_density(guide, guide_args, kwargs, param_map)
    model_log_density, _ = log_density(replay(model, guide_trace), model_args, kwargs, param_map)
    # log p(z) - log q(z)
    elbo = model_log_density - guide_log_density
    # Return (-elbo) since by convention we do gradient descent on a loss and
    # the ELBO is a lower bound that needs to be maximized.
    return -elbo


def per_sample_log_density(model, model_args, model_kwargs, params):
    # note(lumip): I assumed here that all sites with samples will two-dimensional shape
    # of size (batch_size, site_dim), where batch_size is common for all sites.
    # is that an assumption that always holds?

    model = substitute(model, params)
    model_trace = trace(model).get_trace(*model_args, **model_kwargs)
    # note(lumip): explicit loop below faster? indicated by direct comparison, could be noise though. test!
    per_sample_log_joint = np.sum(
        [np.sum(site['fn'].log_prob(site['value']), axis=1)
            for site in model_trace.values()
            if site['type'] == 'sample'
        ],
        axis=0
    )
    # per_sample_log_joint = None
    # todo(lumip): is there a way to get the batch size and init per_sample_log_joint already?
    # for site in model_trace.values():
    #     if site['type'] == 'sample':
    #         # site['value'] holds the traced sampled values and has shape (batch x sample_dim)
    #         if per_sample_log_joint is None:
    #             # todo(lumip): is it guaranteed that len(shape) == 2 always?
    #             per_sample_log_joint = np.sum(site['fn'].log_prob(site['value']), axis=1) 
    #         else:
    #             assert(per_sample_log_joint.shape[0] == site['value'].shape[0])
    #             per_sample_log_joint += np.sum(site['fn'].log_prob(site['value']), axis=1)

    return per_sample_log_joint, model_trace


def per_sample_elbo(param_map, model, guide, model_args, guide_args, kwargs):
    """
    This is the per-sample version of the most basic implementation of the Evidence 
    Lower Bound, which is the fundamental objective in Variational Inference. 
    This implementation has various  limitations (for example it only supports random
    variables with reparameterized samplers) but can be used as a template to build
    more sophisticated loss objectives.

    For more details, refer to http://pyro.ai/examples/svi_part_i.html.

    :param dict param_map: dictionary of current parameter values keyed by site
        name.
    :param model: Python callable with Pyro primitives for the model.
    :param guide: Python callable with Pyro primitives for the guide
        (recognition network).
    :param tuple model_args: arguments to the model (these can possibly vary during
        the course of fitting).
    :param tuple guide_args: arguments to the guide (these can possibly vary during
        the course of fitting).
    :param dict kwargs: static keyword arguments to the model / guide.
    :return: negative of the Evidence Lower Bound (ELBo) per sample to be minimized.
    """
    guide_log_density, guide_trace = per_sample_log_density(guide, guide_args, kwargs, param_map)
    model_log_density, _ = per_sample_log_density(replay(model, guide_trace), model_args, kwargs, param_map)
    # log p(z) - log q(z)
    elbo = model_log_density - guide_log_density
    # Return (-elbo) since by convention we do gradient descent on a loss and
    # the ELBO is a lower bound that needs to be maximized.
    return -elbo
