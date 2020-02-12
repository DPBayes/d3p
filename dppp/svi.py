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
from numpyro.primitives import sample, param
from numpyro.handlers import seed, trace, substitute

from dppp.util import map_over_secondary_dims, example_count


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

    def _compute_per_example_gradients(self, svi_state, *args, **kwargs):
        """ Computes the raw per-example gradients of the model.

        This is the first step in a full update iteration.

        :param svi_state: The current state of the SVI algorithm.
        :param args: Arguments to the loss function.
        :param kwargs: All keyword arguments to model or guide.
        :returns: tuple consisting of the updated svi state, an array of loss
            values per example, and a jax tuple tree of per-example gradients
            per parameter site (each site's gradients have shape (batch_size, *parameter_shape))
        """
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
        return SVIState(svi_state.optim_state, rng_key), per_example_loss, per_example_grads

    def _apply_per_example_gradient_transformations(self, svi_state, px_gradients):
        """ Applies per-example gradient transformations by applying
            `per_example_grad_manipulation_fn` (e.g., clipping) to each per-example
            gradient.

        This is the second step in a full update iteration.

        :param svi_state: The current state of the SVI algorithm.
        :param px_gradients: Jax tuple tree of per-example gradients as returned
            by `_compute_per_example_gradients`
        :returns: tuple consisting of the updated svi state, a list of
            transformed per-example gradients per site and the jax tree structure
            definition. The list is a flattened representation of the jax tree,
            the shape of per-example gradients per parameter is unaffected.
        """
        # px_gradients is a jax tree of jax np.arrays of shape
        #   [batch_size, (param_shape)] for each parameter. flatten it out!
        px_grads_list, px_grads_tree_def = jax.tree_flatten(
            px_gradients
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

        return svi_state, px_grads_list, px_grads_tree_def

    def _combine_and_transform_gradient(self, svi_state, px_grads_list, px_loss, px_grads_tree_def):
        """ Combines the per-example gradients into the batch gradient and
            applies the batch gradient transformation given as
            `batch_grad_manipulation_fn`.

        This is the third step of a full update iteration.

        :param svi_state: The current state of the SVI algorithm.
        :param px_grads_list: List of transformed per-example gradients as returned
            by `_apply_per_example_gradient_transformations`
        :param px_loss: Array of per-example loss values as output by
            `_compute_per_example_gradients`.
        :param px_grads_tree_def: Jax tree definition for the gradient tree as
            returned by `_apply_per_example_gradient_transformations`.
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
        loss_jacobian = np.reshape(loss_combine_vjp(np.array(1.))[0], (1, -1))
        # loss_vjp = lambda px_grads: np.sum(np.multiply(loss_jacobian, px_grads))
        loss_vjp = lambda px_grads: np.matmul(loss_jacobian, px_grads)

        # we map the loss combination vjp func over all secondary dimensions
        #   of gradient sites. This is necessary since some gradient
        #   sites might be matrices in itself (e.g., for NN layers), so a stack
        #   of those would be 3-dimensional and not admittable to np.matmul
        loss_vjp = map_over_secondary_dims(loss_vjp)

        # combine gradients for all parameters in the gradient jax tree
        #   according to the loss combination vjp func
        grads_list = tuple(map(loss_vjp, px_grads_list))

        batch_size = example_count(px_loss)

        # apply batch gradient modification (e.g., DP noise perturbation) (if any)
        if self.batch_grad_manipulation_fn:
            rng_key, rng_key_step = random.split(svi_state.rng_key, 2)
            svi_state = SVIState(svi_state.optim_state, rng_key)
            grads_list = self.batch_grad_manipulation_fn(
                grads_list, batch_size, rng=rng_key_step
            )

        # reassemble the jax tree used by optimizer for the final gradients
        grads = jax.tree_unflatten(
            px_grads_tree_def, grads_list
        )

        return svi_state, loss_val, grads
        

    def _apply_gradient(self, svi_state, batch_gradient):
        """ Takes a (batch) gradient step in parameter space using the specified
            optimizer.

        This is the fourth and last step of a full update iteration.
        :param svi_state: The current state of the SVI algorithm.
        :param batch_gradient: Jax tree of batch gradients per parameter site,
            as returned by `_combine_and_transform_gradient`.
        :returns: tuple consisting of the updated svi state.
        """
        optim_state = self.optim.update(batch_gradient, svi_state.optim_state)
        return SVIState(optim_state, svi_state.rng_key)

    def update(self, svi_state, *args, **kwargs):
        svi_state, per_example_loss, per_example_grads = \
            self._compute_per_example_gradients(svi_state, *args, **kwargs)

        svi_state, per_example_grads, tree_def = \
            self._apply_per_example_gradient_transformations(
                svi_state, per_example_grads
            )

        svi_state, loss, gradient = self._combine_and_transform_gradient(
            svi_state, per_example_grads, per_example_loss, tree_def
        )
        
        return self._apply_gradient(svi_state, gradient), loss


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
            clipping_threshold, dp_scale, **static_kwargs):


        gradients_clipping_fn = get_gradients_clipping_function(
            clipping_threshold
        )

        @jax.jit
        def grad_perturbation_fn(list_of_grads, batch_size, rng):
            def perturb_one(grad, site_rng):
                noise = dist.Normal(0, dp_scale / batch_size).sample(
                    site_rng, sample_shape=grad.shape
                )
                return grad + noise

            per_site_rngs = jax.random.split(rng, len(list_of_grads))
            # todo(lumip): somehow parallelizing/vmapping this would be great
            #   but current vmap will instead vmap over each position in it
            list_of_grads = tuple(
                perturb_one(grad, site_rng) 
                for grad, site_rng in zip(list_of_grads, per_site_rngs)
            )
            return list_of_grads

        super().__init__(
            model, guide, optim, per_example_loss,
            gradients_clipping_fn, grad_perturbation_fn, **static_kwargs
        )

def get_samples_from_trace(trace):
    """ Extracts all sample values from a numpyro trace.

    :param trace: trace object obtained from `numpyro.handlers.trace().get_trace()`
    :return: dictionary of sampled values associated with the names given
        to the sample since via `sample()` in the model
    """
    return {k: v['value'] for k, v in trace.items() if v['type'] == 'sample'}

def sample_prior_predictive(rng_key, model, model_args, substitutes=None):
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
    :return: Values for all sample sites, identified by `sample` calls in the
        model function.
    """
    if substitutes is None: substitutes = dict()
    model = seed(substitute(model, param_map=substitutes), rng_key)
    t = trace(model).get_trace(*model_args)
    return get_samples_from_trace(t)

def sample_posterior_predictive(rng_key, model, model_args, guide, guide_args, params):
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
    :return: Values for all sample sites, identified by `sample` calls in the
        model function.
    """
    model_rng_key, guide_rng_key = jax.random.split(rng_key)

    guide = seed(substitute(guide, param_map=params), guide_rng_key)
    guide_samples = get_samples_from_trace(trace(guide).get_trace(*guide_args))

    model = seed(substitute(model, param_map=guide_samples), model_rng_key)
    model_samples = get_samples_from_trace(trace(model).get_trace(*model_args))

    guide_samples.update(model_samples)
    return guide_samples

def _sample_a_lot(rng_key, n, single_sample_fn):
    rng_keys = jax.random.split(rng_key, n)
    return jax.vmap(single_sample_fn)(rng_keys)

def sample_multi_prior_predictive(rng_key, n, model, model_args, substitutes=None):
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
    :return: Values for all sample sites, identified by `sample` calls in the
        model function.
    """
    single_sample_fn = lambda rng: sample_prior_predictive(
        rng, model, model_args, substitutes=substitutes
    )
    return _sample_a_lot(rng_key, n, single_sample_fn)

def sample_multi_posterior_predictive(rng_key, n, model, model_args, guide, guide_args, params):
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
    :return: Values for all sample sites, identified by `sample` calls in the
        model function.
    """
    single_sample_fn = lambda rng: sample_posterior_predictive(
        rng, model, model_args, guide, guide_args, params
    )
    return _sample_a_lot(rng_key, n, single_sample_fn)

def fix_observations(model, observations):
    """ Fixes observations in a model function for likelihood evaluation.

    :param model: Function representing the model using numpyro distributions
        and the `sample` primitive
    :param: Dictionary of observations associated with the sample sites in the
        model function as designated by calls to `param`
    :return: Model function with fixed observations
    """
    return substitute(model, param_map=observations)
