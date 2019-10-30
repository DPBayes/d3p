
from dppp.util import is_int_scalar, is_array, example_count
from numpyro.handlers import scale
import jax.numpy as np

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
        if not np.isscalar(batch_or_batchsize):
            raise ValueError("if a scalar is given for batch_or_batchsize, it "
                "can't be traced through jit. consider using static_argnums "
                "for the jit invocation.")
        batch_size = batch_or_batchsize
    elif is_array(batch_or_batchsize):
        batch_size = example_count(batch_or_batchsize)
    else:
        raise ValueError("batch_or_batchsize must be an array or an integer")
    if num_obs_total is None:
        num_obs_total = batch_size
    return scale(scale_factor = num_obs_total / batch_size)
