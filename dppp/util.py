import jax
import jax.numpy as np
from functools import reduce, wraps

__all__ = ["map_over_secondary_dims", "has_shape", "is_array", "is_scalar",
    "is_integer", "is_int_scalar", "example_count"]

def map_over_secondary_dims(f):
    """
    Maps a function taking a over all secondary axes of an array.

    f is assumed to take a vector of shape (a,) and output a scalar. 
    Returns a function f_mapped that for an input T with shape
    (a, b_1, ..., b_k) applies f on the entries along the first axis
    over all other axes, i.e., to all T[:,i_1, ..., i_k].
    
    The output of f_mapped will be an array of shape [b_1, ...., b_k] where
    each entry is the corresponding output of f applied as described above.

    :param f: a function from vector to scalar
    :return: function f_mapped applying f to each first axis vector in an
        arbitrary shaped input

    Example: 
    >>> T = [
    >>>      [ [ a_1, a_2 ], [ a_3, a_4 ] ],
    >>>      [ [ b_1, b_2 ], [ b_3, b_4 ] ],
    >>>      [ [ c_1, c_2 ], [ c_3, c_4 ] ]
    >>> ]
    >>> print(T.shape)
    (3, 2, 2)
    >>> R = map_over_secondary_dims(f)(T)
    >>> print(R)
    [
        [f(a_1, b_1, c_1), f(a_2, b_2, c_2)],
        [f(a_3, b_3, c_3), f(a_4, b_4, c_4)]
    ]
    """
    @wraps(f)
    def map_over_secondary_dims_f(T):
        assert(np.ndim(T) >= 1)
        T_ = T.reshape((T.shape[0], -1))
        Z_ = jax.vmap(f, in_axes=1)(T_)
        return Z_.reshape(T.shape[1:])
    return map_over_secondary_dims_f


def example_count(a):
    """Returns the amount of examples/observations in an array interpreted
    as multi-example data set.

    :param a: The data set from which to extract the example count.
    """
    try:
        return np.shape(a)[0]
    except:
        return 1


def has_shape(a):
    """Returns True if the input has the shape attribute, indicating that it is
    of a numpy array type.

    Note that this also applies to scalars in jax.jit decorated functions.
    """
    try:
        a.shape
        return True
    except AttributeError:
        return False

def is_array(a):
    """Returns True if the input has is determined to be an array, i.e.,
    has more than 0 dimensions and a shape attribute.

    Note that this does not apply to scalars in jax.jit decorated functions.

    :param a: Anything that might be an array.
    """
    return has_shape(a) and np.ndim(a) > 0


def is_scalar(x):
    """Returns True if the input can be interpreted as a scalar.

    This fits actual scalars as well as arrays that contain only one element
    (regardless of their number of dimensions). I.e., a (jax.)numpy array
    with shape (1,1,1,1) would be considered a scalar.

    Works with jax.jit.
    """
    # note(lumip): a call to jax.jit(is_scalar)(s), where x is a scalar,
    #   results in an x that is a jax.numpy array without any dimensions but
    #   which has a shape attribute. therefore, np.isscalar(x) as well as
    #   is_array(x) are False -> we have to use has_shape(x) to detect this
    return np.isscalar(x) or (has_shape(x) and reduce(lambda x, a: x*a, np.shape(x), 1) == 1)


def is_integer(x):
    """Returns True if the input value(s) (a scalar or (jax.)numpy array) have integer type.

    Works with jax.jit.

    :param x: Scalar or (jax.)numpy array that could have integer values.
    """
    return (has_shape(x) and np.issubdtype(x.dtype, np.integer)) or np.issubdtype(type(x), np.integer)


def is_int_scalar(x):
    """Returns True if the input can be interepreted as a scalar integer value.

    Works with jax.jit.
    
    :param x: Anything that might be an integer scalar.
    """
    return is_scalar(x) and is_integer(x)

def normalize(x):
    return x / np.linalg.norm(x)