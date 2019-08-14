import jax
import jax.numpy as np

from numpyro.handlers import scale

__all__ = ["map_over_secondary_dims"]

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
    def map_over_secondary_dims_f(T):
        assert(len(T.shape) >= 1)
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
    """Returns true if the input has the shape property (indicating that it is
    some array type).

    :param a: Anything that might have the shape property.
    """
    try:
        a.shape
        return True
    except:
        return False


def is_scalar(x):
    return not has_shape(x) or x.shape == ()


def is_integer(x):
    return (has_shape(x) and np.issubdtype(x.dtype, np.integer)) or np.issubdtype(type(x), np.integer)


def is_int_scalar(x):
    """Returns true if the input can be interepreted as a scalar integer value.
    
    :param x: Anything that might be an integer scalar.
    """
    return is_scalar(x) and is_integer(x)