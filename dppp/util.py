import jax
import jax.numpy as np

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
