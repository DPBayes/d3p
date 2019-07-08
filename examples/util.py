
__all__ = ['sigmoid', 'softmax']

import jax.numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    y = np.max(x)
    x_ = x-y
    return np.exp(x_)/np.sum(np.exp(x_))
