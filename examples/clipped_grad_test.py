import jax
import jax.numpy as np

@jax.custom_transforms
def clip_grad_value(x, threshold):
    """Clips the (reverse) gradient of the input element-wise.

    Any gradient (g_1, ..., g_n) will be clipped as (max(g_1, threshold), ..., max(g_n, threshold)).

    Has no effect in the forwards pass, i.e., behaves as identity function.

    :param x:
    :param threshold: The threshold with which to clip gradient values.
    """
    return x

jax.defvjp_all(
    clip_grad_value,
    lambda x, c:
        (x,
         lambda g:
            (np.clip(g, a_max=c), 1.)
        )
)

@jax.custom_transforms
def clip_grad_norm(x, threshold):
    """Clips the norm of the (reverse) gradient of the input.

    Any gradient g will be clipped s.t. norm(g) <= threshold by multiplying each element with max(1, norm(g)/threshold).

    Has no effect in the forwards pass, i.e., behaves as identity function.

    :param x:
    :param threshold: The threshold with which to clip gradient values.
    """
    return x

jax.defvjp_all(
    clip_grad_norm,
    lambda x, c:
        (x,
         lambda g:
            (g/np.maximum(1., np.linalg.norm(g)/c), 1.)
        )
)

def clip_gradient_values(threshold):
    def clip_gradient_values_decorator(f):
        def clip_gradient_values_f(*args):
            return f(*(clip_grad_value(x, threshold) for x in args))
        return clip_gradient_values_f
    return clip_gradient_values_decorator

def clip_gradient_norms(threshold):
    def clip_gradient_norms_decorator(f):
        def clip_gradient_norms_f(*args):
            return f(*(clip_grad_norm(x, threshold) for x in args))
        return clip_gradient_norms_f
    return clip_gradient_norms_decorator

@clip_gradient_norms(3.0)
def test_dotp(x, y):
    return np.dot(x, y)

x, y = np.array([2, 0.7]), np.array([0.6, 5.4])
(value, gradients) = (jax.value_and_grad(test_dotp, argnums=(0,1))(x, y))
print("test_dotp(x, y): {}".format(value))
print("gradients:\n\t{}".format(gradients))
print("gradient norms:\n\t{}".format([np.linalg.norm(g) for g in gradients]))

print("\n########################\n")

x = np.array([
    [3., 4.],
    [6., 8.],
    [9., 0.1],
])
y = np.array([
    [5., 0.3],
    [1., -1.],
    [2.3, -1.75],
])

@clip_gradient_norms(5.0)
def test_dotp_sum(x, y):
    # return np.tensordot(x, y) # this is equivalent to: sum(row_wise_dot(x, y))
    return np.sum(np.diag(np.matmul(x, np.transpose(y)))) # this also

(value, gradients) = jax.value_and_grad(test_dotp_sum, argnums=(0,1))(x, y)
print("test_dotp_sum(x, y): {}".format(value))
print("gradients:\n\t{}".format(gradients))
print("gradient norms:\n\t{}".format([[np.linalg.norm(v) for v in g] for g in gradients]))
# todo(lumip): clipping doesn't work properly as of yet. figure out why
