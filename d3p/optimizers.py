# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2019- d3p Developers and their Assignees

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from numpyro.optim import _NumPyroOptim, _add_doc
import jax.numpy as jnp
from jax import lax
from jax.tree_util import tree_map, tree_leaves

try:
    from jax.example_libraries.optimizers import make_schedule
except (ImportError, AttributeError):
    # pre jax v0.2.25
    from jax.experimental.optimizers import make_schedule

import numpyro.optim

def adadp(
        step_size=1e-3,
        tol=1.0,
        stability_check=True,
        alpha_min=0.9,
        alpha_max=1.1
    ):  # noqa: E121, E125
    """ Constructs optimizer triple for the adaptive learning rate optimizer of
    Koskela and Honkela.

    Reference:
    A. Koskela, A. Honkela: Learning Rate Adaptation for Federated and
    Differentially Private Learning (https://arxiv.org/abs/1809.03832).

    :param step_size: The initial step size.
    :param tol: Error tolerance for the discretized gradient steps.
    :param stability_check: Settings to True rejects some updates in favor of a more
        stable algorithm.
    :param alpha_min: Lower multiplitcative bound of learning rate update per step.
    :param alpha_max: Upper multiplitcative bound of learning rate update per step.

    :return: An (init_fun, update_fun, get_params) triple.
    """
    step_size = make_schedule(step_size)

    def init(x0):
        lr = step_size(0)
        x_stepped = tree_map(lambda n: jnp.zeros_like(n), x0)
        return x0, lr, x_stepped, x0

    def _compute_update_step(x, g, step_size_):
        return tree_map(lambda x_, g_: x_ - step_size_ * g_, x, g)

    def _update_even_step(args):
        g, state, new_x = args
        x, lr, x_stepped, x_prev = state

        x_prev = x
        x_stepped = _compute_update_step(x, g, lr)

        return new_x, lr, x_stepped, x_prev

    def _update_odd_step(args):
        g, state, new_x = args
        x, lr, x_stepped, x_prev = state

        norm_partials = tree_map(
            lambda x_full, x_halfs: jnp.sum(((x_full - x_halfs)/jnp.maximum(1., x_full)) ** 2),
            x_stepped, new_x
        )

        err_e = jnp.array(tree_leaves(norm_partials))
        # note(lumip): paper specifies the approximate error function as
        #   using absolute values, but since we square anyways, those are
        #   not required here; the resulting array is partial squared sums
        #   of the l2-norm over all gradient elements (per gradient site)

        err_e = jnp.sqrt(jnp.sum(err_e))  # summing partial gradient norm

        new_lr = lr * jnp.minimum(
            jnp.maximum(jnp.sqrt(tol/err_e), 0.9), 1.1
        )

        new_x = tree_map(
            lambda x_prev, new_x: jnp.where(
                stability_check and err_e > tol, x_prev, new_x
            ),
            x_prev, new_x
        )

        return new_x, new_lr, x_stepped, x_prev

    def update(i, g, state):
        x, lr, x_stepped, x_prev = state

        new_x = _compute_update_step(x, g, 0.5 * lr)
        return lax.cond(
            i % 2 == 0,
            (g, state, new_x),
            _update_even_step,
            (g, state, new_x),
            _update_odd_step
        )

    def get_params(state):
        x = state[0]
        return x
    return init, update, get_params


@_add_doc(adadp)
class ADADP(_NumPyroOptim):

    def __init__(self,
                 step_size=1e-3,
                 tol=1.0,
                 stability_check=True,
                 alpha_min=0.9,
                 alpha_max=1.1) -> None:

        super(ADADP, self).__init__(
            adadp, step_size, tol, stability_check, alpha_min, alpha_max
        )
