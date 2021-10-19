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

from typing import Tuple, Callable, Optional
from fourier_accountant.compute_eps import get_epsilon_S, get_epsilon_R
import numpy as np

__all__ = ['approximate_sigma', 'approximate_sigma_remove_relation']
ComputeEpsFn = Callable[[float, Optional[float]], float]


def get_bracketing_bounds(  # noqa: C901
        compute_eps_fn: ComputeEpsFn,
        target_eps: float,
        maxeval: int,
        initial_sigma: Optional[float] = 1.
    ) -> Tuple[np.ndarray, np.ndarray, int]:  # noqa:E121,E125
    """ Determines rough upper and lower bounds for sigma around a target privacy
    epsilon value.

    :param compute_eps_fn: Privacy accountant function returning epsilon for
        a given sigma and precision scale; assumed to be monotonic decreasing in sigma.
    :param target_eps: Desired target epsilon.
    :param maxeval: Maximum number of evaluations of `compute_eps_fn`.
    :param initial_sigma: Initial guess for sigma.
    :return: Tuple (bounds, bound_eps, num_evals) where
        1) bounds is a tuple containing a lower and upper bound to the
            sigma resulting in `target_eps`.
        2) bound_eps are the corresponding epsilon values;
            it holds bound_eps[0] > target_eps > bound_eps[1]
        3) num_evals is the number of function evaluations performed.
    """
    assert(initial_sigma > 0.)
    assert(target_eps > 0)
    assert(maxeval > 0 and isinstance(maxeval, int))

    sig = initial_sigma
    num_evals = 0

    while num_evals < maxeval:
        try:
            num_evals += 1
            eps = compute_eps_fn(sig, precision=1.)

            # we want to make sure we got a reliable value.
            # we double the precision of the evaluation function and are
            # satisfied if the value is within .1
            num_evals += 1
            new_eps = compute_eps_fn(sig, precision=2.)
            if (abs(1 - eps/new_eps) <= .1):  # noqa: 226
                break
            else:
                sig *= 10
        except ValueError:
            sig *= 10

    if num_evals >= maxeval:
        raise RuntimeError("Could not establish bounds in given evaluation limit")

    sig_1 = sig
    eps_1 = eps
    if eps >= target_eps:
        while eps >= target_eps:
            sig *= 4
            while num_evals < maxeval:
                try:
                    num_evals += 1
                    eps = compute_eps_fn(sig)
                    break
                except ValueError:
                    sig = np.mean([sig, sig_1])
                    sig *= 0.9
                    if sig <= sig_1:
                        raise RuntimeError("Could not establish bounds in given evaluation limit")
                finally:
                    if num_evals >= maxeval:
                        raise RuntimeError("Could not establish bounds in given evaluation limit")

        return np.array([sig_1, sig]), np.array([eps_1, eps]), num_evals
    else:
        while eps < target_eps:
            sig /= 4
            while num_evals < maxeval:
                try:
                    num_evals += 1
                    eps = compute_eps_fn(sig)
                    break
                except ValueError:
                    sig *= 1.2
                    if sig >= sig_1:
                        raise RuntimeError("Could not establish bounds in given evaluation limit")
                finally:
                    if num_evals >= maxeval:
                        raise RuntimeError("Could not establish bounds in given evaluation limit")

        return np.array([sig, sig_1]), np.array([eps, eps_1]), num_evals


def update_bounds(
        sig: float,
        eps: float,
        target_eps: float,
        bounds: np.ndarray,
        bound_eps: np.ndarray,
        consecutive_updates: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:  # noqa:E121,E125
    """ Updates bounds for sigma around a target privacy epsilon.

    Updates the lower bound for sigma if `eps` is larger than `target_eps` and
    the upper bound otherwise.

    :param sig: A new value for sigma.
    :param eps: The corresponding value for epsilon.
    :param target_eps: The target value for epsilon.
    :param bounds: Tuple containing a lower and upper bound for the sigma
        corresponding to target_eps.
    :param bound_eps: The corresponding epsilon values for the bounds.
    :param consecutive_updates: Tuple counting the number of consecutive updates
        for lower and upper bound.
    :return: updated bounds, bound_eps and consecutive_updates
    """
    assert(eps <= bound_eps[0])
    assert(eps >= bound_eps[1])

    if eps > target_eps:
        bounds[0] = sig
        bound_eps[0] = eps
        consecutive_updates = [consecutive_updates[0] + 1, 0]
    else:
        bounds[1] = sig
        bound_eps[1] = eps
        consecutive_updates = [0, consecutive_updates[1] + 1]

    return bounds, bound_eps, consecutive_updates


def _approximate_sigma(
        compute_eps_fn: ComputeEpsFn,
        target_eps: float,
        q: float,
        tol: Optional[float] = 1e-4,
        force_smaller: Optional[bool] = False,
        maxeval: Optional[int] = 10
    ) -> Tuple[float, float, int]:  # noqa:E121,E125
    """ Approximates the sigma corresponding to a target privacy epsilon.

    Uses a bracketing approach where an initial rough estimate of lower and upper
    bounds for sigma is iteratively shrunk. Each iteration fits a logarithmic
    function eps,precision->sigma to the bounds which is evaluated at target_eps
    to get the next estimate for sigma, which is in turn used to update the bounds.

    :param compute_eps_fn: Privacy accountant function returning epsilon for
        a given sigma and precision scale; assumed to be monotonic decreasing in sigma.
    :param target_eps: Desired target epsilon.
    :param tol: Absolute tolerance for the epsilon corresponding to the returned
        value for sigma, i.e., `abs(compute_eps_fn(sigma_opt) - target_eps) < tol`
    :param force_smaller: Require that the returned value for sigma results
        in an epsilon that is strictly smaller than `target_eps`. This may
        slightly violate `tol`.
    :param maxeval: Maximum number of evaluations of `compute_eps_fn`. The
        function aborts the search for sigma after `maxeval` function evaluations
        were made and returns the current best estimate. In that case, `tol`
        is violated but `force_smaller` is still adhered to.
    :return: Tuple consisting of
        1) the determined value for sigma
        2) the corresponding espilon
        3) the number of function evaluations made
    """

    initial_sigma = 1. / (0.01 / q)  # heuristic: for q=0.01 sigma=1 is in stable range, so try to scale accordingly
    bounds, bound_eps, num_evals = get_bracketing_bounds(
        compute_eps_fn, target_eps, maxeval, initial_sigma=initial_sigma
    )
    eps = bound_eps[1]
    consecutive_updates = [0, 0]

    while abs(target_eps - eps) > tol and num_evals < maxeval:
        assert(bound_eps[0] >= target_eps)  # loop invariants
        assert(bound_eps[1] <= target_eps)  # these are the assumptions for the procedure to work

        # fitting function eps -> sigma (sig = a-b*log(eps), shape determined empirically for Fourier Accountant)
        b = (bounds[1] - bounds[0]) / (np.log(bound_eps[0]) - np.log(bound_eps[1]))
        a = np.mean(bounds + b * np.log(bound_eps))

        # evaluate fitted function at target_eps to get new estimate for sigma
        # new_sig = a - b * np.log(target_eps*scale)
        new_sig = a - b * np.log(target_eps)
        assert(new_sig >= bounds[0] and new_sig <= bounds[1])
        eps = compute_eps_fn(new_sig)
        num_evals += 1

        bounds, bound_eps, consecutive_updates = update_bounds(
            new_sig, eps, target_eps, bounds, bound_eps, consecutive_updates
        )

        # To guarantee that both bounds get regular updates, we track the number
        # of consecutive updates for a bound and forcibly update the other if
        # that number exceeds a certain value.
        MAX_CONSECUTIVE_UPDATES = 2
        if num_evals < maxeval and (
                consecutive_updates[0] > MAX_CONSECUTIVE_UPDATES or
                consecutive_updates[1] > MAX_CONSECUTIVE_UPDATES):

            # In this case, the optimal sigma is very close to the often
            # updated bound and thus evaluating at the midpoint of the interval
            # will update the previously neglected bound.
            new_sig = np.mean(bounds)
            eps = compute_eps_fn(new_sig)
            num_evals += 1

            bounds, bound_eps, consecutive_updates = update_bounds(
                new_sig, eps, target_eps, bounds, bound_eps, consecutive_updates
            )

    if force_smaller and eps > target_eps:
        idx = bound_eps < target_eps
        new_sig = bounds[idx][0]
        eps = bound_eps[idx][0]

    assert(not force_smaller or eps < target_eps)

    return new_sig, eps, num_evals


def approximate_sigma(
        target_eps: float,
        delta: float,
        q: float,
        num_iter: int,
        tol: Optional[float] = 1e-4,
        force_smaller: Optional[bool] = False,
        maxeval: Optional[int] = 10
    ) -> Tuple[float, float, int]:  # noqa:E121,E125
    """ Approximates the sigma corresponding to a target privacy epsilon using
    the Fourier Accountant for the substitute relation.

    Uses a bracketing approach where an initial rough estimate of lower and upper
    bounds for sigma is iteratively shrunk. Each iteration fits a logarithmic
    function eps,precision->sigma to the bounds which is evaluated at target_eps
    to get the next estimate for sigma, which is in turn used to update the bounds.

    :param compute_eps_fn: Privacy accountant function returning epsilon for
        a given sigma and precision scale; assumed to be monotonic decreasing in sigma.
    :param target_eps: The desired target epsilon.
    :param delta: The delta privacy parameter.
    :param q: The subsampling ratio.
    :param num_iter: The number of batch iterations.
    :param tol: Absolute tolerance for the epsilon corresponding to the returned
        value for sigma, i.e., `abs(compute_eps_fn(sigma_opt) - target_eps) < tol`
    :param force_smaller: Require that the returned value for sigma results
        in an epsilon that is strictly smaller than `target_eps`. This may
        slightly violate `tol`.
    :param maxeval: Maximum number of evaluations of `compute_eps_fn`. The
        function aborts the search for sigma after `maxeval` function evaluations
        were made and returns the current best estimate. In that case, `tol`
        is violated but `force_smaller` is still adhered to.
    :return: Tuple consisting of
        1) the determined value for sigma
        2) the corresponding espilon
        3) the number of function evaluations made
    """
    L = max(20, target_eps * 2)

    def compute_eps(sigma, precision=1):
        return get_epsilon_S(
            delta, sigma, q, ncomp=num_iter, L=L * precision,
            nx=1e6 * (L * precision) / 20
        )

    return _approximate_sigma(compute_eps, target_eps, q, tol, force_smaller, maxeval)


def approximate_sigma_remove_relation(
        target_eps: float,
        delta: float,
        q: float,
        num_iter: int,
        tol: Optional[float] = 1e-4,
        force_smaller: Optional[bool] = False,
        maxeval: Optional[int] = 10
    ) -> Tuple[float, float, int]:  # noqa:E121,E125
    """ Approximates the sigma corresponding to a target privacy epsilon using
    the Fourier Accountant for the add/remove relation.

    Uses a bracketing approach where an initial rough estimate of lower and upper
    bounds for sigma is iteratively shrunk. Each iteration fits a logarithmic
    function eps->sigma to the bounds, which is evaluated at target_eps to get
    the next estimate for sigma, which is in turn used to update the bounds.

    :param compute_eps_fn: Privacy accountant function returning epsilon for
        a given sigma, assumed to be monotonic decreasing.
    :param target_eps: The desired target epsilon.
    :param delta: The delta privacy parameter.
    :param q: The subsampling ratio.
    :param num_iter: The number of batch iterations.
    :param tol: Absolute tolerance for the epsilon corresponding to the returned
        value for sigma, i.e., `abs(compute_eps_fn(sigma_opt) - target_eps) < tol`
    :param force_smaller: Require that the returned value for sigma results
        in an epsilon that is strictly smaller than `target_eps`. This may
        slightly violate `tol`.
    :param maxeval: Maximum number of evaluations of `compute_eps_fn`. The
        function aborts the search for sigma after `maxeval` function evaluations
        were made and returns the current best estimate. In that case, `tol`
        is violated but `force_smaller` is still adhered to.
    :return: Tuple consisting of
        1) the determined value for sigma
        2) the corresponding espilon
        3) the number of function evaluations made
    """
    L = max(20, target_eps * 2)

    def compute_eps(sigma, precision=1.):
        return get_epsilon_R(
            delta, sigma, q, ncomp=num_iter, L=L * precision,
            nx=1e6 * (L * precision) / 20
        )

    return _approximate_sigma(compute_eps, target_eps, q, tol, force_smaller, maxeval)
