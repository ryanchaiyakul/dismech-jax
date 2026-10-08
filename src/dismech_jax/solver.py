from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

from .acceptance import Accept, AcceptAll
from .directions import Direction, Newton, SaddleFree
from .predictors import Predictor, Previous

if TYPE_CHECKING:
    from .system import Aux, System


def _stop_gradient(tree):
    dynamic, static = eqx.partition(tree, eqx.is_inexact_array)
    return eqx.combine(jax.lax.stop_gradient(dynamic), static)


def _newton(
    sys: System,
    z: jax.Array,
    x0: jax.Array,
    aux: Aux,
    iters: int,
    ls_steps: int,
    c1: float,
    tol: float,
    direction: Direction,
    accept: Accept,
) -> tuple[jax.Array, jax.Array]:
    alphas = 0.5 ** jnp.arange(ls_steps)

    def not_converged(carry):
        _, _, res, k, moved = carry
        return (k < iters) & (jnp.linalg.norm(res) > tol) & moved

    def newton_step(carry):
        x, e_old, res, k, _ = carry

        # `res` is -grad(E), so `slope > 0` iff `delta_x` is a descent direction.
        delta_x = direction(sys, x, z, aux, res)
        slope = jnp.dot(res, delta_x)

        # Parallel line search
        test_xs = x + alphas[:, None] * delta_x
        test_es = jax.vmap(lambda _x: sys.energy(_x, z, aux))(test_xs)
        ok = jax.vmap(lambda _x: accept(sys, _x, z, aux))(test_xs)

        # If Armijo fails, take the smallest possible step (if accepted)
        is_good = (test_es <= e_old - c1 * alphas * slope) & ok  # Armijo Condition
        safe_idx = jnp.where(jnp.any(is_good), jnp.argmax(is_good), ls_steps - 1)
        moved = ok[safe_idx]  # False: no accepted step, stop

        next_x = jnp.where(moved, test_xs[safe_idx], x)
        next_e = jnp.where(moved, test_es[safe_idx], e_old)
        next_res = -sys.residual(next_x, z, aux)

        return next_x, next_e, next_res, k + 1, moved

    init_e = sys.energy(x0, z, aux)
    init_res = -sys.residual(x0, z, aux)
    final_x, _, final_res, _, _ = jax.lax.while_loop(
        not_converged, newton_step, (x0, init_e, init_res, 0, jnp.array(True))
    )
    return final_x, jnp.linalg.norm(final_res)


def _solve_one(
    sys: System,
    z: jax.Array,
    x0: jax.Array,
    aux: Aux,
    iters: int,
    ls_steps: int,
    c1: float,
    tol: float,
    direction: Direction,
    accept: Accept,
    passes: int,
) -> tuple[jax.Array, jax.Array, Aux]:
    # Newton iterations are never differentiated through
    sys_sg, z_sg, aux_sg, direction, accept = _stop_gradient(
        (sys, z, aux, direction, accept)
    )

    def newton(x, a):
        return _newton(sys_sg, z_sg, x, a, iters, ls_steps, c1, tol, direction, accept)

    x_star, res_norm = newton(jax.lax.stop_gradient(x0), aux_sg)
    if passes > 1:
        # Newton stopped short (blocked by `accept` or out of iterations):
        # refresh the aux at the current state and continue.
        def again(carry):
            x, _, a, p = carry
            a = sys_sg.update(a, x, z_sg)
            x, r = newton(x, a)
            return x, r, a, p + 1

        x_star, res_norm, aux_used, n = jax.lax.while_loop(
            lambda c: (c[3] < passes) & (c[1] > tol),
            again,
            (x_star, res_norm, aux_sg, 1),
        )
        # The refreshed frames are constants for the gradient (exact if no
        # refresh happened).
        aux = jax.tree.map(lambda r, a: jnp.where(n > 1, r, a), aux_used, aux)

    sys_sg, x_sg, z_sg, aux_sg = _stop_gradient((sys, x_star, z, aux))
    solve_H = sys_sg.linear_solver(x_sg, z_sg, aux_sg)

    # Implicit function theorem: dx/dp = -H_xx^-1 d(residual)/dp, where `p` is
    # anything in `sys`, `z` or `aux`. Always the true H_xx, whatever `direction`.
    x = jax.lax.custom_root(
        lambda x: sys.residual(x, z, aux),
        x_star,
        lambda _f, x: x,  # already solved
        lambda _g, y: solve_H(y),
    )
    return x, res_norm, aux


@eqx.filter_jit
def solve(
    sys: System,
    zs: jax.Array,
    aux: Aux,
    x0: jax.Array | None = None,
    iters: int = 10,
    ls_steps: int = 10,
    c1: float = 1e-4,
    tol: float = 1e-10,
    direction: Direction | None = None,
    predictor: Predictor | None = None,
    accept: Accept | None = None,
    passes: int = 1,
    z0: jax.Array | None = None,
    return_aux: bool = False,
) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, Aux]:
    """Solve `sys.residual(x, z, aux) = 0` for each fixed state `z` in `zs`.

    The solves run in order. Each starts from `predictor` applied to the
    previous solution and the aux is updated in between (path dependence). Use
    a single row for one equilibrium and several rows to load step.

    Gradients (forward and reverse) are exact w.r.t. every array in `sys`,
    `zs` and `aux`, including the path dependence, via the implicit function
    theorem. `x0` and the predictions are only initial guesses and receive no
    gradient.

    Args:
        sys (System): system.
        zs (jax.Array): fixed DOFs `(N, # of fixed DOFs)`.
        aux (Aux): initial aux state, aligned with `sys.terms`.
        x0 (jax.Array | None, optional): free DOFs before the first step.
            Defaults to `sys.x0`.
        iters (int, optional): Maximum number of newton-raphson iterations.
            Defaults to 10.
        ls_steps (int, optional): Number of alphas evaluated. Defaults to 10.
        c1 (float, optional): Armijo coefficient. Defaults to 1e-4.
        tol (float, optional): Newton stops early once the residual norm
            `|dE/dx|` is at most `tol` (absolute, in force units). Use `0.0` to
            always run `iters` iterations. Defaults to 1e-10.
        direction (Direction | None, optional): Newton step (see
            `directions`). Defaults to `SaddleFree()` for dense systems, which
            converges to stable equilibria also past bifurcations, and
            `Newton()` if `sys.block_size` is set.
        predictor (Predictor | None, optional): Initial guess of each step
            from the previous one (see `predictors`). Defaults to `Previous()`.
        accept (Accept | None, optional): Trial states the line search may take
            (see `acceptance`). Defaults to `AcceptAll()`.
        passes (int, optional): Newton runs per step. While unconverged, the
            aux is refreshed at the current state before the next run, e.g. to
            follow a snap-through past an `accept` limit. The refreshed aux is
            a constant for the gradient. Defaults to 1.
        z0 (jax.Array | None, optional): fixed DOFs that `x0` belongs to, the
            `z_old` of the first prediction. Defaults to `sys.z0`.
        return_aux (bool, optional): Also return the aux after each step
            (updated at its solution, i.e. what the next step starts from),
            stacked along a leading axis. Defaults to False.

    Returns:
        tuple[jax.Array, jax.Array]: Free DOFs `xs` `(N, # of free DOFs)` and the
            residual norm `(N,)` at each solve (not differentiable). Check it
            against a tolerance, since Newton does not raise if `iters` runs
            out first. Use `sys.join(xs, zs)` for the full states. With
            `return_aux`, `(xs, res_norms, auxs)`.
    """
    x0 = sys.x0 if x0 is None else x0
    z0 = sys.z0 if z0 is None else z0
    if direction is None:
        direction = Newton() if sys.block_size is not None else SaddleFree()
    predictor = Previous() if predictor is None else predictor
    accept = AcceptAll() if accept is None else accept

    def step(carry, z):
        x, aux, z_old = carry
        x = jax.lax.stop_gradient(predictor(*_stop_gradient((sys, x, z_old, z, aux))))
        x, res_norm, aux = _solve_one(
            sys, z, x, aux, iters, ls_steps, c1, tol, direction, accept, passes
        )
        aux = sys.update(aux, x, z)
        return (x, aux, z), (x, res_norm, aux if return_aux else None)

    _, (xs, res_norms, auxs) = jax.lax.scan(step, (x0, aux, z0), zs)
    return (xs, res_norms, auxs) if return_aux else (xs, res_norms)
