from __future__ import annotations

from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

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
) -> tuple[jax.Array, jax.Array]:
    alphas = 0.5 ** jnp.arange(ls_steps)

    def not_converged(carry):
        _, _, res, k = carry
        return (k < iters) & (jnp.linalg.norm(res) > tol)

    def newton_step(carry):
        x, e_old, res, k = carry

        delta_x = sys.linear_solver(x, z, aux)(res)

        # `res` is -grad(E), so `slope > 0` iff `delta_x` is a descent direction.
        # If H is indefinite it may not be: use steepest descent.
        slope = jnp.dot(res, delta_x)
        is_descent = slope > 0.0
        delta_x = jnp.where(is_descent, delta_x, res)
        slope = jnp.where(is_descent, slope, jnp.dot(res, res))

        # Parallel line search
        test_xs = x + alphas[:, None] * delta_x
        test_es = jax.vmap(lambda _x: sys.energy(_x, z, aux))(test_xs)

        # If Armijo fails, take the smallest possible step
        is_good = test_es <= e_old - c1 * alphas * slope  # Armijo Condition
        safe_idx = jnp.where(jnp.any(is_good), jnp.argmax(is_good), ls_steps - 1)

        next_x = test_xs[safe_idx]
        next_e = test_es[safe_idx]
        next_res = -sys.residual(next_x, z, aux)

        return next_x, next_e, next_res, k + 1

    init_e = sys.energy(x0, z, aux)
    init_res = -sys.residual(x0, z, aux)
    final_x, _, final_res, _ = jax.lax.while_loop(
        not_converged, newton_step, (x0, init_e, init_res, 0)
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
) -> tuple[jax.Array, jax.Array]:
    # Newton iterations are never differentiated through
    x_star, res_norm = _newton(
        *_stop_gradient((sys, z, x0, aux)), iters, ls_steps, c1, tol
    )
    sys_sg, x_sg, z_sg, aux_sg = _stop_gradient((sys, x_star, z, aux))
    solve_H = sys_sg.linear_solver(x_sg, z_sg, aux_sg)

    # Implicit function theorem: dx/dp = -H_xx^-1 d(residual)/dp, where `p` is
    # anything in `sys`, `z` or `aux`.
    x = jax.lax.custom_root(
        lambda x: sys.residual(x, z, aux),
        x_star,
        lambda _f, x: x,  # already solved
        lambda _g, y: solve_H(y),
    )
    return x, res_norm


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
) -> tuple[jax.Array, jax.Array]:
    """Solve `sys.residual(x, z, aux) = 0` for each fixed state `z` in `zs`.

    The solves run in order. Each starts from the previous solution and the aux
    is updated in between (path dependence). Use a single row for one
    equilibrium and several rows to load step.

    Gradients (forward and reverse) are exact w.r.t. every array in `sys`,
    `zs` and `aux`, including the path dependence, via the implicit function
    theorem. `x0` is only an initial guess and receives no gradient.

    Args:
        sys (System): system.
        zs (jax.Array): fixed DOFs `(N, # of fixed DOFs)`.
        aux (Aux): initial aux state, aligned with `sys.terms`.
        x0 (jax.Array | None, optional): initial guess for the free DOFs.
            Defaults to `sys.x0`.
        iters (int, optional): Maximum number of newton-raphson iterations.
            Defaults to 10.
        ls_steps (int, optional): Number of alphas evaluated. Defaults to 10.
        c1 (float, optional): Armijo coefficient. Defaults to 1e-4.
        tol (float, optional): Newton stops early once the residual norm
            `|dE/dx|` is at most `tol` (absolute, in force units). Use `0.0` to
            always run `iters` iterations. Defaults to 1e-10.

    Returns:
        tuple[jax.Array, jax.Array]: Free DOFs `xs` `(N, # of free DOFs)` and the
            residual norm `(N,)` at each solve (not differentiable). Check it
            against a tolerance, since Newton does not raise if `iters` runs
            out first. Use `sys.join(xs, zs)` for the full states.
    """
    x0 = sys.x0 if x0 is None else x0

    def step(carry, z):
        x, aux = carry
        x, res_norm = _solve_one(sys, z, x, aux, iters, ls_steps, c1, tol)
        return (x, sys.update(aux, x, z)), (x, res_norm)

    _, (xs, res_norms) = jax.lax.scan(step, (x0, aux), zs)
    return xs, res_norms
