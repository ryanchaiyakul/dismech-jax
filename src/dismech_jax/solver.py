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
    t: jax.Array,
    q0: jax.Array,
    aux: Aux,
    iters: int,
    ls_steps: int,
    c1: float,
) -> jax.Array:
    alphas = 0.5 ** jnp.arange(ls_steps)

    def newton_step(carry, _):
        q, m_old, res = carry

        # TODO: use SOCU for blockdiagonal
        H = sys.jacobian(q, t, aux)
        H_reg = H.at[jnp.diag_indices(H.shape[0])].add(1e-8)
        delta_q = jnp.linalg.solve(H_reg, res)
        # Descent rate of the merit along the Newton direction
        slope = jnp.dot(res, delta_q) if sys.is_conservative else jnp.dot(res, res)

        # Parallel line search
        test_qs = q + alphas[:, None] * delta_q
        test_merits = jax.vmap(lambda _q: sys.merit(_q, t, aux))(test_qs)

        # If Armijo fails, take the smallest possible step
        is_good = test_merits <= m_old - c1 * alphas * slope  # Armijo Condition
        safe_idx = jnp.where(jnp.any(is_good), jnp.argmax(is_good), ls_steps - 1)

        next_q = test_qs[safe_idx]
        next_m = test_merits[safe_idx]
        next_res = -sys.residual(next_q, t, aux)

        return (next_q, next_m, next_res), jnp.linalg.norm(next_res)

    q_init = sys.bc.apply(q0, t)
    init_m = sys.merit(q_init, t, aux)
    init_res = -sys.residual(q_init, t, aux)
    (final_q, _, _), _ = jax.lax.scan(
        newton_step, (q_init, init_m, init_res), None, iters
    )
    return final_q


@eqx.filter_jit
def solve_step(
    sys: System,
    t: jax.Array,
    q0: jax.Array,
    aux: Aux,
    iters: int = 10,
    ls_steps: int = 10,
    c1: float = 1e-4,
) -> jax.Array:
    """Solve `sys.residual(q, t, aux) = 0` starting from `q0`.

    Differentiable (forward and reverse mode) w.r.t. every array in `sys`,
    `aux` and `t` via the implicit function theorem. `q0` is only an initial
    guess and receives no gradient.
    """
    # Newton iterations are never differentiated through
    q_star = _newton(*_stop_gradient((sys, t, q0, aux)), iters, ls_steps, c1)
    H = jax.lax.stop_gradient(sys.jacobian(q_star, t, aux))
    H_reg = H.at[jnp.diag_indices(H.shape[0])].add(1e-8)

    return jax.lax.custom_root(
        lambda q: sys.residual(q, t, aux),
        q_star,
        lambda _f, q: q,  # already solved
        lambda _g, y: jnp.linalg.solve(H_reg, y),
    )


@eqx.filter_jit
def solve(
    sys: System,
    ts: jax.Array,
    aux: Aux,
    q0: jax.Array | None = None,
    iters: int = 10,
    ls_steps: int = 10,
    c1: float = 1e-4,
    substeps: int = 1,
) -> jax.Array:
    """Solve for the quasi-static equilibrium at every `t` in `ts`.

    Gradients are exact w.r.t. every array in `sys` (e.g. rest strain, BC
    values, material models), `aux` and `ts`, including the path dependence
    through the aux updates between steps.

    Args:
        sys (System): system.
        ts (jax.Array): times `(N,)`.
        aux (Aux): initial aux state, aligned with `sys.terms`.
        q0 (jax.Array | None, optional): initial guess. Defaults to `sys.q0`.
        iters (int, optional): Number of newton-raphson iterations. Defaults to 10.
        ls_steps (int, optional): Number of alphas evaluated. Defaults to 10.
        c1 (float, optional): Armijo coefficient. Defaults to 1e-4.
        substeps (int, optional): Equal substeps between consecutive
            `ts`. Defaults to 1.

    Returns:
        jax.Array: Solved state `(N, # of DOFs)`.
    """
    q0 = sys.q0 if q0 is None else q0

    def step(carry, t):
        q, aux = carry
        q = solve_step(sys, t, q, aux, iters, ls_steps, c1)
        return (q, sys.update(aux, q)), None

    def outer(carry, sub_ts):
        carry, _ = jax.lax.scan(step, carry, sub_ts)
        return carry, carry[0]

    carry, _ = step((q0, aux), ts[0])

    # Equal substeps between consecutive `ts`, ending exactly on each `t`
    fracs = jnp.arange(1, substeps + 1, dtype=ts.dtype) / substeps
    sub_ts = ts[:-1, None] + (ts[1:] - ts[:-1])[:, None] * fracs
    _, qs = jax.lax.scan(outer, carry, sub_ts)
    return jnp.concatenate([carry[0][None], qs])
