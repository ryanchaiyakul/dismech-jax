"""Newton step directions: `direction(sys, x, z, aux, res) -> delta_x`.

`res = -dE/dx` at `(x, z)`. The solver line searches along `delta_x`, so it
should be a descent direction (`res . delta_x > 0`)."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from .system import Aux, System


class Direction(eqx.Module):
    @abstractmethod
    def __call__(
        self, sys: System, x: jax.Array, z: jax.Array, aux: Aux, res: jax.Array
    ) -> jax.Array: ...


class Newton(Direction):
    """`H_xx^-1 res`, or steepest descent `res` where that does not descend
    (`H_xx` indefinite). Dense or block tridiagonal (`sys.linear_solver`)."""

    reg: float = 1e-8

    def __call__(self, sys, x, z, aux, res):
        delta_x = sys.linear_solver(x, z, aux, self.reg)(res)
        return jnp.where(jnp.dot(res, delta_x) > 0.0, delta_x, res)


class SaddleFree(Direction):
    """`|H_xx|^-1 res`, dense only.

    `H_xx` is Jacobi scaled, `D^-1/2 H_xx D^-1/2` with `D = |diag(H_xx)|`, so
    stiff (stretch) and soft (twist) DOFs are comparable, then every eigenvalue
    is replaced by `max(|lambda|, eps)`. This is a descent direction even where
    `H_xx` is indefinite, so Newton moves away from saddles and converges to
    stable equilibria, e.g. past a buckling bifurcation. Where `H_xx` is
    positive definite it is the Newton step."""

    eps: float = 1e-12

    def __call__(self, sys, x, z, aux, res):
        if sys.block_size is not None:
            raise ValueError("`SaddleFree` needs a dense system (`block_size=None`).")
        H, _ = sys.hessian(x, z, aux)
        d = jnp.abs(jnp.diag(H))
        s = 1.0 / jnp.sqrt(jnp.maximum(d, 1e-12 * jnp.max(d)))
        lam, V = jnp.linalg.eigh(s[:, None] * H * s[None, :])
        return s * (V @ ((V.T @ (s * res)) / jnp.maximum(jnp.abs(lam), self.eps)))


class Clipped(Direction):
    """`inner`, scaled down so no free DOF changes by more than `max_step` per
    iteration: a trust region against wild steps far from equilibrium.

    `max_step` is a scalar or one value per free DOF, in DOF units (e.g. a
    length for positions and `inf` to leave twist angles unlimited)."""

    inner: Direction
    max_step: float | jax.Array

    def __call__(self, sys, x, z, aux, res):
        delta_x = self.inner(sys, x, z, aux, res)
        ratio = jnp.max(jnp.abs(delta_x) / self.max_step)
        return delta_x / jnp.maximum(1.0, ratio)
