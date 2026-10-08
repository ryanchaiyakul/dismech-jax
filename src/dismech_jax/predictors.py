"""Load-step predictors: `predictor(sys, x, z_old, z_new, aux) -> x_guess`.

`x` is the equilibrium at the fixed DOFs `z_old` (with `aux`). The guess is the
initial Newton iterate at `z_new`; it never affects the converged state's
gradients, but near a bifurcation it decides which branch Newton falls into."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import equinox as eqx
import jax

if TYPE_CHECKING:
    from .system import Aux, System


class Predictor(eqx.Module):
    @abstractmethod
    def __call__(
        self, sys: System, x: jax.Array, z_old: jax.Array, z_new: jax.Array, aux: Aux
    ) -> jax.Array: ...


class Previous(Predictor):
    """The previous equilibrium."""

    def __call__(self, sys, x, z_old, z_new, aux):
        return x


class Tangent(Predictor):
    """First-order continuation: `x - H_xx^-1 H_xz (z_new - z_old)`, the tangent
    of the equilibrium path (implicit function theorem). `H_xz dz` is a JVP of
    the residual, so this also works for block tridiagonal systems. Unreliable
    where `H_xx` is (near) singular, e.g. exactly at a bifurcation."""

    reg: float = 1e-8

    def __call__(self, sys, x, z_old, z_new, aux):
        _, hxz_dz = jax.jvp(
            lambda z: sys.residual(x, z, aux), (z_old,), (z_new - z_old,)
        )
        return x - sys.linear_solver(x, z_old, aux, self.reg)(hxz_dz)


class Linear(Predictor):
    """`x + W (z_new - z_old)` for a fixed `(# free DOFs, # fixed DOFs)` map `W`,
    e.g. spreading a moving clamp's increment over the free span."""

    W: jax.Array

    def __call__(self, sys, x, z_old, z_new, aux):
        return x + self.W @ (z_new - z_old)
