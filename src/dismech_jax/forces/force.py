from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax

from ..states import State


class Force[AuxT: State | None](eqx.Module):
    """A generalized force term `F(q, t)` and its stiffness `H = -dF/dq`.

    A system is in equilibrium when the sum of its terms' `F` vanishes on the
    free DOFs. Every array field is a parameter the solution can be
    differentiated w.r.t.
    """

    @abstractmethod
    def F(self, q: jax.Array, t: jax.Array, aux: AuxT) -> jax.Array:
        """Returns the generalized force on the global state `q` at `t`."""

    @abstractmethod
    def H(self, q: jax.Array, t: jax.Array, aux: AuxT) -> jax.Array:
        """Returns the dense stiffness `-dF/dq`."""

    def update(self, aux: AuxT, q: jax.Array) -> AuxT:
        """Returns the aux updated at the converged state `q`. Default is a no-op."""
        return aux


class Energy[AuxT: State | None](Force[AuxT]):
    """A conservative force `F = -dE/dq`. Only `E` is required."""

    @abstractmethod
    def E(self, q: jax.Array, t: jax.Array, aux: AuxT) -> jax.Array:
        """Returns the scalar potential energy of `q` at `t`."""

    def F(self, q: jax.Array, t: jax.Array, aux: AuxT) -> jax.Array:
        return -jax.grad(self.E)(q, t, aux)

    def H(self, q: jax.Array, t: jax.Array, aux: AuxT) -> jax.Array:
        return jax.hessian(self.E)(q, t, aux)
