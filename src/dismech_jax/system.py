from __future__ import annotations

from typing import Any, Self

import diffrax
import equinox as eqx
import jax
import jax.numpy as jnp

from .bc import AbstractBC
from .forces import Energy, Force, StencilEnergy

type Aux = tuple[Any, ...]


class System(eqx.Module):
    """Equilibrium of the sum of `terms` under the boundary condition `bc`.

    `aux` is a tuple aligned with `terms` (`None` for terms without aux).
    Gradients of the solution are defined w.r.t. every array in the system
    (BC values, rest strains, material models, ...), `aux` and `t`.
    """

    q0: jax.Array
    bc: AbstractBC
    terms: tuple[Force, ...]
    mass: jax.Array | None = None

    def with_bc(self, bc: AbstractBC) -> Self:
        """Get a copy of the system with `self.bc` replaced with `bc`."""
        return eqx.tree_at(lambda s: s.bc, self, bc)

    def with_rest(self, q_rest: jax.Array, aux: Aux) -> Self:
        """Get a copy with every stencil term's rest strain measured at `q_rest`."""
        terms = tuple(
            f.with_rest(q_rest, a) if isinstance(f, StencilEnergy) else f
            for f, a in zip(self.terms, aux, strict=True)
        )
        return eqx.tree_at(lambda s: s.terms, self, terms)

    @property
    def is_conservative(self) -> bool:
        """True if every term is an `Energy`, so `merit` is the total energy."""
        return all(isinstance(f, Energy) for f in self.terms)

    def residual(self, q: jax.Array, t: jax.Array, aux: Aux) -> jax.Array:
        """Get the equilibrium residual of state `q` at `t`.

        Free rows are the masked net force `-sum(F)` and constrained rows are
        `q - bc(q)`, so the residual depends on the BC values (making them
        differentiable) and its Jacobian is non-singular.
        """
        q_bc = self.bc.apply(q, t)
        mask = self.bc.mask(q)
        F = sum(
            (f.F(q_bc, t, a) for f, a in zip(self.terms, aux, strict=True)),
            jnp.zeros_like(q),
        )
        return -mask * F + q - q_bc

    def jacobian(self, q: jax.Array, t: jax.Array, aux: Aux) -> jax.Array:
        """Get the Jacobian of `residual` w.r.t. `q` (masked stiffness, identity on BCs)."""
        q_bc = self.bc.apply(q, t)
        mask = self.bc.mask(q)
        H = sum(
            (f.H(q_bc, t, a) for f, a in zip(self.terms, aux, strict=True)),
            jnp.zeros((q.shape[0], q.shape[0]), dtype=q.dtype),
        )
        H = H * mask[:, None] * mask[None, :]
        diag_idx = jnp.arange(H.shape[0])
        return H.at[diag_idx, diag_idx].add(1.0 - mask)

    def merit(self, q: jax.Array, t: jax.Array, aux: Aux) -> jax.Array:
        """Line search objective: total energy if conservative, else `|R|^2 / 2`."""
        if self.is_conservative:
            return sum(
                (f.E(q, t, a) for f, a in zip(self.terms, aux, strict=True)),  # type: ignore
                jnp.zeros(()),
            )
        return 0.5 * jnp.sum(self.residual(q, t, aux) ** 2)

    def update(self, aux: Aux, q: jax.Array) -> Aux:
        """Update each term's aux at the converged state `q`."""
        return tuple(f.update(a, q) for f, a in zip(self.terms, aux, strict=True))

    def get_ode_term(self) -> diffrax.ODETerm:
        """Get `diffrax.ODETerm` to solve a ODE with `args=aux`."""

        if self.mass is None:
            raise ValueError("get_ode_term: system has no mass.")

        @eqx.filter_jit
        def rhs(t, y, aux):
            t = jnp.asarray(t)

            # split [q, v]
            n_dofs = self.q0.shape[0]
            q, v = y[:n_dofs], y[n_dofs:]

            # Get fixed DOF
            q_fixed, v_fixed = jax.jvp(
                lambda t: self.bc.apply(q, t), (t,), (jnp.ones_like(t),)
            )

            # update [q, v]
            v = v * self.bc.mask(q) + v_fixed * (1.0 - self.bc.mask(q))
            a = -self.residual(q_fixed, t, aux) / self.mass
            return jnp.concatenate([v, a])

        return diffrax.ODETerm(rhs)
