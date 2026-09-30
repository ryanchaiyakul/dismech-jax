from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .energies import Energy
from .solver import solve

type Aux = tuple[Any, ...]


class System(eqx.Module):
    terms: tuple[Energy, ...]
    idx_x: jax.Array
    idx_z: jax.Array
    q0: jax.Array

    @classmethod
    def create(
        cls, terms: Sequence[Energy], q0: jax.Array, fixed: jax.Array | Sequence[int]
    ) -> System:
        """Build a system from `terms`, the initial state `q0` and the indices of
        the fixed DOFs `fixed`."""
        n = q0.shape[0]
        fixed = np.unique(np.asarray(fixed, dtype=int))
        if fixed.size and (fixed[0] < 0 or fixed[-1] >= n):
            raise ValueError("`fixed` contains an index outside of the state.")
        free = np.setdiff1d(np.arange(n), fixed)
        return cls(tuple(terms), jnp.asarray(free), jnp.asarray(fixed), q0)

    @property
    def n_dofs(self) -> int:
        return self.q0.shape[0]

    @property
    def x0(self) -> jax.Array:
        """Free DOFs of `q0`."""
        return self.q0[self.idx_x]

    @property
    def z0(self) -> jax.Array:
        """Fixed DOFs of `q0`."""
        return self.q0[self.idx_z]

    def join(self, x: jax.Array, z: jax.Array) -> jax.Array:
        """Assemble the full state `q` from `x` and `z` (leading batch dims ok)."""
        dtype = jnp.result_type(x, z)
        q = jnp.zeros(x.shape[:-1] + (self.n_dofs,), dtype=dtype)
        return q.at[..., self.idx_x].set(x).at[..., self.idx_z].set(z)

    def split(self, q: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Split the full state `q` into `(x, z)`."""
        return q[..., self.idx_x], q[..., self.idx_z]

    def energy(self, x: jax.Array, z: jax.Array, aux: Aux) -> jax.Array:
        """Total potential energy."""
        q = self.join(x, z)
        return sum(
            (f.E(q, a) for f, a in zip(self.terms, aux, strict=True)),
            jnp.zeros(()),
        )

    def _grad(self, x: jax.Array, z: jax.Array, aux: Aux) -> jax.Array:
        q = self.join(x, z)
        return sum(
            (f.F(q, a) for f, a in zip(self.terms, aux, strict=True)),
            jnp.zeros_like(q),
        )

    def residual(self, x: jax.Array, z: jax.Array, aux: Aux) -> jax.Array:
        """Gradient of the energy w.r.t. the free DOFs, `dE/dx`. Zero at equilibrium."""
        return self._grad(x, z, aux)[self.idx_x]

    def support_force(self, x: jax.Array, z: jax.Array, aux: Aux) -> jax.Array:
        """Force the supports exert on the fixed DOFs, `dE/dz`."""
        return self._grad(x, z, aux)[self.idx_z]

    def hessian(
        self, x: jax.Array, z: jax.Array, aux: Aux
    ) -> tuple[jax.Array, jax.Array]:
        """Blocks `(H_xx, H_xz)` of the energy Hessian."""
        q = self.join(x, z)
        H = sum(
            (f.H(q, a) for f, a in zip(self.terms, aux, strict=True)),
            jnp.zeros((self.n_dofs, self.n_dofs), dtype=q.dtype),
        )
        return H[jnp.ix_(self.idx_x, self.idx_x)], H[jnp.ix_(self.idx_x, self.idx_z)]

    def update(self, aux: Aux, x: jax.Array, z: jax.Array) -> Aux:
        """Update each term's aux at the state `(x, z)`."""
        q = self.join(x, z)
        return tuple(f.update(a, q) for f, a in zip(self.terms, aux, strict=True))

    def solve(
        self,
        zs: jax.Array,
        aux: Aux,
        x0: jax.Array | None = None,
        iters: int = 10,
        ls_steps: int = 10,
        c1: float = 1e-4,
        tol: float = 1e-10,
    ) -> tuple[jax.Array, jax.Array]:
        """Solve for equilibrium at each fixed state in `zs`. See `solver.solve`."""
        return solve(self, zs, aux, x0, iters, ls_steps, c1, tol)
