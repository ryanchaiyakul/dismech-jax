from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .energies import Attractor, Energy, Leading, StencilEnergy
from .linalg import (
    block_tridiag_factor,
    block_tridiag_matvec,
    block_tridiag_solve_factored,
)
from .solver import solve

if TYPE_CHECKING:
    from .acceptance import Accept
    from .directions import Direction
    from .predictors import Predictor

type Aux = tuple[Any, ...]


class System(eqx.Module):
    terms: tuple[Energy, ...]
    idx_x: jax.Array
    idx_z: jax.Array
    q0: jax.Array
    block_size: int | None = eqx.field(static=True, default=None)

    @classmethod
    def create(
        cls,
        terms: Sequence[Energy],
        q0: jax.Array,
        fixed: jax.Array | Sequence[int],
        block_size: int | None = None,
    ) -> System:
        """Build a system from `terms`, the initial state `q0` and the indices of
        the fixed DOFs `fixed`.

        With `block_size`, the Hessian is treated as block tridiagonal with blocks
        of `block_size` DOFs and Newton uses a block Thomas solve (`O(n b^2)`)
        instead of a dense one (`O(n^3)`). Every term must implement `H_blocks`
        and every stencil must span at most two consecutive blocks."""
        n = q0.shape[0]
        fixed = np.unique(np.asarray(fixed, dtype=int))
        if fixed.size and (fixed[0] < 0 or fixed[-1] >= n):
            raise ValueError("`fixed` contains an index outside of the state.")
        free = np.setdiff1d(np.arange(n), fixed)
        if block_size is not None:
            for term in terms:
                if isinstance(term, StencilEnergy):
                    conn = np.asarray(term.conn)
                    span = (
                        conn.max(axis=1) // block_size - conn.min(axis=1) // block_size
                    )
                    if np.any(span > 1):
                        raise ValueError(
                            f"`block_size={block_size}` is too small: a stencil spans "
                            f"{int(span.max()) + 1} blocks, so the Hessian is not "
                            "block tridiagonal."
                        )
        return cls(tuple(terms), jnp.asarray(free), jnp.asarray(fixed), q0, block_size)

    def with_attractor(
        self, aux: Aux, idx: jax.Array | Sequence[int]
    ) -> tuple[System, Aux]:
        """Append an `Attractor` pulling the DOFs `idx` toward targets.

        The state gets `len(idx) + 1` fixed ghost DOFs at its end, the targets
        then the stiffness `k`, so the fixed DOFs become `[z, target, k]`
        (`z` as before) and both can change every load step through `zs`.
        `x` is unchanged. In `sys.z0` the targets are `q0[idx]` and `k = 0`.
        Dense only (the ghost DOFs break the block structure)."""
        if self.block_size is not None:
            raise ValueError("`with_attractor` needs a dense system.")
        n, idx = self.n_dofs, jnp.asarray(idx)
        m = idx.shape[0]
        q0 = jnp.concatenate([self.q0, self.q0[idx], jnp.zeros(1, self.q0.dtype)])
        attractor = Attractor(idx, n + jnp.arange(m), jnp.asarray(n + m))
        sys = System.create(
            (*(Leading(t, n) for t in self.terms), attractor),
            q0,
            np.concatenate([np.asarray(self.idx_z), n + np.arange(m + 1)]),
        )
        return sys, (*aux, None)

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

    def linear_solver(
        self, x: jax.Array, z: jax.Array, aux: Aux, reg: float = 1e-8
    ) -> Callable[[jax.Array], jax.Array]:
        """`v -> (H_xx + reg I)^-1 v` at the state `(x, z)`.

        Dense unless `block_size` is set, then block tridiagonal."""
        if self.block_size is None:
            H, _ = self.hessian(x, z, aux)
            H = H.at[jnp.diag_indices(H.shape[0])].add(reg)
            return lambda v: jnp.linalg.solve(H, v)

        b, q = self.block_size, self.join(x, z)
        D, L = (
            sum(Ds)
            for Ds in zip(
                *(f.H_blocks(q, a, b) for f, a in zip(self.terms, aux, strict=True))
            )
        )
        # Keep the full-state layout so the blocks stay aligned: fixed and padding
        # DOFs get identity rows and columns (decoupled, solution 0).
        nb = D.shape[0]
        m = jnp.zeros(nb * b, q.dtype).at[self.idx_x].set(1.0).reshape(nb, b)
        D = D * m[:, :, None] * m[:, None, :] + jax.vmap(jnp.diag)(1.0 - m + reg * m)
        L = L * m[1:, :, None] * m[:-1, None, :]
        factors = block_tridiag_factor(D, L)

        def solve_blocks(v: jax.Array) -> jax.Array:
            r = (
                jnp.zeros(nb * b, v.dtype)
                .at[self.idx_x]
                .set(v, unique_indices=True)
                .reshape(nb, b)
            )
            x = jax.lax.custom_linear_solve(  # transposes as a solve (see linalg)
                lambda y: block_tridiag_matvec(D, L, y),
                r,
                lambda _matvec, r: block_tridiag_solve_factored(factors, r),
                symmetric=True,
            )
            return x.reshape(-1)[self.idx_x]

        return solve_blocks

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
        direction: Direction | None = None,
        predictor: Predictor | None = None,
        accept: Accept | None = None,
        passes: int = 1,
        z0: jax.Array | None = None,
        return_aux: bool = False,
    ) -> tuple[jax.Array, jax.Array] | tuple[jax.Array, jax.Array, Aux]:
        """Solve for equilibrium at each fixed state in `zs`. See `solver.solve`."""
        return solve(
            self,
            zs,
            aux,
            x0,
            iters,
            ls_steps,
            c1,
            tol,
            direction,
            predictor,
            accept,
            passes,
            z0,
            return_aux,
        )
