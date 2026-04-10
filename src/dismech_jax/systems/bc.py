from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class AbstractBC(eqx.Module):
    """Functional interface for boundary conditions."""

    def apply(self, q: jax.Array, _lambda: jax.Array) -> jax.Array:
        """Applies the boundary condition to the state vector q."""
        raise NotImplementedError

    def mask(self, q: jax.Array) -> jax.Array:
        """Returns a boolean/float mask where 0.0 represents a constrained DOF."""
        raise NotImplementedError

    @property
    def in_axes(self) -> AbstractBC | None:
        """Returns the in_axes for vmap. Default is None (not batched)."""
        return None


class LinearBC(AbstractBC):
    """Linear boundary conditions."""

    idx_b: jax.Array
    xb_m: jax.Array
    xb_c: jax.Array

    def apply(self, q: jax.Array, _lambda: jax.Array) -> jax.Array:
        values = self.xb_m * _lambda + self.xb_c

        if values.ndim == 2 and values.shape[0] == 1:
            values = jnp.squeeze(values, axis=0)

        return q.at[self.idx_b].set(values)

    def mask(self, q: jax.Array) -> jax.Array:
        return jnp.ones_like(q).at[self.idx_b].set(0.0)


class BatchedLinearBC(LinearBC):
    """Linear boundary conditions where xb_m is batched."""

    @property
    def in_axes(self) -> "BatchedLinearBC" | None:
        return BatchedLinearBC(idx_b=None, xb_c=None, xb_m=0)  # type: ignore


class DirectBC(AbstractBC):
    """
    Direct boundary conditions with lambda-dependent prescribed values.

    xb has shape (n_lambda, n_b), and lambdas has shape (n_lambda,).
    At runtime, the BC vector corresponding to the closest lambda is used.
    """

    idx_b: jax.Array
    xb: jax.Array
    lambdas: jax.Array

    def apply(self, q: jax.Array, _lambda: jax.Array) -> jax.Array:
        lam = self.lambdas
        xb = self.xb

        # clamp to range
        lam_query = jnp.clip(_lambda, lam[0], lam[-1])

        # find right interval index i so that lam[i] <= lam_query <= lam[i+1]
        i = jnp.searchsorted(lam, lam_query, side="right") - 1
        i = jnp.clip(i, 0, lam.shape[0] - 2)

        lam0 = lam[i]
        lam1 = lam[i + 1]
        xb0 = xb[i]
        xb1 = xb[i + 1]

        t = (lam_query - lam0) / jnp.maximum(lam1 - lam0, 1e-12)
        values = (1.0 - t) * xb0 + t * xb1

        return q.at[self.idx_b].set(values)

    def mask(self, q: jax.Array) -> jax.Array:
        return jnp.ones_like(q).at[self.idx_b].set(0.0)


class BatchedDirectBC(DirectBC):
    """Direct boundary conditions where xb is batched over trajectories."""

    @property
    def in_axes(self) -> "BatchedDirectBC" | None:
        return BatchedDirectBC(idx_b=None, xb=0, lambdas=None)  # type: ignore