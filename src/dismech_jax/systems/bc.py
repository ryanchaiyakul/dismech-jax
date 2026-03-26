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

        # Handle accidental singleton leading dimension, e.g. (1, n_b) -> (n_b,)
        if values.ndim == 2 and values.shape[0] == 1:
            values = jnp.squeeze(values, axis=0)

        return q.at[self.idx_b].set(values)

    def mask(self, q: jax.Array) -> jax.Array:
        return jnp.ones_like(q).at[self.idx_b].set(0.0)


class BatchedLinearBC(LinearBC):
    """Linear boundary conditions where xb_m is batched."""

    @property
    def in_axes(self) -> BatchedLinearBC | None:
        return BatchedLinearBC(idx_b=None, xb_c=None, xb_m=0)  # type: ignore
