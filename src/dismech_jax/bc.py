from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp


class AbstractBC(eqx.Module):
    """Functional interface for boundary conditions."""

    def apply(self, q: jax.Array, t: jax.Array) -> jax.Array:
        """Applies the boundary condition to the state vector q."""
        raise NotImplementedError

    def mask(self, q: jax.Array) -> jax.Array:
        """Returns a boolean/float mask where 0.0 represents a constrained DOF."""
        raise NotImplementedError


class LinearBC(AbstractBC):
    """Linear boundary conditions `q[idx_b] = xb_m * t + xb_c`."""

    idx_b: jax.Array
    xb_m: jax.Array
    xb_c: jax.Array

    def apply(self, q: jax.Array, t: jax.Array) -> jax.Array:
        return q.at[self.idx_b].set(self.xb_m * t + self.xb_c)

    def mask(self, q: jax.Array) -> jax.Array:
        return jnp.ones_like(q).at[self.idx_b].set(0.0)
