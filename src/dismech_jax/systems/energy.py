from __future__ import annotations


import equinox as eqx
import jax
import jax.numpy as jnp


class AbstractEnergy(eqx.Module):
    """Functional interface for external forces (via potential energy)."""

    def __call__(self, q: jax.Array, _lambda: jax.Array) -> jax.Array:
        """Returns the scalar potential energy of the external field."""
        raise NotImplementedError

    @property
    def in_axes(self) -> AbstractEnergy | None:
        """Returns the in_axes for vmap. Default is None (not batched)."""
        return None


class Gravity(AbstractEnergy):
    F_ext: jax.Array

    def __call__(self, q: jax.Array, _lambda: jax.Array) -> jax.Array:
        return -jnp.sum(self.F_ext * q)
