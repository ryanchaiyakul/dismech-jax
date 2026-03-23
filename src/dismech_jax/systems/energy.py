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

class LinearExtForceEnergy(AbstractEnergy):
    """External energy with linearly varying force:
       F_ext(lambda) = F_m * lambda + F_c
    """

    F_m: jax.Array
    F_c: jax.Array

    def __call__(self, q: jax.Array, _lambda: jax.Array) -> jax.Array:
        F_ext = self.F_m * _lambda + self.F_c
        return -jnp.sum(F_ext * q)
    
class BatchedLinearExtForceEnergy(LinearExtForceEnergy):
    @property
    def in_axes(self) -> BatchedLinearExtForceEnergy | None:
        return BatchedLinearExtForceEnergy(F_m=0, F_c=0)  # type: ignore