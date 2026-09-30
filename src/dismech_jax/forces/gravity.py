from __future__ import annotations

import jax
import jax.numpy as jnp

from .force import Energy


class Gravity(Energy[None]):
    """Constant external force field `F_ext` (potential `-F_ext . q`)."""

    F_ext: jax.Array

    def E(self, q: jax.Array, t: jax.Array, aux: None) -> jax.Array:
        return -jnp.sum(self.F_ext * q)

    def F(self, q: jax.Array, t: jax.Array, aux: None) -> jax.Array:
        return self.F_ext

    def H(self, q: jax.Array, t: jax.Array, aux: None) -> jax.Array:
        return jnp.zeros((q.shape[0], q.shape[0]), dtype=q.dtype)
