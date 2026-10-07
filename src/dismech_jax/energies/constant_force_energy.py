from __future__ import annotations

import jax
import jax.numpy as jnp

from .energy import Energy


class ConstantForceEnergy(Energy[None]):
    F_ext: jax.Array

    def E(self, q: jax.Array, aux: None) -> jax.Array:
        return -jnp.sum(self.F_ext * q)

    def F(self, q: jax.Array, aux: None) -> jax.Array:
        return -self.F_ext

    def H(self, q: jax.Array, aux: None) -> jax.Array:
        return jnp.zeros((q.shape[0], q.shape[0]), dtype=q.dtype)

    def H_blocks(self, q: jax.Array, aux: None, b: int) -> tuple[jax.Array, jax.Array]:
        nb = -(-q.shape[0] // b)
        return jnp.zeros((nb, b, b), q.dtype), jnp.zeros((nb - 1, b, b), q.dtype)
