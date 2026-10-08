from __future__ import annotations

import jax
import jax.numpy as jnp

from .energy import Energy


class Attractor(Energy[None]):
    """Spring pulling the DOFs `idx` toward targets held in other DOFs:

        E = k/2 * |q[idx] - q[idx_target]|^2,  k = q[idx_k].

    The targets and `k` live in the state, normally as fixed DOFs, so they can
    change every load step through the `zs` of `solve` (set `k = 0` to release).
    Use `System.with_attractor` to append them to a system."""

    idx: jax.Array
    idx_target: jax.Array
    idx_k: jax.Array

    def E(self, q: jax.Array, aux: None) -> jax.Array:
        d = q[self.idx] - q[self.idx_target]
        return 0.5 * q[self.idx_k] * jnp.sum(d**2)

    def F(self, q: jax.Array, aux: None) -> jax.Array:
        d = q[self.idx] - q[self.idx_target]
        k = q[self.idx_k]
        return (
            jnp.zeros_like(q)
            .at[self.idx]
            .add(k * d)
            .at[self.idx_target]
            .add(-k * d)
            .at[self.idx_k]
            .add(0.5 * jnp.sum(d**2))
        )


class Leading[AuxT](Energy[AuxT]):
    """`term` acting on the leading `n` DOFs of a longer state, e.g. after
    appending the ghost DOFs of an `Attractor`."""

    term: Energy[AuxT]
    n: int

    def E(self, q: jax.Array, aux: AuxT) -> jax.Array:
        return self.term.E(q[: self.n], aux)

    def F(self, q: jax.Array, aux: AuxT) -> jax.Array:
        return jnp.zeros_like(q).at[: self.n].set(self.term.F(q[: self.n], aux))

    def H(self, q: jax.Array, aux: AuxT) -> jax.Array:
        H = jnp.zeros((q.shape[0], q.shape[0]), q.dtype)
        return H.at[: self.n, : self.n].set(self.term.H(q[: self.n], aux))

    def update(self, aux: AuxT, q: jax.Array) -> AuxT:
        return self.term.update(aux, q[: self.n])
