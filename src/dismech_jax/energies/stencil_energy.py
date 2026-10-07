from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from ..stencils import Stencil
from .energy import Energy


class StencilEnergy[AuxT](Energy[AuxT]):
    stencils: Stencil[AuxT]
    conn: jax.Array
    model: eqx.Module

    def E(self, q: jax.Array, aux: AuxT) -> jax.Array:
        return jnp.sum(
            jax.vmap(lambda s, ql, a: s.get_energy(ql, self.model, a))(
                self.stencils, q[self.conn], aux
            )
        )

    def _H_local(self, q: jax.Array, aux: AuxT) -> jax.Array:
        return jax.vmap(lambda s, ql, a: jax.hessian(s.get_energy)(ql, self.model, a))(
            self.stencils, q[self.conn], aux
        )

    def H(self, q: jax.Array, aux: AuxT) -> jax.Array:
        H_local = self._H_local(q, aux)
        H = jnp.zeros((q.shape[0], q.shape[0]), dtype=q.dtype)
        return H.at[self.conn[:, :, None], self.conn[:, None, :]].add(H_local)

    def H_blocks(self, q: jax.Array, aux: AuxT, b: int) -> tuple[jax.Array, jax.Array]:
        # Requires every stencil to span at most two consecutive blocks
        # (checked by `System.create`).
        H_local = self._H_local(q, aux)
        nb = -(-q.shape[0] // b)
        rows = jnp.broadcast_to(self.conn[:, :, None], H_local.shape)
        cols = jnp.broadcast_to(self.conn[:, None, :], H_local.shape)
        bi, bj, ri, rj = rows // b, cols // b, rows % b, cols % b

        # One flat buffer: diagonal blocks, sub-diagonal blocks `(bj + 1, bj)`,
        # then a dump slot for the upper blocks (the transpose of the lower ones).
        n_diag, n_sub = nb * b * b, (nb - 1) * b * b
        idx = jnp.where(
            bi == bj,
            (bi * b + ri) * b + rj,
            jnp.where(bi == bj + 1, n_diag + (bj * b + ri) * b + rj, n_diag + n_sub),
        )
        flat = (
            jnp.zeros(n_diag + n_sub + 1, q.dtype).at[idx.ravel()].add(H_local.ravel())
        )
        return flat[:n_diag].reshape(nb, b, b), flat[n_diag:-1].reshape(nb - 1, b, b)

    def update(self, aux: AuxT, q: jax.Array) -> AuxT:
        if aux is None:
            return aux
        return jax.vmap(lambda a, ql: a.update(ql))(aux, q[self.conn])
