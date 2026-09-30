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

    def H(self, q: jax.Array, aux: AuxT) -> jax.Array:
        H_local = jax.vmap(
            lambda s, ql, a: jax.hessian(s.get_energy)(ql, self.model, a)
        )(self.stencils, q[self.conn], aux)
        H = jnp.zeros((q.shape[0], q.shape[0]), dtype=q.dtype)
        return H.at[self.conn[:, :, None], self.conn[:, None, :]].add(H_local)

    def update(self, aux: AuxT, q: jax.Array) -> AuxT:
        if aux is None:
            return aux
        return jax.vmap(lambda a, ql: a.update(ql))(aux, q[self.conn])
