from __future__ import annotations

from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp

from ..states import State
from ..stencils import Stencil
from .force import Energy


class StencilEnergy[AuxT: State | None](Energy[AuxT]):
    """Internal energy assembled from a batch of identical stencils.

    `conn[s]` holds the global DOF indices of stencil `s` in the order its
    `get_strain` expects. `model` is the constitutive law shared by every
    stencil and `aux` is batched alongside the stencils.
    """

    stencils: Stencil[AuxT]  # batched, leading dim S
    conn: jax.Array  # (S, n_local) int: local -> global DOF
    model: eqx.Module  # f(del_strain) -> energy density

    def E(self, q: jax.Array, t: jax.Array, aux: AuxT) -> jax.Array:
        return jnp.sum(
            jax.vmap(lambda s, ql, a: s.get_energy(ql, self.model, a))(
                self.stencils, q[self.conn], aux
            )
        )

    def H(self, q: jax.Array, t: jax.Array, aux: AuxT) -> jax.Array:
        """Scatter-assemble the local stencil Hessians."""
        H_local = jax.vmap(
            lambda s, ql, a: jax.hessian(s.get_energy)(ql, self.model, a)
        )(self.stencils, q[self.conn], aux)
        H = jnp.zeros((q.shape[0], q.shape[0]), dtype=q.dtype)
        return H.at[self.conn[:, :, None], self.conn[:, None, :]].add(H_local)

    def update(self, aux: AuxT, q: jax.Array) -> AuxT:
        """Update each stencil's aux with its own local DOFs."""
        if aux is None:
            return aux
        return jax.vmap(lambda a, ql: a.update(ql))(aux, q[self.conn])

    def with_rest(self, q_rest: jax.Array, aux: AuxT) -> Self:
        """Get a copy whose rest strain is measured at `q_rest`.

        Differentiable, so gradients w.r.t. a rest configuration flow through.
        """
        bar_strain = jax.vmap(lambda s, ql, a: s.get_strain(ql, a))(
            self.stencils, q_rest[self.conn], aux
        )
        return eqx.tree_at(lambda e: e.stencils.bar_strain, self, bar_strain)
