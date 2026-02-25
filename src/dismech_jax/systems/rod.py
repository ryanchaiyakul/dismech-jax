from __future__ import annotations

import jax
import jax.numpy as jnp
import equinox as eqx

from ..models import DER
from ..stencils import Triplet
from ..states import TripletState
from ..params import Geometry, Material
from .system import System


class BC(eqx.Module):
    """Linear boundary condition."""

    idx_b: jax.Array
    xb_m: jax.Array
    xb_c: jax.Array


class Rod(System[TripletState]):
    triplets: Triplet
    F_ext: jax.Array
    bc: BC

    @classmethod
    def from_geometry(
        cls, geom: Geometry, material: Material, N: int = 30
    ) -> tuple[Rod, jax.Array, TripletState]:
        if N < 3:
            raise ValueError("Cannot create a rod with less than 3 nodes.")
        if geom.length < 1e-6:
            raise ValueError("Cannot create a rod less than 1 um.")

        q0 = jnp.zeros(4 * N - 1)
        xs = jnp.linspace(0, geom.length, N)
        q0 = q0.at[0::4].set(xs)
        batch_q = Rod.global_q_to_batch_q(q0)

        l_ks = jnp.diff(xs)
        mass = Rod.get_mass(geom, material, l_ks)

        N_triplets = batch_q.shape[0]
        batch_l_ks = jax.vmap(lambda i: jax.lax.dynamic_slice(l_ks, (i,), (2,)))(
            jnp.arange(N_triplets)
        )

        t_pair = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        d1_pair = jnp.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        ts = jnp.broadcast_to(t_pair, (N_triplets, 2, 3))
        d1s = jnp.broadcast_to(d1_pair, (N_triplets, 2, 3))
        betas = jnp.zeros(N_triplets)

        batch_aux = jax.vmap(TripletState)(ts, d1s, betas)
        triplets = jax.vmap(lambda q, a, l_k: Triplet.init(q, a, l_k=l_k))(
            batch_q, batch_aux, batch_l_ks
        )

        F_ext = jnp.zeros_like(q0).at[2::4].set(mass[2::4] * -9.81)
        temp_bc = BC(jnp.array([0]), jnp.array([0.0]), jnp.array([0.0]))
        rod = Rod(triplets=triplets, F_ext=F_ext, bc=temp_bc)
        return rod, q0, batch_aux

    def with_bc(self, bc: BC) -> Rod:
        return eqx.tree_at(lambda r: r.bc, self, bc)

    def get_DER(self, geom: Geometry, material: Material) -> DER:
        return DER.from_legacy(self.triplets.l_k[0, 0], geom, material)

    def get_q(self, _lambda: jax.Array, q0: jax.Array) -> jax.Array:
        return q0.at[self.bc.idx_b].set(self.bc.xb_m * _lambda + self.bc.xb_c)

    def get_F(self, q: jax.Array, model: eqx.Module, aux: TripletState) -> jax.Array:
        mask = jnp.ones_like(q).at[self.bc.idx_b].set(0.0)
        F_int = jax.grad(self.get_energy, 0)(q, model, aux)
        return mask * (self.F_ext - F_int)

    def get_H(self, q: jax.Array, model: eqx.Module, aux: TripletState) -> jax.Array:
        mask = jnp.ones_like(q).at[self.bc.idx_b].set(0.0)
        H = jax.hessian(self.get_energy, 0)(q, model, aux)
        H = H * mask[:, None] * mask[None, :]
        diag_idx = jnp.arange(H.shape[0])
        return H.at[diag_idx, diag_idx].add(1.0 - mask)

    def get_energy(self, q: jax.Array, model: eqx.Module, aux: TripletState) -> jax.Array:
        batch_qs = self.global_q_to_batch_q(q)
        return jnp.sum(
            jax.vmap(lambda t, q_loc, _aux: t.get_energy(q_loc, model, _aux))(
                self.triplets, batch_qs, aux
            )
        )

    @staticmethod
    def global_q_to_batch_q(q: jax.Array) -> jax.Array:
        N_triplets = (q.shape[0] + 1) // 4 - 2
        starts = jnp.arange(N_triplets) * 4
        return jax.vmap(lambda s: jax.lax.dynamic_slice(q, (s,), (11,)))(starts)

    @staticmethod
    def get_mass(geom: Geometry, material: Material, l_ks: jax.Array) -> jax.Array:
        N = l_ks.shape[0] + 1  # Number of nodes
        mass = jnp.zeros(N * 4 - 1)
        A = geom.axs if geom.axs else jnp.pi * geom.r0**2

        # Node contributions
        weights = 0.5 * l_ks[0]
        v_ref_len = jnp.ones(N) * weights
        v_ref_len = v_ref_len.at[0].add(weights)
        v_ref_len = v_ref_len.at[-1].add(weights)
        dm_nodes = v_ref_len * A * material.density
        node_dofs = jnp.arange(3 * N).reshape(-1, 3)
        mass = mass.at[node_dofs].add(dm_nodes[:, None])

        # Edge contributions (moment of inertia)
        factor = geom.jxs / geom.axs if geom.jxs and geom.axs else geom.r0**2 / 2
        edge_mass = l_ks * A * material.density * factor
        edge_dofs = 3 * N + jnp.arange(N - 1)
        mass = mass.at[edge_dofs].set(edge_mass)
        return mass
