from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass


from .geometry import Geometry
from .state import State
from .util import map_node_to_dof


@register_dataclass
@dataclass(frozen=True)
class Triplet:
    """
    Internal 3-node stencil. Externally avaliable through `Triplets`
    """

    index: jax.Array  # idx
    dof: jax.Array  # [x0, y0, z0, x1, y1, z1, x2, y2, z2, e1, e2]
    sign: jax.Array  # [-1/+1, -1/+1]
    K: jax.Array  # [ks1, ks2, kb1, kb2, kt]
    # Computed in from_state
    ref_len: jax.Array  # [len1, len2]
    ini_strain: jax.Array  # [s1, s2, b1, b2, t]

    @classmethod
    def from_state(
        cls,
        index: jax.Array,
        dof: jax.Array,
        sign: jax.Array,
        K: jax.Array,
        state: State,
    ) -> Triplet:
        ref_len = jnp.array(
            [
                jnp.linalg.norm(state.q[dof[3:6]] - state.q[dof[0:3]]),
                jnp.linalg.norm(state.q[dof[6:9]] - state.q[dof[3:6]]),
            ]
        )
        tmp = Triplet(index, dof, sign, K, ref_len, jnp.empty(5))
        return Triplet(index, dof, sign, K, ref_len, tmp.get_strain(state))

    def get_stretch_strain(self, state: State) -> jax.Array:
        e1 = jnp.linalg.norm(state.q[self.n1] - state.q[self.n0])
        e2 = jnp.linalg.norm(state.q[self.n2] - state.q[self.n1])
        return jnp.array([e1 + e2]) / self.ref_len - 1.0

    def get_bend_strain(self, state: State) -> jax.Array:
        m1e = state.m1[self.e0]
        m2e = state.m2[self.e0] * self.sign[0]
        m1f = state.m1[self.e1]
        m2f = state.m2[self.e1] * self.sign[1]
        ee = state.q[self.n1] - state.q[self.n0]
        ef = state.q[self.n2] - state.q[self.n1]
        te = ee / jnp.maximum(jnp.linalg.norm(ee), 1e-12)
        tf = ef / jnp.maximum(jnp.linalg.norm(ef), 1e-12)
        chi = 1.0 + jnp.dot(te, tf)
        kb = 2.0 * jnp.cross(te, tf) / chi
        kappa1 = 0.5 * jnp.dot(kb, m2e + m2f)
        kappa2 = 0.5 * jnp.dot(kb, m1e + m1f)
        return jnp.array([kappa1, kappa2])

    def get_twist_strain(self, state: State) -> jax.Array:
        theta_e = state.q[self.e0] * self.sign[0]
        theta_f = state.q[self.e1] * self.sign[1]
        return jnp.array([theta_f - theta_e + state.ref_twist[self.index]])

    def get_strain(self, state: State) -> jax.Array:
        return jnp.concat(
            [
                self.get_stretch_strain(state),
                self.get_bend_strain(state),
                self.get_twist_strain(state),
            ]
        )

    def get_energy(self, state: State) -> jax.Array:
        return jnp.sum(self.K * (self.get_strain(state) - self.ini_strain) ** 2)

    def get_grad_energy(self, state: State) -> jax.Array:
        def energy_wrt_q(q: jax.Array) -> jax.Array:
            new_state = state.__replace__(q=q)
            return self.get_energy(new_state)

        return jax.grad(energy_wrt_q)(state.q)

    def get_hess_energy(self, state: State) -> jax.Array:
        def energy_wrt_q(q: jax.Array) -> jax.Array:
            new_state = state.__replace__(q=q)
            return self.get_energy(new_state)

        return jax.hessian(energy_wrt_q)(state.q)

    @property
    def n0(self) -> jax.Array:
        return self.dof[0:3]

    @property
    def n1(self) -> jax.Array:
        return self.dof[3:6]

    @property
    def n2(self) -> jax.Array:
        return self.dof[6:9]

    @property
    def e0(self) -> jax.Array:
        return self.dof[10]

    @property
    def e1(self) -> jax.Array:
        return self.dof[11]


@register_dataclass
@dataclass
class Triplets:
    ts: Triplet

    # TODO: create more natural mesh alternative
    @classmethod
    def from_geo(cls, geo: Geometry, state: State) -> Triplets:
        bt_springs = jnp.asarray(geo.bend_twist_springs, dtype=jnp.int32)
        dofs = jnp.hstack(
            (
                map_node_to_dof(bt_springs[:, [0, 2, 4]]).reshape(
                    bt_springs.shape[0], -1
                ),
                bt_springs[:, [1, 3]],
            )
        )
        N = dofs.shape[0]
        K = jnp.broadcast_to(
            jnp.array((20000, 20000, 0.00166667, 0.66666667, 0.00111111)), (N, 5)
        )
        signs = jnp.ones((N, 2))
        index = jnp.arange(N)
        return cls(
            jax.vmap(Triplet.from_state, (0, 0, 0, 0, None))(
                index, dofs, signs, K, state
            )
        )

    def get_stretch_strain(self, state: State) -> jax.Array:
        return jax.vmap(lambda t: t.get_stretch_strain(state))(self.ts)

    def get_bend_strain(self, state: State) -> jax.Array:
        return jax.vmap(lambda t: t.get_bend_strain(state))(self.ts)

    def get_twist_strain(self, state: State) -> jax.Array:
        return jax.vmap(lambda t: t.get_twist_strain(state))(self.ts)

    def get_energy(self, state: State) -> jax.Array:
        return jax.vmap(lambda t: t.get_energy(state))(self.ts)

    def get_grad_energy(self, state: State) -> jax.Array:
        return jnp.sum(jax.vmap(lambda t: t.get_grad_energy(state))(self.ts), axis=0)

    def get_hess_energy(self, state: State) -> jax.Array:
        return jnp.sum(jax.vmap(lambda t: t.get_hess_energy(state))(self.ts), axis=0)
