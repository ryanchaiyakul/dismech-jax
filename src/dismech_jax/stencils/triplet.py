from __future__ import annotations
from dataclasses import dataclass
from typing import Self
import abc

from jax.tree_util import register_dataclass
import jax
import jax.numpy as jnp

from dismech_jax.state import State
from dismech_jax.util import map_node_to_dof
from dismech_jax.strains import (
    get_stretch_strain,
    get_grad_stretch_strain,
    get_grad_hess_stretch_strain,
)


@register_dataclass
@dataclass
class BaseTriplet(abc.ABC):
    node_dofs: jax.Array  # [n0, n1, n2]
    # edge_dofs: jax.Array # [e0, e1]
    l_k: jax.Array  # [l_k0, l_k1]
    nat_strain: jax.Array  # [ε, κ₁, κ₂, τ]

    @jax.jit
    def get_strain(self, state: State) -> jax.Array:
        return self._static_get_strain(self.node_dofs, self.l_k, state)

    @staticmethod
    @jax.jit
    def _static_get_strain(node_dofs: jax.Array, l_k, state: State) -> jax.Array:
        n0, n1, n2 = state.q[node_dofs]
        return jnp.array([get_stretch_strain(n0, n1, n2, l_k[0], l_k[1])])

    @jax.jit
    def get_grad_strain(self, state: State) -> tuple[jax.Array, jax.Array]:
        # 4 x 11
        n0, n1, n2 = state.q[self.node_dofs]
        strain, grad = get_grad_stretch_strain(n0, n1, n2, self.l_k[0], self.l_k[1])
        return jnp.array([strain]), grad[None, ...]

    @jax.jit
    def get_grad_hess_strain(
        self, state: State
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # 4 x 11 x 11
        n0, n1, n2 = state.q[self.node_dofs]
        strain, grad, hess = get_grad_hess_stretch_strain(
            n0, n1, n2, self.l_k[0], self.l_k[1]
        )
        return jnp.array([strain]), grad[None, ...], hess[None, ...]

    @jax.jit
    def get_energy(self, state: State) -> jax.Array:
        del_strain = self.get_strain(state) - self.nat_strain
        return self._core_energy_func(del_strain)

    @jax.jit
    def _core_energy_func(self, del_strain: jax.Array) -> jax.Array:
        return 0.5 * del_strain.T @ self.get_K(del_strain) @ del_strain

    @jax.jit
    def get_grad_energy(self, state: State) -> jax.Array:
        strain, grad_strain = self.get_grad_strain(state)
        del_strain = strain - self.nat_strain
        # gradient of energy w.r.t. strain
        dE_dstrain = jax.grad(self._core_energy_func)(del_strain)
        return grad_strain.T @ dE_dstrain

    @jax.jit
    def get_grad_hess_energy(self, state: State) -> tuple[jax.Array, jax.Array]:
        strain, grad_strain, hess_strain = self.get_grad_hess_strain(state)
        del_strain = strain - self.nat_strain

        # gradient & hessian of energy w.r.t. strain
        dE_dstrain = jax.grad(self._core_energy_func)(del_strain)
        H_eps = jax.hessian(self._core_energy_func)(del_strain)

        hess_energy = grad_strain.T @ H_eps @ grad_strain
        hess_energy += jnp.tensordot(dE_dstrain, hess_strain, axes=1)
        return grad_strain.T @ dE_dstrain, hess_energy

    @abc.abstractmethod
    @jax.jit
    def get_K(self, strain: jax.Array) -> jax.Array: ...


@register_dataclass
@dataclass
class DERTriplet(BaseTriplet):
    K: jax.Array  # diagonal: [EA, EI1, EI2, GJ]

    @classmethod
    def init(
        cls,
        nodes: jax.Array,
        l_k: jax.Array,
        l_eff: jax.Array,
        EA: jax.Array,
        EI: jax.Array,
        GJ: jax.Array,
        state: State,
    ) -> DERTriplet:
        diag = jnp.array(
            [EA * jnp.mean(l_k)]  # , EI[0] / l_eff, EI[1] / l_eff, GJ / l_eff]
        )
        K = jnp.diag(diag)
        node_dofs = map_node_to_dof(nodes)
        nat_strain = cls._static_get_strain(node_dofs, l_k, state)
        return cls(node_dofs, l_k, nat_strain, K)

    @jax.jit
    def get_K(self, _: jax.Array) -> jax.Array:
        return self.K

@register_dataclass
@dataclass
class Triplet(BaseTriplet):

    @classmethod
    def init(
        cls,
        nodes: jax.Array,
        l_k: jax.Array,
        l_eff: jax.Array,
        state: State,
    ) -> Self:
        node_dofs = map_node_to_dof(nodes)
        nat_strain = cls._static_get_strain(node_dofs, l_k, state)
        return cls(node_dofs, l_k, nat_strain)
