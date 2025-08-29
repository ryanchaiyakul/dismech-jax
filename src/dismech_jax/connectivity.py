from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

from .geometry import Geometry
from .util import map_node_to_dof


@register_dataclass
@dataclass(frozen=True)
class Connectivity:
    # edges: jax.Array  # [n1, n2]
    # triangles: jax.Array # [n1, n2, n3]
    # triplets: jax.Array # [n1, e1, n2, e2, n3]

    # Precomputed DOF helpers
    edge_node_dofs: jax.Array  # nodal DOFs (,2,3)
    edge_dofs: jax.Array  # edge DOFs (,1)
    triplet_edge_dofs: jax.Array  # edge DOFS (,2)
    triplet_signs: jax.Array  # edge sign (+1, -1) (,2)

    @classmethod
    def init(
        cls,
        nodes: jax.Array,
        edges: jax.Array,
        triplets: jax.Array,
        triplet_signs: jax.Array,
    ) -> Connectivity:
        """Create intermediate constructs from nodes, edges, and traingles (tbd)"""
        return Connectivity(
            edge_node_dofs=map_node_to_dof(edges)
            if edges.size
            else jnp.empty((0, 2, 3), dtype=edges.dtype),
            edge_dofs=jnp.arange(
                nodes.shape[0],
                nodes.shape[0] + edges.shape[0],
            ),
            triplet_edge_dofs=triplets[:, [1, 3]],
            triplet_signs=triplet_signs,
        )

    @classmethod
    def from_geo(cls, geo: Geometry) -> Connectivity:
        return Connectivity.init(
            jnp.asarray(geo.nodes, dtype=jnp.int32),
            jnp.asarray(geo.edges, dtype=jnp.int32),
            jnp.asarray(geo.bend_twist_springs, dtype=jnp.int32),
            jnp.asarray(geo.bend_twist_signs, dtype=jnp.int32),
        )
