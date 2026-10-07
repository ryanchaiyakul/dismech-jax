from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from ..energies import ConstantForceEnergy, Energy, StencilEnergy
from ..models import DER
from ..states import TripletState
from ..stencils import Triplet
from ..system import System
from .params import Geometry, Material


def make_rod(
    geom: Geometry,
    material: Material,
    N: int = 30,
    fixed: jax.Array | None = None,
    origin: jax.Array | None = None,
    gravity: jax.Array | None = None,
    model: eqx.Module | None = None,
    extra_terms: tuple[Energy, ...] = (),
    block_size: int | None = None,
) -> tuple[System, tuple]:
    """Build a discrete elastic rod with DOFs `[x0, y0, z0, theta0, x1, ..., zN]`.

    The terms are `(StencilEnergy[Triplet], ConstantForceEnergy, *extra_terms)`, so aux
    is `(TripletState, None, *[None] * len(extra_terms))` for aux-free extras.

    Args:
        geom (Geometry): Geometry object.
        material (Material): Material object.
        N (int, optional): Number of nodes. Defaults to 30.
        fixed (jax.Array | None, optional): Indices of the fixed DOFs. Their
            values in `solve` are the `zs` passed there; `sys.z0` is the
            undeformed value. Defaults to none (all DOFs free).
        origin (jax.Array | None, optional): Position of the first node.
            Defaults to `[0, 0, 0]`.
        gravity (jax.Array | None, optional): Gravitational acceleration.
            Defaults to `[0, 0, -9.81]`.
        model (eqx.Module | None, optional): Constitutive law. Defaults to
            `DER.from_legacy(geom, material)`.
        extra_terms (tuple[Energy, ...], optional): Additional aux-free energy
            terms. Defaults to none.
        block_size (int | None, optional): Block size of the block tridiagonal
            Newton solve (see `System.create`). A triplet spans 11 consecutive
            DOFs, so 8 is the smallest valid size. Defaults to None (dense).

    Returns:
        tuple[System, tuple]: System and initial aux.
    """
    if N < 3:
        raise ValueError("Cannot create a rod with less than 3 nodes.")
    if geom.length < 1e-6:
        raise ValueError("Cannot create a rod less than 1 um.")
    if fixed is None:
        fixed = jnp.array([], dtype=int)
    if origin is None:
        origin = jnp.array([0.0, 0.0, 0.0])
    if gravity is None:
        gravity = jnp.array([0.0, 0.0, -9.81])
    if model is None:
        model = DER.from_legacy(geom, material)

    q0 = jnp.zeros(4 * N - 1)
    xs = jnp.linspace(0, geom.length, N) + origin[0]
    q0 = q0.at[0::4].set(xs)
    q0 = q0.at[1::4].set(origin[1])
    q0 = q0.at[2::4].set(origin[2])

    # Triplet s spans nodes s, s+1, s+2 and edges s, s+1
    N_triplets = N - 2
    conn = jnp.arange(N_triplets)[:, None] * 4 + jnp.arange(11)[None, :]

    l_ks = jnp.diff(xs)
    batch_l_ks = jnp.stack([l_ks[:-1], l_ks[1:]], axis=1)

    t_pair = jnp.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    d1_pair = jnp.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    tangents = jnp.broadcast_to(t_pair, (N_triplets, 2, 3))
    d1s = jnp.broadcast_to(d1_pair, (N_triplets, 2, 3))
    betas = jnp.zeros(N_triplets)

    batch_aux = jax.vmap(TripletState)(tangents, d1s, betas)
    triplets = jax.vmap(lambda q, a, l_k: Triplet.init(q, a, l_k=l_k))(
        q0[conn], batch_aux, batch_l_ks
    )

    mass = _get_mass(geom, material, l_ks)
    mass_reshaped = jnp.pad(mass, (0, 1)).reshape(-1, 4)
    F_reshaped = jnp.zeros_like(mass_reshaped)
    F_reshaped = F_reshaped.at[:, :3].set(mass_reshaped[:, :3] * gravity)
    F_ext = F_reshaped.ravel()[:-1]

    rod = System.create(
        terms=(
            StencilEnergy(triplets, conn, model),
            ConstantForceEnergy(F_ext),
            *extra_terms,
        ),
        q0=q0,
        fixed=fixed,
        block_size=block_size,
    )
    return rod, (batch_aux, None, *[None] * len(extra_terms))


def _get_mass(geom: Geometry, material: Material, l_ks: jax.Array) -> jax.Array:
    N = l_ks.shape[0] + 1  # Number of nodes
    mass = jnp.zeros(N * 4 - 1)
    A = geom.axs if geom.axs else jnp.pi * geom.r0**2

    # Node contributions
    weights = 0.5 * l_ks[0]
    v_ref_len = jnp.ones(N) * 2 * weights
    v_ref_len = v_ref_len.at[0].set(weights)
    v_ref_len = v_ref_len.at[-1].set(weights)
    dm_nodes = v_ref_len * A * material.density
    node_start_indices = jnp.arange(N) * 4
    for i in range(3):  # Fill x, y, and z
        mass = mass.at[node_start_indices + i].set(dm_nodes)

    # Edge contributions (moment of inertia)
    factor = geom.jxs / geom.axs if geom.jxs and geom.axs else geom.r0**2 / 2
    dm_edges = l_ks * A * material.density * factor
    edge_indices = jnp.arange(N - 1) * 4 + 3
    mass = mass.at[edge_indices].set(dm_edges)
    return mass
