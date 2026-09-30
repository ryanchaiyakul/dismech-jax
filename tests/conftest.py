"""Shared cantilever rod: a rod clamped at node 0 (and node 1, fixing the tangent)."""

import math

import jax
import jax.numpy as jnp
import pytest

from dismech_jax import Geometry, Material, make_rod

jax.config.update("jax_enable_x64", True)

# Defaults of `build_rod`
L, R0, E, RHO, G = 1.0, 0.01, 1e10, 1200.0, 9.81
EI = E * math.pi * R0**4 / 4
W = RHO * math.pi * R0**2 * G  # weight per unit length


@pytest.fixture
def build_rod():
    """`(N, **overrides) -> (System, aux)` for a cantilever rod under gravity.

    `L`, `R0`, `E`, `RHO` and `G` can be overridden per call.
    """

    def build(N: int, L=L, R0=R0, E=E, RHO=RHO, G=G):
        return make_rod(
            Geometry(length=L, r0=R0),
            Material(density=RHO, youngs_rod=E, poisson_rod=0.3),
            N=N,
            fixed=jnp.arange(7),
            gravity=jnp.array([0.0, 0.0, -G]),
        )

    return build
