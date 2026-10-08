import jax.numpy as jnp
import numpy as np
from conftest import R0, RHO

from dismech_jax import Geometry, Material, make_rod


def test_energy_invariant_under_update():
    # The energy at `q` must not depend on whether the aux was updated at `q`:
    # otherwise an equilibrium stops being one after `update` (large steps).
    N = 11
    rod, aux = make_rod(
        Geometry(length=1.0, r0=R0),
        Material(density=RHO, youngs_rod=1e5, poisson_rod=0.3),
        N=N,
        fixed=jnp.arange(7),
    )
    s = np.linspace(0, 1, N)
    q = np.array(rod.q0)
    q[1::4] = 0.3 * np.sin(2 * s)  # out-of-plane helix-like bend: tangents rotate in 3D
    q[2::4] = 0.3 * (1 - np.cos(3 * s))
    q[3::4] = np.random.default_rng(0).normal(scale=0.5, size=N - 1)
    q = jnp.asarray(q)
    x, z = rod.split(q)

    aux_new = rod.update(aux, x, z)
    assert jnp.abs(aux_new[0].beta).max() > 1e-2, "test needs a nonzero reference twist"
    assert jnp.isclose(rod.energy(x, z, aux), rod.energy(x, z, aux_new), rtol=1e-10)
    # The update re-references the twist angles (theta -> theta + g(positions)), so
    # only the twist gradient is chart independent; with it zero, equilibria persist.
    tw = np.asarray(rod.idx_x) % 4 == 3
    np.testing.assert_allclose(
        rod.residual(x, z, aux)[tw],
        rod.residual(x, z, aux_new)[tw],
        rtol=1e-8,
        atol=1e-10,
    )


def test_equilibrium_survives_large_step():
    # One solve with a large twist and bend of the free end; the equilibrium must
    # still be one after the aux update.
    N = 15
    fixed = np.r_[np.arange(7), np.arange(4 * (N - 2), 4 * N - 1)]
    rod, aux = make_rod(
        Geometry(length=1.0, r0=R0),
        Material(density=RHO, youngs_rod=1e5, poisson_rod=0.3),
        N=N,
        fixed=jnp.asarray(fixed),
        gravity=jnp.zeros(3),
    )
    zpos = {int(d): i for i, d in enumerate(np.asarray(rod.idx_z))}
    z = np.array(rod.z0)
    for node in (N - 2, N - 1):
        z[zpos[4 * node]] -= 0.3  # compress
        z[zpos[4 * node + 2]] += 0.2  # lift
    z[zpos[4 * (N - 2) + 3]] = 2.0  # twist the last edge
    z = jnp.asarray(z)
    xs, res = rod.solve(z[None], aux, iters=200)
    assert res[-1] < 1e-8
    aux_new = rod.update(aux, xs[-1], z)
    assert jnp.linalg.norm(rod.residual(xs[-1], z, aux_new)) < 1e-6
