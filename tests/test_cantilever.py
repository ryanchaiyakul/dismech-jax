import equinox as eqx
import jax
import jax.numpy as jnp
from conftest import EI, L, W

RESOLUTIONS = (21, 41, 81)


def test_gravity(build_rod):
    errs = []
    for N in RESOLUTIONS:
        rod, aux = build_rod(N)
        zs = rod.z0[None]
        xs, res = rod.solve(zs, aux, iters=20)
        assert jnp.all(res < 1e-6 * W * L), "Newton did not converge"
        q = rod.join(xs[-1], zs[-1])

        h = L / (N - 1)
        x, z = q[0::4], q[2::4]
        s, Ls = jnp.maximum(x - h / 2, 0.0), L - h / 2
        z_ref = -W * s**2 * (6 * Ls**2 - 4 * Ls * s + s**2) / (24 * EI)
        assert abs(z_ref[-1]) < 0.01 * L, "small-deflection regime violated"
        errs.append(float(jnp.max(jnp.abs(z - z_ref)) / jnp.max(jnp.abs(z_ref))))

    assert errs[-1] < 2e-4, f"errors {errs}"
    for coarse, fine in zip(errs, errs[1:]):
        assert fine < 0.35 * coarse, f"errors {errs}"  # ~second order


def test_grad_gravity(build_rod):
    rod, aux = build_rod(41)
    zs = rod.z0[None]
    F0 = rod.terms[1].F_ext

    def tip_z(s):
        r = eqx.tree_at(lambda r: r.terms[1].F_ext, rod, s * F0)
        xs, _ = r.solve(zs, aux, iters=30)
        return r.join(xs[-1], zs[-1])[-1]

    g = jax.grad(tip_z)(1.0)
    fd = (tip_z(1.0 + 1e-6) - tip_z(1.0 - 1e-6)) / 2e-6
    assert jnp.isclose(g, fd, rtol=1e-5)
    assert jnp.isclose(g, tip_z(1.0), rtol=1e-3)
