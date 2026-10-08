import jax
import jax.numpy as jnp
import numpy as np
from conftest import R0, RHO

from dismech_jax import Geometry, Material, MaxTurn, make_rod


def _rod(N=11):
    fixed = np.r_[np.arange(7), np.arange(4 * (N - 2), 4 * N - 1)]
    return make_rod(
        Geometry(length=1.0, r0=R0),
        Material(density=RHO, youngs_rod=1e5, poisson_rod=0.3),
        N=N,
        fixed=jnp.asarray(fixed),
    )


def _attracted(N=11):
    rod, aux = _rod(N)
    pos = np.asarray(rod.idx_x)[np.asarray(rod.idx_x) % 4 != 3]  # free positions
    sys, aux_a = rod.with_attractor(aux, pos)
    return rod, aux, sys, aux_a, pos


def test_layout_and_force():
    rod, aux, sys, aux_a, pos = _attracted()
    assert sys.n_dofs == rod.n_dofs + pos.size + 1
    np.testing.assert_array_equal(sys.idx_x, rod.idx_x)
    assert sys.z0.shape[0] == rod.z0.shape[0] + pos.size + 1
    q = sys.q0 + 1e-2 * jnp.sin(jnp.arange(sys.n_dofs))
    q = q.at[-1].set(3.0)  # k
    att = sys.terms[-1]
    np.testing.assert_allclose(att.F(q, None), jax.grad(att.E)(q, None), rtol=1e-12)


def test_k_zero_is_the_rod_and_large_k_reaches_target():
    rod, aux, sys, aux_a, pos = _attracted()
    zr = rod.z0.at[-3].add(0.2)  # lift the right end (node N-1 z)
    x_rod, res = rod.solve(zr[None], aux, iters=50)
    assert res[0] < 1e-8

    target = jnp.asarray(rod.q0)[pos] + 0.05  # arbitrary target
    for k, check in ((0.0, "rod"), (1e6, "target")):
        z = jnp.concatenate([zr, target, jnp.array([k])])
        xs, res = sys.solve(z[None], aux_a, iters=50)
        assert res[0] < 1e-8
        if check == "rod":
            np.testing.assert_allclose(xs[0], x_rod[0], atol=1e-10)
        else:
            np.testing.assert_allclose(sys.join(xs[0], z)[pos], target, atol=1e-4)


def test_return_aux():
    rod, aux = _rod()
    zs = rod.z0 + jnp.linspace(0.05, 0.2, 4)[:, None] * (
        jnp.arange(rod.z0.shape[0]) == rod.z0.shape[0] - 2
    )
    xs, res, auxs = rod.solve(zs, aux, iters=50, return_aux=True)
    a = aux
    for x, z in zip(xs, zs):
        a = rod.update(a, x, z)
    np.testing.assert_allclose(auxs[0].t[-1], a[0].t, atol=1e-12)
    np.testing.assert_allclose(auxs[0].beta[-1], a[0].beta, atol=1e-12)


def test_max_turn_on_attracted_system():
    rod, aux, sys, aux_a, pos = _attracted()
    z = jnp.concatenate([rod.z0, jnp.asarray(rod.q0)[pos], jnp.array([1.0])])
    assert MaxTurn()(sys, sys.x0, z, aux_a)
