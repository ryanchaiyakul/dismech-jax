import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import R0, RHO

from dismech_jax import (
    Clipped,
    Geometry,
    Linear,
    Material,
    MaxTurn,
    Newton,
    Previous,
    SaddleFree,
    Tangent,
    make_rod,
)


def _rod(N=15, E=1e5, block_size=None, gravity=(0.0, 0.0, 0.0)):
    # Clamped at both ends (two nodes and the edge twist each)
    fixed = np.r_[np.arange(7), np.arange(4 * (N - 2), 4 * N - 1)]
    return make_rod(
        Geometry(length=1.0, r0=R0),
        Material(density=RHO, youngs_rod=E, poisson_rod=0.3),
        N=N,
        fixed=jnp.asarray(fixed),
        gravity=jnp.asarray(gravity),
        block_size=block_size,
    )


def _right_end(rod, N, dx=0.0, dz=0.0, twist=0.0):
    # Move the right clamp (last two nodes and the last edge twist)
    zpos = {int(d): i for i, d in enumerate(np.asarray(rod.idx_z))}
    z = rod.z0
    for node in (N - 2, N - 1):
        z = z.at[zpos[4 * node]].add(dx).at[zpos[4 * node + 2]].add(dz)
    return z.at[zpos[4 * (N - 2) + 3]].add(twist)


def test_saddle_free_is_newton_where_convex():
    rod, aux = _rod()
    z = _right_end(rod, 15, dz=0.05)
    x = rod.x0
    res = -rod.residual(x, z, aux)
    assert jnp.linalg.eigvalsh(rod.hessian(x, z, aux)[0])[0] > 0
    d_sf = SaddleFree()(rod, x, z, aux, res)
    d_n = Newton(reg=0.0)(rod, x, z, aux, res)
    np.testing.assert_allclose(d_sf, d_n, rtol=1e-6, atol=1e-9 * jnp.max(jnp.abs(d_n)))


def test_saddle_free_descends_at_saddle():
    # Straight rod under end compression: H is indefinite
    rod, aux = _rod()
    z = _right_end(rod, 15, dx=-0.2)
    x = rod.x0 + 1e-4 * jnp.sin(jnp.arange(rod.x0.shape[0]))
    assert jnp.linalg.eigvalsh(rod.hessian(x, z, aux)[0])[0] < 0
    res = -rod.residual(x, z, aux)
    assert jnp.dot(res, SaddleFree()(rod, x, z, aux, res)) > 0


def test_saddle_free_needs_dense():
    rod, aux = _rod(block_size=8)
    with pytest.raises(ValueError, match="dense"):
        rod.solve(rod.z0[None], aux, direction=SaddleFree())


def test_clipped():
    rod, aux = _rod()
    z = _right_end(rod, 15, dz=0.3)
    res = -rod.residual(rod.x0, z, aux)
    d = Clipped(Newton(), max_step=1e-3)(rod, rod.x0, z, aux, res)
    assert jnp.isclose(jnp.max(jnp.abs(d)), 1e-3)
    # Per-DOF limits: positions capped, twists unlimited
    tw = np.asarray(rod.idx_x) % 4 == 3
    cap = jnp.where(tw, jnp.inf, 1e-3)
    d = Clipped(Newton(), max_step=cap)(rod, rod.x0, z, aux, res)
    assert jnp.isclose(jnp.max(jnp.abs(d[~tw])), 1e-3)


def test_tangent_predictor_is_first_order():
    rod, aux = _rod(gravity=(0.0, 0.0, -9.81))
    xs, _ = rod.solve(rod.z0[None], aux, iters=50)
    aux = rod.update(aux, xs[0], rod.z0)
    errs = []
    for h in (0.02, 0.01):
        z = _right_end(rod, 15, dz=h)
        guess = {
            p: P(rod, xs[0], rod.z0, z, aux)
            for p, P in (("prev", Previous()), ("tangent", Tangent()))
        }
        errs.append(
            {
                p: float(jnp.linalg.norm(rod.residual(g, z, aux)))
                for p, g in guess.items()
            }
        )
    # Residual of the guess: O(h) for the previous state, O(h^2) for the tangent
    assert errs[1]["prev"] / errs[0]["prev"] == pytest.approx(0.5, rel=0.1)
    assert errs[1]["tangent"] / errs[0]["tangent"] == pytest.approx(0.25, rel=0.1)
    assert errs[1]["tangent"] < 0.05 * errs[1]["prev"]


def test_linear_predictor():
    rod, aux = _rod()
    nx, nz = rod.x0.shape[0], rod.z0.shape[0]
    W = jnp.arange(nx * nz, dtype=float).reshape(nx, nz) * 1e-3
    z = _right_end(rod, 15, dz=0.1)
    np.testing.assert_allclose(
        Linear(W)(rod, rod.x0, rod.z0, z, aux), rod.x0 + W @ (z - rod.z0)
    )
    np.testing.assert_allclose(Linear(0 * W)(rod, rod.x0, rod.z0, z, aux), rod.x0)


def _blend(rod, N):
    # Linear predictor: spread the right clamp's increment over the rod
    zpos = {int(d): i for i, d in enumerate(np.asarray(rod.idx_z))}
    idx = np.asarray(rod.idx_x)
    W = np.zeros((idx.size, rod.z0.shape[0]))
    for r, d in enumerate(idx):
        W[r, zpos[4 * (N - 2) + d % 4]] = (d // 4 - 1) / (N - 3)
    return Linear(jnp.asarray(W))


def test_max_turn_blocks_and_passes_follow():
    # One solve that turns edges by more than 60 degrees (end lifted and pushed in)
    N = 15
    rod, aux = _rod(N)
    z = _right_end(rod, N, dx=-0.6, dz=0.3, twist=1.0)
    acc = MaxTurn()

    pred = _blend(rod, N)
    xs, res = rod.solve(z[None], aux, iters=200, predictor=pred)
    t0 = aux[0].t[:, 0]
    q = rod.join(xs[0], z)
    e = jnp.diff(jnp.stack([q[0::4], q[1::4], q[2::4]], 1), axis=0)[:-1]
    turn = jnp.sum(e * t0, 1) / jnp.linalg.norm(e, axis=1)
    assert res[0] < 1e-8 and jnp.min(turn) < 0.5, "test needs a large turn"
    assert not acc(rod, xs[0], z, aux)

    # A single pass stops at the turn limit; more passes reach the equilibrium
    xs1, res1 = rod.solve(z[None], aux, iters=200, predictor=pred, accept=acc)
    assert res1[0] > 1e-6 and acc(rod, xs1[0], z, aux)
    xsp, resp = rod.solve(
        z[None], aux, iters=200, predictor=pred, accept=acc, passes=10
    )
    assert resp[0] < 1e-8


def test_grad_with_strategies():
    # Gradients stay exact (IFT with the true Hessian) whatever the strategies
    N = 15
    rod, aux = _rod(N, gravity=(0.0, 0.0, -9.81))
    pred = _blend(rod, N)

    def tip(s, **kw):
        z = _right_end(rod, N, dx=-0.1 * s, dz=0.05 * s, twist=0.5 * s)
        zs = rod.z0 + jnp.linspace(0.25, 1, 4)[:, None] * (z - rod.z0)
        xs, _ = rod.solve(zs, aux, iters=100, **kw)
        return rod.join(xs[-1], zs[-1])[4 * (N // 2) + 2]

    for kw in (
        {},
        dict(direction=Newton()),
        dict(predictor=pred, accept=MaxTurn(), passes=5),
        dict(predictor=Tangent()),
    ):
        g = jax.grad(lambda s: tip(s, **kw))(1.0)
        fd = (tip(1.0 + 1e-5, **kw) - tip(1.0 - 1e-5, **kw)) / 2e-5
        assert jnp.isclose(g, fd, rtol=1e-4), (kw, g, fd)


def test_buckling_converges_to_stable_branch():
    # Compressing a straight clamped rod passes a buckling bifurcation where the
    # Hessian is indefinite; the saddle-free step must land on a stable buckled state.
    N = 21
    fixed = np.r_[np.arange(7), np.arange(4 * (N - 2), 4 * N - 1)]
    rod, aux = make_rod(
        Geometry(length=1.0, r0=R0),
        Material(density=RHO, youngs_rod=1e5, poisson_rod=0.3),
        N=N,
        fixed=jnp.asarray(fixed),
        gravity=jnp.array([0.0, 0.0, -1e-3]),  # imperfection
    )
    zpos = {int(d): i for i, d in enumerate(np.asarray(rod.idx_z))}
    cols = [zpos[4 * n] for n in (N - 2, N - 1)]
    zs = jnp.tile(rod.z0, (10, 1))
    zs = zs.at[:, cols].add(-0.3 * jnp.linspace(0.1, 1, 10)[:, None])
    xs, res = rod.solve(zs, aux, iters=100)
    assert jnp.all(res < 1e-8)
    q = rod.join(xs[-1], zs[-1])
    assert abs(q[4 * (N // 2) + 2]) > 0.1  # buckled out of the line
    aux_end = aux
    for x, z in zip(xs, zs):
        aux_end = rod.update(aux_end, x, z)
    H, _ = rod.hessian(xs[-1], zs[-1], aux_end)
    assert jnp.linalg.eigvalsh(H)[0] > 0  # stable
