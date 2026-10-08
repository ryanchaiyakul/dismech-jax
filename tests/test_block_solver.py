import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import G, L, R0, RHO

from dismech_jax import Geometry, Material, Newton, make_rod
from dismech_jax.linalg import block_tridiag_solve


def _rod(N, block_size, E=1e5):
    # Soft rod: large deflection, so the solves are nonlinear
    return make_rod(
        Geometry(length=L, r0=R0),
        Material(density=RHO, youngs_rod=E, poisson_rod=0.3),
        N=N,
        fixed=jnp.arange(7),
        gravity=jnp.array([0.0, 0.0, -G]),
        block_size=block_size,
    )


def test_block_tridiag_solve():
    rng = np.random.default_rng(0)
    nb, b = 5, 8
    A = np.zeros((nb * b, nb * b))
    for k in range(nb):
        M = rng.normal(size=(b, b))
        A[k * b : (k + 1) * b, k * b : (k + 1) * b] = M @ M.T + b * np.eye(b)
        if k < nb - 1:
            C = rng.normal(size=(b, b))
            A[(k + 1) * b : (k + 2) * b, k * b : (k + 1) * b] = C
            A[k * b : (k + 1) * b, (k + 1) * b : (k + 2) * b] = C.T
    D = np.stack([A[k * b : (k + 1) * b, k * b : (k + 1) * b] for k in range(nb)])
    Lo = np.stack(
        [A[(k + 1) * b : (k + 2) * b, k * b : (k + 1) * b] for k in range(nb - 1)]
    )
    r = rng.normal(size=nb * b)
    x = block_tridiag_solve(
        jnp.asarray(D), jnp.asarray(Lo), jnp.asarray(r).reshape(nb, b)
    )
    np.testing.assert_allclose(x.ravel(), np.linalg.solve(A, r), rtol=1e-10, atol=1e-12)


def test_block_size_too_small():
    with pytest.raises(ValueError, match="block tridiagonal"):
        _rod(11, block_size=4)


def test_matches_dense():
    (dense, aux), (block, _) = _rod(21, None), _rod(21, 8)
    x, z = dense.split(dense.q0 + 1e-2 * jnp.sin(jnp.arange(dense.n_dofs)))
    v = jnp.cos(jnp.arange(x.shape[0]))
    np.testing.assert_allclose(
        block.linear_solver(x, z, aux)(v), dense.linear_solver(x, z, aux)(v), rtol=1e-9
    )

    zs = dense.z0[None]
    xd, rd = dense.solve(zs, aux, iters=30)
    xb, rb = block.solve(zs, aux, iters=30)
    assert rd[-1] < 1e-8 and rb[-1] < 1e-8
    np.testing.assert_allclose(xb, xd, atol=1e-10)


def test_grad_matches_dense():
    def tip_z(rod, aux, s):
        # Lift the clamp (node 1 z) over a few load steps (path dependent). The
        # hanging soft rod has a (twist) saddle, so compare the same plain Newton
        # step: the saddle-free default would leave it, which the block solve can't.
        zs = jnp.tile(rod.z0, (3, 1)).at[:, 6].set(s * jnp.arange(1, 4) / 3)
        xs, _ = rod.solve(zs, aux, iters=30, direction=Newton())
        return rod.join(xs[-1], zs[-1])[-1]  # tip z

    (dense, aux), (block, _) = _rod(21, None), _rod(21, 8)
    gd = jax.grad(lambda s: tip_z(dense, aux, s))(0.02)
    gb = jax.grad(lambda s: tip_z(block, aux, s))(0.02)
    assert abs(gd) > 1e-3
    assert jnp.isclose(gb, gd, rtol=1e-8)
