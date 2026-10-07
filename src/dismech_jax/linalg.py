import jax
import jax.numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve

type BlockFactors = tuple[jax.Array, jax.Array, jax.Array, jax.Array]


def block_tridiag_factor(D: jax.Array, L: jax.Array) -> BlockFactors:
    """Block LU (Thomas) factorization of a symmetric block tridiagonal `A`.

    No pivoting between blocks (each block's LU pivots), so `A` should be well
    conditioned block by block, e.g. a regularized Hessian.

    Args:
        D (jax.Array): diagonal blocks `(nb, b, b)`, `D[k] = A[k, k]`.
        L (jax.Array): sub-diagonal blocks `(nb - 1, b, b)`, `L[k] = A[k + 1, k]`.

    Returns:
        BlockFactors: factors for `block_tridiag_solve_factored`.
    """
    zero = jnp.zeros_like(D[:1])
    L_k = jnp.concatenate([zero, L])  # A[k, k - 1]
    U_k = jnp.concatenate([jnp.swapaxes(L, 1, 2), zero])  # A[k, k + 1]

    # Schur complements S_k = D_k - L_k S_{k-1}^-1 U_{k-1}, with GU_k = S_k^-1 U_k
    def step(GU_prev, blk):
        D, L, U = blk
        lu, piv = lu_factor(D - L @ GU_prev)
        GU = lu_solve((lu, piv), U)
        return GU, (lu, piv, GU)

    _, (lu, piv, GU) = jax.lax.scan(step, jnp.zeros_like(D[0]), (D, L_k, U_k))
    return lu, piv, GU, L_k


def block_tridiag_solve_factored(factors: BlockFactors, rhs: jax.Array) -> jax.Array:
    """Solve `A x = rhs` `(nb, b)` from `block_tridiag_factor(D, L)`. Linear in `rhs`."""
    lu, piv, GU, L_k = factors

    def forward(gy_prev, blk):  # gy_k = S_k^-1 (r_k - L_k gy_{k-1})
        lu, piv, L, r = blk
        gy = lu_solve((lu, piv), r - L @ gy_prev)
        return gy, gy

    _, gy = jax.lax.scan(forward, jnp.zeros_like(rhs[0]), (lu, piv, L_k, rhs))

    def backward(x_next, blk):  # x_k = gy_k - GU_k x_{k+1}
        GU, gy = blk
        x = gy - GU @ x_next
        return x, x

    _, x = jax.lax.scan(backward, jnp.zeros_like(rhs[0]), (GU, gy), reverse=True)
    return x


def block_tridiag_matvec(D: jax.Array, L: jax.Array, x: jax.Array) -> jax.Array:
    """`A x` `(nb, b)` for a symmetric block tridiagonal `A` (see `block_tridiag_factor`)."""
    y = jnp.einsum("kij,kj->ki", D, x)
    y = y.at[1:].add(jnp.einsum("kij,kj->ki", L, x[:-1]))
    return y.at[:-1].add(jnp.einsum("kji,kj->ki", L, x[1:]))


def block_tridiag_solve(D: jax.Array, L: jax.Array, rhs: jax.Array) -> jax.Array:
    """Solve `A x = rhs` for a symmetric block tridiagonal `A`. See
    `block_tridiag_factor` for `D` and `L`; `rhs` is `(nb, b)`.

    Differentiable in `rhs` in forward and reverse mode. Wrapped in
    `custom_linear_solve` so the transpose is another solve with `A` rather than
    a transpose of the scans (which JAX cannot do here)."""
    factors = block_tridiag_factor(D, L)
    return jax.lax.custom_linear_solve(
        lambda x: block_tridiag_matvec(D, L, x),
        rhs,
        lambda _matvec, r: block_tridiag_solve_factored(factors, r),
        symmetric=True,
    )
