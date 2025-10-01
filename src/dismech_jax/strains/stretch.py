import jax
import jax.numpy as jnp


@jax.jit
def _get_strain(n0: jax.Array, n1: jax.Array, l_k: jax.Array) -> jax.Array:
    edge = n1 - n0
    edge_len = jnp.linalg.norm(edge)
    return edge_len / l_k - 1.0


@jax.jit
def _get_grad_strain(
    n0: jax.Array, n1: jax.Array, l_k: jax.Array
) -> tuple[jax.Array, jax.Array]:
    edge = n1 - n0
    edge_len = jnp.linalg.norm(edge)
    eps = edge_len / l_k - 1.0

    # Gradient
    tangent = edge / edge_len
    dF_unit = tangent / l_k
    dF = jnp.concatenate((-dF_unit, dF_unit))
    return eps, dF


@jax.jit
def _get_grad_hess_strain(
    n0: jax.Array, n1: jax.Array, l_k: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    edge = n1 - n0
    edge_len = jnp.linalg.norm(edge)
    eps = edge_len / l_k - 1.0

    # Gradient
    tangent = edge / edge_len
    dF_unit = tangent / l_k
    dF = jnp.concatenate((-dF_unit, dF_unit))

    # Hessian
    Id3 = jnp.eye(3)
    K_term = 1 / edge_len * (jnp.outer(edge, edge)) / edge_len**2
    M = 2.0 / l_k * ((1 / l_k - 1 / edge_len) * Id3 + K_term)
    M2 = jnp.where(
        jnp.isclose(eps, 0.0),
        jnp.zeros_like(M),
        1.0 / (2.0 * eps) * (M - 2.0 * jnp.outer(dF_unit, dF_unit)),
    )
    dJ = jnp.block(
        [
            [M2, -M2],
            [-M2, M2],
        ]
    )
    return eps, dF, dJ


@jax.jit
def get_stretch_strain(
    n0: jax.Array, n1: jax.Array, n2: jax.Array, l_k0: jax.Array, l_k1: jax.Array
) -> jax.Array:
    return 0.5 * (_get_strain(n0, n1, l_k0) + _get_strain(n1, n2, l_k1))


@jax.jit
def get_grad_stretch_strain(
    n0: jax.Array, n1: jax.Array, n2: jax.Array, l_k0: jax.Array, l_k1: jax.Array
) -> tuple[jax.Array, jax.Array]:
    epsx0, dF0 = _get_grad_strain(n0, n1, l_k0)
    epsx1, dF1 = _get_grad_strain(n1, n2, l_k1)
    eps = 0.5 * (epsx0 + epsx1)

    # Gradient
    dF0_n0 = dF0[:3]
    dF0_n1 = dF0[3:]
    dF1_n1 = dF1[:3]
    dF1_n2 = dF1[3:]
    dF = 0.5 * jnp.concatenate([dF0_n0, dF0_n1 + dF1_n1, dF1_n2])
    return eps, dF


@jax.jit
def get_grad_hess_stretch_strain(
    n0: jax.Array, n1: jax.Array, n2: jax.Array, l_k0: jax.Array, l_k1: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array]:
    epsx0, dF0, dJ0 = _get_grad_hess_strain(n0, n1, l_k0)
    epsx1, dF1, dJ1 = _get_grad_hess_strain(n1, n2, l_k1)
    eps = 0.5 * (epsx0 + epsx1)

    # Gradient
    dF0_n0 = dF0[:3]
    dF0_n1 = dF0[3:]
    dF1_n1 = dF1[:3]
    dF1_n2 = dF1[3:]
    dF = 0.5 * jnp.concatenate([dF0_n0, dF0_n1 + dF1_n1, dF1_n2])

    # Hessian
    J00 = dJ0[:3, :3]
    J01 = dJ0[:3, 3:]
    J10 = dJ0[3:, :3]
    J11 = dJ0[3:, 3:]
    J11_prime = dJ1[:3, :3]
    J12_prime = dJ1[:3, 3:]
    J21_prime = dJ1[3:, :3]
    J22_prime = dJ1[3:, 3:]
    J_11_total = J11 + J11_prime
    dJ = 0.5 * jnp.block(
        [
            [J00, J01, jnp.zeros((3, 3))],
            [J10, J_11_total, J12_prime],
            [jnp.zeros((3, 3)), J21_prime, J22_prime],
        ]
    )
    return eps, dF, dJ
