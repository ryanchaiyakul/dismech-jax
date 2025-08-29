import jax
import jax.numpy as jnp
from jax.typing import ArrayLike


def map_node_to_dof(n: ArrayLike) -> jax.Array:
    return 3 * jnp.asarray(n)[..., None] + jnp.arange(3)


def parallel_transport(u: jax.Array, t1: jax.Array, t2: jax.Array) -> jax.Array:
    b = jnp.cross(t1, t2)
    b_norm = jnp.linalg.norm(b)
    b_unit = b / jnp.maximum(b_norm, 1e-12)

    # Gram-Schmidt orthogonalization from dismech-matlab
    # https://github.com/StructuresComp/dismech-matlab/blob/main/util_functions/parallel_transport.m
    b_unit = b_unit - jnp.dot(b_unit, t1) * t1
    b_unit = b_unit / jnp.maximum(jnp.linalg.norm(b_unit), 1e-12)
    b_unit = b_unit - jnp.dot(b_unit, t2) * t2
    b_unit = b_unit / jnp.maximum(jnp.linalg.norm(b_unit), 1e-12)

    n1 = jnp.cross(t1, b_unit)
    n2 = jnp.cross(t2, b_unit)
    transported = (
        jnp.dot(u, t1) * t2 + jnp.dot(u, n1) * n2 + jnp.dot(u, b_unit) * b_unit
    )
    return jnp.where(b_norm < 1e-12, u, transported)


def signed_angle(u: jax.Array, v: jax.Array, n: ArrayLike) -> jax.Array:
    w = jnp.cross(u, v)
    w_norm = jnp.linalg.norm(w)
    dot_uv = jnp.dot(u, v)
    angle = jnp.atan2(w_norm, dot_uv)
    sign = jnp.sign(jnp.dot(n, w))  # no branching
    return angle * sign


def rotate_axis_angle(u: jax.Array, v: jax.Array, theta: ArrayLike) -> jax.Array:
    return (
        jnp.cos(theta) * u
        + jnp.sin(theta) * jnp.cross(v, u)
        + jnp.dot(v, u) * (1 - jnp.cos(theta)) * v
    )
