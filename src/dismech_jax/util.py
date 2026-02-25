import jax
import jax.numpy as jnp


def parallel_transport(u: jax.Array, t0: jax.Array, t1: jax.Array) -> jax.Array:
    b = jnp.cross(t0, t1)
    d = jnp.dot(t0, t1)
    denom = 1.0 + d + 1e-8
    b_cross_u = jnp.cross(b, u)
    return u + b_cross_u + jnp.cross(b, b_cross_u) / denom


def signed_angle(u: jax.Array, v: jax.Array, n: jax.Array) -> jax.Array:
    w = jnp.cross(u, v)
    dot_uv = jnp.dot(u, v)
    signed_sin = jnp.dot(w, n)
    return jnp.atan2(signed_sin, dot_uv)


def rotate_axis_angle(u: jax.Array, v: jax.Array, theta: jax.Array) -> jax.Array:
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    return c * u + s * jnp.cross(v, u) + jnp.dot(v, u) * (1 - c) * v


def material_frame(
    d1_old: jax.Array, t_old: jax.Array, t_new: jax.Array, theta: jax.Array
) -> tuple[jax.Array, jax.Array]:
    d1_new = parallel_transport(d1_old, t_old, t_new)
    d2_new = jnp.cross(t_new, d1_new)
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    m1 = c * d1_new + s * d2_new
    m2 = -s * d1_new + c * d2_new
    return m1, m2


def get_ref_twist(
    d1e: jax.Array, d1f: jax.Array, te: jax.Array, tf: jax.Array, r: jax.Array
) -> jax.Array:
    ut = parallel_transport(d1e, te, tf)
    ut = rotate_axis_angle(ut, tf, r)
    return r + signed_angle(ut, d1f, tf)
