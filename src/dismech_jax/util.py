from jax.typing import ArrayLike
import jax
import jax.numpy as jnp


def map_node_to_dof(n: ArrayLike) -> jax.Array:
    return 3 * jnp.asarray(n)[..., None] + jnp.arange(3)