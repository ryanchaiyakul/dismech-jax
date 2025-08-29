import jax.numpy as jnp
import dismech_jax


def test_separate_joint_edges():
    # Null
    rre, rse = dismech_jax.Mesh.separate_joint_edges(
        jnp.empty((0, 2)), jnp.empty((0, 3))
    )
    assert rre.size == 0 and rse.size == 0

    # Only edges
    rre, rse = dismech_jax.Mesh.separate_joint_edges(
        jnp.array([[0, 1], [1, 2], [2, 3]]), jnp.empty((0, 3))
    )
    assert jnp.array_equal(rre, jnp.array([[0, 1], [1, 2], [2, 3]])) and rse.size == 0
