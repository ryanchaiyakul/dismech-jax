"""1D slinky toy system: two-node spring under prescribed end displacement."""

import jax
import jax.numpy as jnp
import equinox as eqx

import dismech_jax as djx


class Slinky1D(djx.System):
    """Simple 2-node 1D spring under exact displacement control."""

    l_k: jax.Array
    x_left: jax.Array

    def get_q(self, disp: jax.Array, q0: jax.Array) -> jax.Array:
        """
        ``disp`` is the exact prescribed right-node x-position.
        If stored data are displacement relative to the initial position instead
        of absolute position, use ``q.at[1].set(q0[1] + disp)``.
        """
        q = q0.at[0].set(self.x_left)
        q = q.at[1].set(disp)
        return q

    def get_eps(self, q: jax.Array) -> jax.Array:
        return (q[1] - q[0]) / self.l_k - 1.0

    def get_E(self, disp: jax.Array, q: jax.Array, model: eqx.Module, aux: None) -> jax.Array:
        eps = self.get_eps(q)
        return model(jnp.array([eps]))

    def get_F(self, disp: jax.Array, q: jax.Array, model: eqx.Module, aux: None) -> jax.Array:
        return jnp.zeros_like(q)

    def get_H(self, disp: jax.Array, q: jax.Array, model: eqx.Module, aux: None) -> jax.Array:
        return jnp.eye(q.shape[0])

    def get_reaction_force(self, disp: jax.Array, q: jax.Array, model: eqx.Module, aux: None) -> jax.Array:
        dEdq = jax.grad(self.get_E, argnums=1)(disp, q, model, aux)
        return dEdq[1]
