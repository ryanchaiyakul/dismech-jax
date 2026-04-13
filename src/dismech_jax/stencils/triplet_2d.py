import jax
import jax.numpy as jnp

from .stencil import Stencil


class Triplet2D(Stencil[None]):
    """Discrete diferential geometry strain triplet in 2D."""

    l_k: jax.Array  # [l_ke, l_kf]

    def get_strain(self, q: jax.Array, aux=None) -> jax.Array:
        l_ke, l_kf = self.l_k
        n0 = q[0:3]
        n1 = q[3:6]
        n2 = q[6:9]
        eps0 = self.get_epsilon(n0, n1, l_ke)
        eps1 = self.get_epsilon(n1, n2, l_kf)
        kappa = self.get_kappa_2d(n0, n1, n2)
        return jnp.array([eps0, eps1, kappa])

    @staticmethod
    def get_kappa_2d(n0: jax.Array, n1: jax.Array, n2: jax.Array) -> jax.Array:
        ee = n1 - n0
        ef = n2 - n1
        norm_e = jnp.linalg.norm(ee)
        norm_f = jnp.linalg.norm(ef)
        te = ee / norm_e
        tf = ef / norm_f
        chi = 1.0 + jnp.sum(te * tf)
        kb = 2.0 * jnp.cross(te, tf) / chi
        return kb[2]
