import jax
import jax.numpy as jnp

from .stencil import Stencil
from ..states import TripletState
from ..util import material_frame


class Triplet(Stencil[TripletState]):
    """Discrete diferential geometry strain triplet."""

    l_k: jax.Array  # [l_ke, l_kf]

    def get_strain(self, q: jax.Array, aux: TripletState) -> jax.Array:
        assert type(aux) is TripletState
        te_old, tf_old = aux.t
        d1e, d1f = aux.d1
        beta = aux.beta
        l_ke, l_kf = self.l_k
        n0 = q[0:3]
        n1 = q[4:7]
        n2 = q[8:11]
        theta_e = q[3]
        theta_f = q[7]
        ee = n1 - n0
        ef = n2 - n1
        te = ee / jnp.linalg.norm(ee)
        tf = ef / jnp.linalg.norm(ef)
        m1e, m2e = material_frame(d1e, te_old, te, theta_e)
        m1f, m2f = material_frame(d1f, tf_old, tf, theta_f)
        eps0 = self.get_epsilon(n0, n1, l_ke)
        eps1 = self.get_epsilon(n1, n2, l_kf)
        kappa1, kappa2 = self.get_kappa(n0, n1, n2, m1e, m2e, m1f, m2f)
        tau = self.get_tau(theta_e, theta_f, beta)
        return jnp.array([eps0, eps1, kappa1, kappa2, tau])


class Triplet2D(Stencil[None]):
    """Discrete diferential geometry strain triplet in 2D."""

    l_k: jax.Array  # [l_ke, l_kf]

    def get_strain(self, q: jax.Array, aux: None) -> jax.Array:
        l_ke, l_kf = self.l_k
        n0 = q[0:2]
        n1 = q[2:4]
        n2 = q[4:6]
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
        return kb
