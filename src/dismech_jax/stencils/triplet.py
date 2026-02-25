import jax
import jax.numpy as jnp

from .stencil import Stencil
from ..states import TripletState
from ..util import material_frame


class Triplet(Stencil):
    """Discrete diferential geometry strain triplet."""

    l_k: jax.Array  # [l_ke, l_kf]

    def get_strain(self, q: jax.Array, aux: TripletState | None = None) -> jax.Array:
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
