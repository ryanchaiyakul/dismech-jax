import jax
import jax.numpy as jnp

from ..states import TripletState
from ..util import get_ref_twist, material_frame, parallel_transport
from .stencil import Stencil


class Triplet(Stencil[TripletState]):
    """Discrete diferential geometry strain triplet.

    Strains are densities `[eps_e, eps_f, kappa1/l, kappa2/l, tau/l]` with the
    Voronoi length `l = mean(l_k)` as the measure, so the energy
    `l * model(del_strain)` recovers `EA eps^2 l / 2` and `EI kappa^2 / (2 l)`.

    The reference twist is recomputed from the current tangents (the aux `beta`
    only seeds it), so the energy matches `TripletState.update` at any `q` and an
    equilibrium stays one after the aux is updated, whatever the step size.
    """

    l_k: jax.Array  # [l_ke, l_kf]

    def get_measure(self) -> jax.Array:
        return jnp.mean(self.l_k)

    def get_strain(self, q: jax.Array, aux: TripletState) -> jax.Array:
        te_old, tf_old = aux.t
        d1e, d1f = aux.d1
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
        beta = get_ref_twist(
            parallel_transport(d1e, te_old, te),
            parallel_transport(d1f, tf_old, tf),
            te,
            tf,
            aux.beta,
        )
        tau = self.get_tau(theta_e, theta_f, beta)
        l_v = self.get_measure()
        return jnp.array([eps0, eps1, kappa1 / l_v, kappa2 / l_v, tau / l_v])
