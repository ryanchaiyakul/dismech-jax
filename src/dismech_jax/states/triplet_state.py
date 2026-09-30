from __future__ import annotations

from typing import Self

import jax
import jax.numpy as jnp

from ..util import get_ref_twist, parallel_transport
from .state import State


class TripletState(State):
    t: jax.Array  # [[te], [tf]]
    d1: jax.Array  # [[d1e], [d1f]]
    beta: jax.Array  # beta

    def update(self, q: jax.Array) -> Self:
        te_old, tf_old = self.t
        d1e_old, d1f_old = self.d1
        beta_old = self.beta
        n0 = q[0:3]
        n1 = q[4:7]
        n2 = q[8:11]
        ee = n1 - n0
        ef = n2 - n1
        te = ee / jnp.linalg.norm(ee)
        tf = ef / jnp.linalg.norm(ef)
        d1e = parallel_transport(d1e_old, te_old, te)
        d1f = parallel_transport(d1f_old, tf_old, tf)
        t_new = jnp.array([te, tf])
        d1_new = jnp.array([d1e, d1f])
        beta_new = get_ref_twist(d1e, d1f, te, tf, beta_old)
        return type(self)(t_new, d1_new, beta_new)
