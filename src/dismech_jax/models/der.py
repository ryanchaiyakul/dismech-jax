from typing import Self

import equinox as eqx
import jax
import jax.numpy as jnp

from ..legacy import Geometry, Material


class DER(eqx.Module):
    K: jax.Array  # [EA1, EA2, EI1, EI2, GJ]

    @classmethod
    def from_legacy(cls, geom: Geometry, material: Material) -> Self:
        A = geom.axs if geom.axs else jnp.pi * geom.r0**2
        EA = material.youngs_rod * A

        if geom.ixs1 and geom.ixs2:
            EI1 = material.youngs_rod * geom.ixs1
            EI2 = material.youngs_rod * geom.ixs2
        else:
            EI1 = EI2 = material.youngs_rod * jnp.pi * geom.r0**4 / 4

        J = geom.jxs if geom.jxs else jnp.pi * geom.r0**4 / 2
        GJ = material.youngs_rod / (2 * (1 + material.poisson_rod)) * J

        return cls(jnp.array([EA, EA, EI1, EI2, GJ]))

    def __call__(self, del_strain: jax.Array) -> jax.Array:
        return 0.5 * jnp.sum(self.K * del_strain**2)
