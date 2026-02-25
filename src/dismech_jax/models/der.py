import equinox as eqx
import jax
import jax.numpy as jnp

from ..params import Geometry, Material


class DER(eqx.Module):
    K: jax.Array  # [EA1, EA2, EI1, EI2, GJ]

    @classmethod
    def from_legacy(cls, l_k: jax.Array, geom: Geometry, material: Material):
        A = geom.axs if geom.axs else jnp.pi * geom.r0**2
        EA = material.youngs_rod * A

        if geom.ixs1 and geom.ixs2:
            EI1 = material.youngs_rod * geom.ixs1
            EI2 = material.youngs_rod * geom.ixs2
        else:
            EI1 = EI2 = material.youngs_rod * jnp.pi * geom.r0**4 / 4

        # TODO: what is proper name
        something = geom.jxs if geom.jxs else jnp.pi * geom.r0**4 / 2
        GJ = material.youngs_rod / (2 * (1 + material.poisson_rod)) * something

        # Rescale
        EA *= l_k
        EI1 /= l_k
        EI2 /= l_k
        GJ /= l_k

        return cls(jnp.array([EA, EA, EI1, EI2, GJ]))

    def __call__(self, del_strain: jax.Array) -> jax.Array:
        return jnp.sum(self.K * del_strain**2)
