from typing import Self

import jax
import jax.numpy as jnp

from ..legacy import Geometry, Material
from .der import DER


class Sano(DER):
    """Ribbon model with Sano's regularized bend-twist coupling.

    A narrow ribbon has a hard (1) and an easy (2) bending axis whose easy-axis
    bending and twist are coupled (Sadowsky). Sano regularizes the coupling with a
    length `zeta` so it stays bounded as the easy curvature vanishes. With the
    integrated strains `k2 = l * kappa2`, `t = l * tau` and Voronoi length `l`,

        E = DER + 1/2 EI2 / l * t^4 / ((l / zeta)^2 + k2^2).

    In the density strains used by `Triplet` (`kappa2 = k2 / l`, `tau = t / l`)
    the `l` factors cancel, so the density is independent of the mesh:

        e = DER + 1/2 EI2 * tau^4 / (1 / zeta^2 + kappa2^2).

    `zeta -> 0` recovers the Kirchhoff rod (`inv_zeta_sq -> inf`). The coupling is
    non-convex in `kappa2` where `3 kappa2^2 < 1 / zeta^2`, so Newton steps may need
    a line search. Stretch is unchanged from `DER`.
    """

    inv_zeta_sq: jax.Array  # 1 / zeta^2

    @classmethod
    def from_legacy(  # type: ignore[override]
        cls, geom: Geometry, material: Material, zeta: float
    ) -> Self:
        der = DER.from_legacy(geom, material)
        return cls(K=der.K, inv_zeta_sq=jnp.asarray(1.0 / zeta**2))

    def __call__(self, del_strain: jax.Array) -> jax.Array:
        kappa2, tau = del_strain[3], del_strain[4]
        coupling = tau**4 / (self.inv_zeta_sq + kappa2**2)
        return super().__call__(del_strain) + 0.5 * self.K[3] * coupling
