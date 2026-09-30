from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
import jax


class Energy[AuxT](eqx.Module):
    @abstractmethod
    def E(self, q: jax.Array, aux: AuxT) -> jax.Array: ...

    def F(self, q: jax.Array, aux: AuxT) -> jax.Array:
        return jax.grad(self.E)(q, aux)

    def H(self, q: jax.Array, aux: AuxT) -> jax.Array:
        return jax.hessian(self.E)(q, aux)

    def update(self, aux: AuxT, q: jax.Array) -> AuxT:
        return aux
