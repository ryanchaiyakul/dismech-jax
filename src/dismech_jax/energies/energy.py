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

    def H_blocks(self, q: jax.Array, aux: AuxT, b: int) -> tuple[jax.Array, jax.Array]:
        """Hessian as block tridiagonal `(D, L)` with block size `b`: diagonal
        blocks `(nb, b, b)` and sub-diagonal blocks `(nb - 1, b, b)`, where
        `nb = ceil(len(q) / b)` and the padding DOFs are zero.

        Only terms whose Hessian is block tridiagonal can implement this."""
        raise NotImplementedError(
            f"{type(self).__name__} has no block tridiagonal Hessian; "
            "use `block_size=None` for dense solves."
        )

    def update(self, aux: AuxT, q: jax.Array) -> AuxT:
        return aux
