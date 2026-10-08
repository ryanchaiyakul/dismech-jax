"""Step acceptance: `accept(sys, x, z, aux) -> bool`.

The line search only takes trial states that are accepted. If none is, Newton
stops early; with `passes > 1` the solver then refreshes the aux at the current
state and continues (see `solver.solve`)."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from .system import Aux, System


class Accept(eqx.Module):
    @abstractmethod
    def __call__(
        self, sys: System, x: jax.Array, z: jax.Array, aux: Aux
    ) -> jax.Array: ...


class AcceptAll(Accept):
    def __call__(self, sys, x, z, aux):
        return jnp.array(True)


class MaxTurn(Accept):
    """Reject states where an edge tangent turned by more than `acos(min_cos)`
    from its reference tangent in the aux, for rods built by `make_rod` (DOFs
    `[x0, y0, z0, theta0, x1, ...]`, `TripletState` aux of term `term`).

    The reference frames are parallel transported from these tangents, which is
    singular at a 180 degree turn. In a snap-through one solve can turn an edge
    that far; with `passes > 1` the turn is instead followed in increments of at
    most `acos(min_cos)`, refreshing the frames in between."""

    min_cos: float = 0.5  # 60 degrees
    term: int = 0

    def __call__(self, sys, x, z, aux):
        t = aux[self.term].t  # (n_triplets, 2, 3): [te, tf] per triplet
        t_ref = jnp.concatenate([t[:, 0], t[-1:, 1]])
        q = sys.join(x, z)[: 4 * t.shape[0] + 7]  # the rod's DOFs (N = n_triplets + 2)
        e = jnp.diff(jnp.stack([q[0::4], q[1::4], q[2::4]], axis=1), axis=0)
        cos = jnp.sum(e * t_ref, axis=1) / jnp.linalg.norm(e, axis=1)
        return jnp.min(cos) > self.min_cos
