from abc import abstractmethod
from typing import Self

import jax
import equinox as eqx


class Aux(eqx.Module):
    @abstractmethod
    def update(self, q: jax.Array) -> Self: ...
