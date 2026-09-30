from abc import abstractmethod
from typing import Self

import equinox as eqx
import jax


class State(eqx.Module):
    @abstractmethod
    def update(self, q: jax.Array) -> Self: ...
