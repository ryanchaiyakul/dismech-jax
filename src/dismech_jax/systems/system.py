from abc import abstractmethod
from typing import Generic, TypeVar

import jax
import equinox as eqx

from ..states import State

AuxT = TypeVar("AuxT", bound=State)


class System(eqx.Module, Generic[AuxT]):
    @abstractmethod
    def get_q(self, _lambda: jax.Array, q0: jax.Array) -> jax.Array: ...
    @abstractmethod
    def get_F(self, q: jax.Array, model: eqx.Module, aux: AuxT) -> jax.Array: ...
    @abstractmethod
    def get_H(self, q: jax.Array, model: eqx.Module, aux: AuxT) -> jax.Array: ...
