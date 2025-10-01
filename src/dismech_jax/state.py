from __future__ import annotations
from dataclasses import dataclass

from jax.tree_util import register_dataclass
import jax


@register_dataclass
@dataclass
class State:
    q: jax.Array  # [nodes | edges]
