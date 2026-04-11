from dataclasses import dataclass

import jax

@dataclass
class Properties:
    length: float | None = None
    r0: float = 0.005
    axs: float | None = None
    jxs: float | None = None
    ixs1: float | None = None
    ixs2: float | None = None
    density: float = 1000.0
    E: float = 1e6
    N: int = 20
    start: jax.Array | None = None
    end: jax.Array | None = None
