from dataclasses import dataclass


@dataclass
class Geometry:
    length: float
    r0: float
    axs: float | None = None
    jxs: float | None = None
    ixs1: float | None = None
    ixs2: float | None = None


@dataclass
class Material:
    density: float
    youngs_rod: float = 0.0
    poisson_rod: float = 0.0
