import dismech_jax
import pathlib
import pytest

import sys
import os

# Add the 'src' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))


def rel_path(fname: str) -> pathlib.Path:
    """
    Localizes path to module path
    """
    return pathlib.Path(__file__).parent / fname


# rod cantilever


@pytest.fixture
def rod_cantilever_n21() -> dismech_jax.Geometry:
    return dismech_jax.Geometry.from_txt(
        rel_path("resources/rod_cantilever/horizontal_rod_n21.txt")
    )


@pytest.fixture
def rod_cantilever_n26() -> dismech_jax.Geometry:
    return dismech_jax.Geometry.from_txt(
        rel_path("resources/rod_cantilever/horizontal_rod_n26.txt")
    )


@pytest.fixture
def rod_cantilever_n51() -> dismech_jax.Geometry:
    return dismech_jax.Geometry.from_txt(
        rel_path("resources/rod_cantilever/horizontal_rod_n51.txt")
    )


@pytest.fixture
def triplet_rod_cantilever_n51(
    rod_cantilever_n51: dismech_jax.Geometry,
) -> dismech_jax.Triplets:
    connectivity = dismech_jax.Connectivity.from_geo(rod_cantilever_n51)
    state = dismech_jax.State.from_geo(rod_cantilever_n51, connectivity)
    return dismech_jax.Triplets.from_geo(rod_cantilever_n51, state)


@pytest.fixture
def rod_cantilever_n101() -> dismech_jax.Geometry:
    return dismech_jax.Geometry.from_txt(
        rel_path("resources/rod_cantilever/horizontal_rod_n101.txt")
    )


@pytest.fixture
def hexparachute_n6() -> dismech_jax.Geometry:
    return dismech_jax.Geometry.from_txt(
        rel_path("resources/parachute/hexparachute_n6_python.txt")
    )
