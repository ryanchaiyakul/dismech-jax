import dismech_jax

geo = dismech_jax.Geometry.from_txt(
    "tests/resources/rod_cantilever/horizontal_rod_n21.txt"
)
top = dismech_jax.Connectivity.from_geo(geo)
state = dismech_jax.State.from_geo(geo, top)
print(state)