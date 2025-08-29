import jax
import jax.numpy as jnp
import dismech_jax


mesh = dismech_jax.Mesh.from_txt(
    "tests/resources/rod_cantilever/horizontal_rod_n51.txt"
)
print(mesh._nodes)
"""
geo = dismech_jax.Geometry.from_txt(
    "tests/resources/rod_cantilever/horizontal_rod_n51.txt"
)
top = dismech_jax.Connectivity.from_geo(geo)
state = dismech_jax.State.from_geo(geo, top)
triplets = dismech_jax.Triplets.from_geo(geo, state)
sim = dismech_jax.Sim(triplets, top)
params = dismech_jax.SimParams(1e-2, 25, 1e-4, jnp.arange(3, state.q.shape[0]))
key = jax.random.PRNGKey(0)
dq = jnp.zeros_like(state.q)
#dq = dq.at[3:153].set(jax.random.uniform(key, shape=(153-3,), minval=-1e-6, maxval=1e-6))
new_state = state.update(dq, params.dt, top)
f_free = triplets.get_grad_energy(new_state)[params.free_dof]
j_free = triplets.get_hess_energy(new_state)[jnp.ix_(params.free_dof, params.free_dof)]
print(jnp.linalg.solve(j_free, f_free))
#print(new_state.q)
#print(sim.step(params, new_state).q)
"""
