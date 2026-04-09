"""Force, energy, and stiffness predictions for trained models."""

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from slinky_1d_system import Slinky1D


def predict_force(slinky: Slinky1D, q0: jax.Array, model: eqx.Module, disp_vals: jax.Array) -> jax.Array:
    def one_force(disp):
        q = slinky.get_q(disp, q0)
        return slinky.get_reaction_force(disp, q, model, None)

    return jax.vmap(one_force)(disp_vals)


def predict_energy(model: eqx.Module, strains: jax.Array) -> jax.Array:
    return jax.vmap(lambda eps: model(jnp.array([eps])))(strains)


def predict_effective_stiffness(model: eqx.Module, strains: jax.Array) -> jax.Array:
    def scalar_energy(e):
        return model(jnp.array([e]))

    d2e = jax.grad(jax.grad(scalar_energy))
    return jax.vmap(d2e)(strains)


def summary_sharpness(stiffness_vals) -> float:
    s = np.asarray(stiffness_vals)
    if len(s) < 2:
        return 0.0
    ds = np.diff(s)
    return float(np.mean(np.abs(ds)))
