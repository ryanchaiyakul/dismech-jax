from __future__ import annotations
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass
from jax.typing import ArrayLike

from .connectivity import Connectivity
from .state import State
from .triplet import Triplets


@register_dataclass
@dataclass
class SimParams:
    dt: ArrayLike
    max_iter: ArrayLike
    tol: ArrayLike
    free_dof: jax.Array


@register_dataclass
@dataclass
class Sim:
    ts: Triplets
    top: Connectivity

    def step(self, params: SimParams, state: State) -> State:
        def cond_fn(carry: tuple[State, ArrayLike, ArrayLike, ArrayLike]) -> bool:
            _, _, err, iter_count = carry
            return (err > params.tol) & (iter_count < params.max_iter)  # type: ignore

        def body_fn(
            carry: tuple[State, ArrayLike, ArrayLike, ArrayLike],
        ) -> tuple[State, ArrayLike, ArrayLike, ArrayLike]:
            state, alpha, _, iter_count = carry

            f_free = self.ts.get_grad_energy(state)[params.free_dof]
            j_free = self.ts.get_hess_energy(state)[
                jnp.ix_(params.free_dof, params.free_dof)
            ]
            dq = -jnp.linalg.solve(j_free, f_free)
            dq *= self.get_alpha(alpha)
            new_state = state.update(
                jnp.zeros_like(state.q).at[params.free_dof].set(dq), params.dt, self.top
            )
            err = jnp.linalg.norm(f_free)
            jax.debug.print("f_free: {}", f_free)
            return (new_state, alpha, err, iter_count + 1)

        # Initial values
        f_free_init = self.ts.get_grad_energy(state)[params.free_dof]
        err_init = jnp.linalg.norm(f_free_init)
        init_carry = (state, 1.0, err_init, 0)

        final_state, final_alpha, final_err, final_iter = jax.lax.while_loop(
            cond_fn, body_fn, init_carry
        )

        return final_state

    def get_alpha(self, alpha: ArrayLike) -> ArrayLike:
        return jnp.maximum(alpha * 0.9, 0.1)
