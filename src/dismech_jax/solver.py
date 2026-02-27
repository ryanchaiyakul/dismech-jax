import jax
import jax.numpy as jnp
import equinox as eqx

from .states import State
from .systems import System


def compute_ift_gradient(
    q_star: jax.Array, grad_obj: jax.Array, model: eqx.Module, aux: State, sys: System
) -> eqx.Module:
    H = sys.get_H(q_star, model, aux)
    H_reg = H.at[jnp.diag_indices(H.shape[0])].add(1e-8)
    v = jnp.linalg.solve(H_reg, grad_obj)
    _, vjp_fn = jax.vjp(lambda _m: sys.get_F(q_star, _m, aux), model)
    (grads,) = vjp_fn(v)
    return grads


@eqx.filter_custom_vjp
@eqx.filter_jit
def solve_step(
    model: eqx.Module,
    _lambda: jax.Array,
    q0: jax.Array,
    aux: State,
    sys: System,
    iters: int = 10,
    ls_steps: int = 10,
    c1: float = 1e-4,
) -> jax.Array:
    alphas = 0.5 ** jnp.arange(ls_steps)

    def newton_step(carry, _):
        q, e_old, res = carry

        # TODO: use SOCU for blockdiagonal
        H = sys.get_H(q, model, aux)
        H_reg = H.at[jnp.diag_indices(H.shape[0])].add(1e-8)
        delta_q = jnp.linalg.solve(H_reg, res)
        slope = -jnp.dot(res, delta_q)

        # Parallel line search
        test_qs = q + alphas[:, None] * delta_q
        test_energies = jax.vmap(lambda _q: sys.get_E(_q, model, aux))(test_qs)

        # If Armijo fails, take the smallest possible step
        is_good = test_energies <= e_old + c1 * alphas * slope  # Armijo Condition
        safe_idx = jnp.where(jnp.any(is_good), jnp.argmax(is_good), ls_steps - 1)

        next_q = test_qs[safe_idx]
        next_e = test_energies[safe_idx]
        next_res = sys.get_F(next_q, model, aux)

        return (next_q, next_e, next_res), jnp.linalg.norm(next_res)

    q_init = sys.get_q(_lambda, q0)
    init_e = sys.get_E(q_init, model, aux)
    init_res = sys.get_F(q_init, model, aux)
    (final_q, _, _), _ = jax.lax.scan(
        newton_step, (q_init, init_e, init_res), None, iters
    )
    return final_q


@solve_step.def_fwd
def solve_step_fwd(
    perturbed: eqx.Module,
    model: eqx.Module,
    _lambda: jax.Array,
    q0: jax.Array,
    aux: State,
    sys: System,
    iters: int = 10,
    ls_steps: int = 10,
    c1: float = 1e-4,
) -> tuple[jax.Array, jax.Array]:
    final_q = solve_step(model, _lambda, q0, aux, sys, iters, ls_steps, c1)
    return final_q, final_q


@solve_step.def_bwd
def solve_step_bwd(
    res: jax.Array,
    grad_obj: jax.Array,
    perturbed: eqx.Module,
    model: eqx.Module,
    _lambda: jax.Array,
    q0: jax.Array,
    aux: State,
    sys: System,
    iters: int = 10,
    ls_steps: int = 10,
    c1: float = 1e-4,
) -> eqx.Module:
    return compute_ift_gradient(res, grad_obj, model, aux, sys)


@eqx.filter_custom_vjp
@eqx.filter_jit
def solve(
    model: eqx.Module,
    lambdas: jax.Array,
    q0: jax.Array,
    aux: State,
    sys: System,
    iters: int = 10,
    ls_steps: int = 10,
    c1: float = 1e-4,
    max_dt: float = 1e-1,
) -> jax.Array:
    def scan_fn(res: tuple[jax.Array, State, jax.Array], target_lambda: jax.Array):
        _q, _aux, _current_lambda = res

        def cond_fn(val: tuple[jax.Array, State, jax.Array]):
            _, _, curr_L = val
            return curr_L < target_lambda

        def body_fn(carry: tuple[jax.Array, State, jax.Array]):
            q, aux, curr_L = carry
            next_L = jnp.minimum(curr_L + max_dt, target_lambda)
            new_q = solve_step(model, next_L, q, aux, sys, iters, ls_steps, c1)
            new_aux = jax.vmap(lambda a: a.update(new_q))(aux)
            return new_q, new_aux, next_L

        final_q, final_aux, final_L = jax.lax.while_loop(
            cond_fn, body_fn, (_q, _aux, _current_lambda)
        )
        return (final_q, final_aux, final_L), final_q

    _, qs = jax.lax.scan(
        scan_fn, (q0, aux, jnp.asarray(lambdas[0], dtype=lambdas.dtype)), lambdas
    )
    return qs


@solve.def_fwd
def solve_fwd(
    perturbed: eqx.Module,
    model: eqx.Module,
    lambdas: jax.Array,
    q0: jax.Array,
    aux: State,
    sys: System,
    iters: int = 10,
    ls_steps: int = 10,
    c1: float = 1e-4,
    max_dt: float = 1e-1,
) -> tuple[jax.Array, tuple[jax.Array, State]]:
    def scan_fwd_fn(res: tuple[jax.Array, State, jax.Array], target_lambda: jax.Array):
        _q, _aux, _current_lambda = res

        def cond_fn(val):
            _, _, curr_L = val
            return curr_L < target_lambda

        def body_fn(val):
            q, aux, curr_L = val
            next_L = jnp.minimum(curr_L + max_dt, target_lambda)

            new_q = solve_step(model, next_L, q, aux, sys, iters, ls_steps, c1)
            new_aux = jax.vmap(lambda a: a.update(new_q))(aux)

            return new_q, new_aux, next_L

        final_q, final_aux, final_L = jax.lax.while_loop(
            cond_fn, body_fn, (_q, _aux, _current_lambda)
        )
        return (final_q, final_aux, final_L), (final_q, _aux)

    _, (qs, auxs) = jax.lax.scan(
        scan_fwd_fn, (q0, aux, jnp.asarray(lambdas[0], dtype=lambdas.dtype)), lambdas
    )
    return qs, (qs, auxs)


@solve.def_bwd
def solve_bwd(
    res: tuple[jax.Array, State],
    grad_obj: jax.Array,
    perturbed: eqx.Module,
    model: eqx.Module,
    lambdas: jax.Array,
    q0: jax.Array,
    aux: State,
    sys: System,
    iters: int = 10,
    ls_steps: int = 10,
    c1: float = 1e-4,
    max_dt: float = 1e9,
) -> eqx.Module:
    qs, auxs = res
    batched_ift_fn = jax.vmap(compute_ift_gradient, in_axes=(0, 0, None, 0, None))
    batched_grads = batched_ift_fn(qs, grad_obj, model, auxs, sys)
    total_grad = jax.tree.map(lambda x: jnp.sum(x, axis=0), batched_grads)
    return total_grad
