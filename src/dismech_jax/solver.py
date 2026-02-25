import jax
import jax.numpy as jnp
import equinox as eqx

from .aux import Aux
from .systems import System


def compute_ift_gradient(
    q_star: jax.Array, grad_obj: jax.Array, model: eqx.Module, aux: Aux, sys: System
) -> eqx.Module:
    H = sys.get_H(q_star, model, aux)
    v = jnp.linalg.solve(H + 1e-8 * jnp.eye(H.shape[0]), grad_obj)
    _, vjp_fn = jax.vjp(lambda _m: sys.get_F(q_star, _m, aux), model)
    (grads,) = vjp_fn(v)
    return grads


@eqx.filter_custom_vjp
@eqx.filter_jit
def solve_step(
    model: eqx.Module,
    _lambda: jax.Array,
    q0: jax.Array,
    aux: Aux,
    sys: System,
    iters: int = 10,
) -> jax.Array:
    q_init = sys.get_q(_lambda, q0)

    def scan_fn(q: jax.Array, _: jax.Array) -> tuple[jax.Array, jax.Array]:
        H = sys.get_H(q, model, aux)
        res = sys.get_F(q, model, aux)
        delta_q = jnp.linalg.solve(H + 1e-8 * jnp.eye(q.shape[0]), res)
        return q + delta_q, jnp.linalg.norm(res)

    final_q, res_history = jax.lax.scan(scan_fn, q_init, None, iters)
    return final_q


@solve_step.def_fwd
def solve_step_fwd(
    perturbed: eqx.Module,
    model: eqx.Module,
    _lambda: jax.Array,
    q0: jax.Array,
    aux: Aux,
    sys: System,
    iters: int = 10,
) -> tuple[jax.Array, jax.Array]:
    final_q = solve_step(model, _lambda, q0, aux, sys, iters)
    return final_q, final_q


@solve_step.def_bwd
def solve_step_bwd(
    res: jax.Array,
    grad_obj: jax.Array,
    perturbed: eqx.Module,
    model: eqx.Module,
    _lambda: jax.Array,
    q0: jax.Array,
    aux: Aux,
    sys: System,
    iters: int = 10,
) -> eqx.Module:
    return compute_ift_gradient(res, grad_obj, model, aux, sys)


@eqx.filter_custom_vjp
@eqx.filter_jit
def solve(
    model: eqx.Module,
    lambdas: jax.Array,
    q0: jax.Array,
    aux: Aux,
    sys: System,
    iters: int = 10,
) -> jax.Array:
    def scan_fn(res: tuple[jax.Array, Aux], _lambda: jax.Array):
        _q, _aux = res
        new_q = solve_step(model, _lambda, _q, _aux, sys, iters)
        new_aux = jax.vmap(lambda a: a.update(new_q))(_aux)
        return (new_q, new_aux), new_q

    _, qs = jax.lax.scan(scan_fn, (q0, aux), lambdas)
    return qs


@solve.def_fwd
def solve_fwd(
    perturbed: eqx.Module,
    model: eqx.Module,
    lambdas: jax.Array,
    q0: jax.Array,
    aux: Aux,
    sys: System,
    iters: int = 10,
) -> tuple[jax.Array, tuple[jax.Array, Aux]]:
    def scan_fwd_fn(res: tuple[jax.Array, Aux], _lambda: jax.Array):
        _q, _aux = res
        new_q = solve_step(model, _lambda, _q, _aux, sys, iters)
        new_aux = jax.vmap(lambda a: a.update(new_q))(_aux)
        return (new_q, new_aux), (new_q, _aux)

    _, (qs, auxs) = jax.lax.scan(scan_fwd_fn, (q0, aux), lambdas)
    return qs, (qs, auxs)


@solve.def_bwd
def solve_bwd(
    res: jax.Array,
    grad_obj: jax.Array,
    perturbed: eqx.Module,
    model: eqx.Module,
    lambdas: jax.Array,
    q0: jax.Array,
    aux: Aux,
    sys,
    iters: int = 10,
) -> eqx.Module:
    qs, auxs = res
    batched_ift_fn = jax.vmap(compute_ift_gradient, in_axes=(0, 0, None, 0, None))
    batched_grads = batched_ift_fn(qs, grad_obj, model, auxs, sys)
    total_grad = jax.tree.map(lambda x: jnp.sum(x, axis=0), batched_grads)
    return total_grad
