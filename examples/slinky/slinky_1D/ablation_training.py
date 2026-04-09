"""Shared training loop for slinky 1D ablations (with gradient clipping)."""

from dataclasses import asdict
from typing import Any, Dict, Tuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from ablation_config import CaseConfig
from ablation_models import make_model
from ablation_predict import (
    predict_effective_stiffness,
    predict_energy,
    predict_force,
    summary_sharpness,
)


def train_one_case(
    case: CaseConfig,
    problem: Dict[str, Any],
    seed: int = 42,
    lr: float = 1e-3,
    num_epochs: int = 10000,
    log_freq: int = 500,
    schedule_boundary: int = 7500,
    schedule_scale: float = 0.1,
    gradient_clip_norm: float = 1.0,
    full_metrics: bool = True,
) -> Tuple[eqx.Module, Dict[str, Any]]:
    """
    Train a single case on force matching. Uses Adam with optional piecewise
    LR drop and global-norm gradient clipping (default ``gradient_clip_norm=1.0``).

    If ``full_metrics`` is False, skip energy / stiffness curves and sharpness
    (faster for multi-seed sweeps).
    """
    slinky = problem["slinky"]
    q0 = problem["q0"]
    disps = problem["disps"]
    force_truth = problem["force_truth"]
    train_mask = problem["train_mask"]
    test_mask = problem["test_mask"]

    train_disps = disps[train_mask]
    test_disps = disps[test_mask]
    train_force_truth = force_truth[train_mask]
    test_force_truth = force_truth[test_mask]

    model = make_model(case, jax.random.PRNGKey(seed))

    def train_loss(m):
        pred_force = predict_force(slinky, q0, m, train_disps)
        return jnp.mean((train_force_truth - pred_force) ** 2)

    def test_loss(m):
        pred_force = predict_force(slinky, q0, m, test_disps)
        return jnp.mean((test_force_truth - pred_force) ** 2)

    schedule = optax.piecewise_constant_schedule(
        init_value=lr,
        boundaries_and_scales={schedule_boundary: schedule_scale},
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(gradient_clip_norm),
        optax.adam(learning_rate=schedule),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(carry, _):
        m, ost = carry
        loss_val, grads = eqx.filter_value_and_grad(train_loss)(m)
        updates, ost = optimizer.update(grads, ost, m)
        m = eqx.apply_updates(m, updates)
        return (m, ost), loss_val

    def train_loop(m, ost):
        def scan_fn(carry, i):
            next_carry, train_loss_val = train_step(carry, None)
            m_next, _ = next_carry
            test_loss_val = test_loss(m_next)

            def log_loss(_):
                jax.debug.print(
                    "[{case}] seed={seed} epoch={epoch} train={train} test={test}",
                    case=case.name,
                    seed=seed,
                    epoch=i,
                    train=train_loss_val,
                    test=test_loss_val,
                )

            jax.lax.cond(i % log_freq == 0, log_loss, lambda _: None, operand=None)
            return next_carry, (train_loss_val, test_loss_val)

        (final_model, _), loss_history = jax.lax.scan(
            scan_fn,
            (m, ost),
            jnp.arange(num_epochs + 1),
        )
        return final_model, loss_history

    final_model, loss_history = train_loop(model, opt_state)

    train_hist_arr, test_hist_arr = loss_history
    train_hist = np.asarray(train_hist_arr)
    test_hist = np.asarray(test_hist_arr)

    pred_full_force = predict_force(slinky, q0, final_model, disps)
    train_mse = float(jnp.mean((force_truth[train_mask] - pred_full_force[train_mask]) ** 2))
    test_mse = float(jnp.mean((force_truth[test_mask] - pred_full_force[test_mask]) ** 2))

    base: Dict[str, Any] = {
        "case": asdict(case),
        "seed": seed,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "pulled_node_x": np.asarray(problem["pulled_node_x"]),
        "force_truth": np.asarray(force_truth),
        "pred_force": np.asarray(pred_full_force),
        "train_mask": np.asarray(train_mask),
        "test_mask": np.asarray(test_mask),
        "train_hist": train_hist,
        "test_hist": test_hist,
    }

    if full_metrics:
        strains = problem["strains"]
        pred_energy = predict_energy(final_model, strains)
        pred_stiffness = predict_effective_stiffness(final_model, strains)
        gap = float(test_mse - train_mse)
        sharpness = summary_sharpness(np.asarray(pred_stiffness))
        base.update(
            {
                "generalization_gap": gap,
                "stiffness_sharpness": sharpness,
                "disps": np.asarray(disps),
                "strains": np.asarray(strains),
                "pred_energy": np.asarray(pred_energy),
                "pred_stiffness": np.asarray(pred_stiffness),
                "meta": problem["meta"],
            }
        )

    return final_model, base
