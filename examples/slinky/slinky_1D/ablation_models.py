"""Construct energy / stiffness models from ``CaseConfig``."""

import jax
import equinox as eqx

from Energy_NN import MLP_Energy, ICNN_Energy, MLP_Stiffness, Signed_MLP_Stiffness

from ablation_config import CaseConfig


def make_model(case: CaseConfig, key: jax.Array) -> eqx.Module:
    kwargs = dict(
        key=key,
        K_initial=case.K_initial,
        which_case=case.which_case,
        weight_scale=case.weight_scale,
    )

    if case.family == "energy_mlp":
        if case.which_case == "baseline":
            return MLP_Energy(hidden_sizes=(10,), **kwargs)
        return MLP_Energy(hidden_sizes=case.hidden_sizes, **kwargs)

    if case.family == "energy_icnn":
        if case.which_case == "baseline":
            return ICNN_Energy(hidden_sizes=(10,), **kwargs)
        return ICNN_Energy(hidden_sizes=case.hidden_sizes, **kwargs)

    if case.family == "stiffness_mlp":
        if case.which_case == "only_baseline":
            return MLP_Stiffness(hidden_sizes=(10,), **kwargs)
        return MLP_Stiffness(hidden_sizes=case.hidden_sizes, **kwargs)

    if case.family == "signed_stiffness_mlp":
        return Signed_MLP_Stiffness(hidden_sizes=case.hidden_sizes, **kwargs)

    raise ValueError(f"Unknown family: {case.family}")
