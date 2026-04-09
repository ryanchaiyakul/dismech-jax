"""Case definitions for slinky 1D ablations."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class CaseConfig:
    name: str
    family: str
    hidden_sizes: tuple
    which_case: str
    K_initial: float = 0.1
    weight_scale: float = 1.0


def build_case_list() -> List[CaseConfig]:
    """Full ablation grid (experiment-data driver)."""
    return [
        CaseConfig("energy_baseline", "energy_mlp", (), "baseline"),
        CaseConfig("energy_mlp_L1", "energy_mlp", (10,), "mlp"),
        CaseConfig("energy_mlp_L2", "energy_mlp", (10, 10), "mlp"),
        CaseConfig("energy_baseline_plus_mlp_L1", "energy_mlp", (10,), "combined"),
        CaseConfig("energy_baseline_plus_mlp_L2", "energy_mlp", (10, 10), "combined"),
        CaseConfig("energy_icnn_L1", "energy_icnn", (10,), "icnn"),
        CaseConfig("energy_icnn_L2", "energy_icnn", (10, 10), "icnn"),
        CaseConfig("energy_baseline_plus_icnn_L1", "energy_icnn", (10,), "combined"),
        CaseConfig("energy_baseline_plus_icnn_L2", "energy_icnn", (10, 10), "combined"),
        CaseConfig("stiffness_baseline", "stiffness_mlp", (), "only_baseline"),
        CaseConfig("stiffness_only_mlp_L1", "stiffness_mlp", (10,), "only_mlp"),
        CaseConfig("stiffness_only_mlp_L2", "stiffness_mlp", (10, 10), "only_mlp"),
        CaseConfig("stiffness_baseline_plus_mlp_L1", "stiffness_mlp", (10,), "combined"),
        CaseConfig("stiffness_baseline_plus_mlp_L2", "stiffness_mlp", (10, 10), "combined"),
        # CaseConfig("signed_stiffness_only_mlp_L1", "signed_stiffness_mlp", (10,), "only_mlp"),
        # CaseConfig("signed_stiffness_only_mlp_L2", "signed_stiffness_mlp", (10, 10), "only_mlp"),
        CaseConfig("stiffness_baseline_plus_signed_mlp_L1", "signed_stiffness_mlp", (10,), "combined"),
        CaseConfig("stiffness_baseline_plus_signed_mlp_L2", "signed_stiffness_mlp", (10, 10), "combined"),
    ]


def init_sensitivity_cases() -> List[CaseConfig]:
    """Small set of cases for multi-seed initialization sensitivity."""
    return [
        CaseConfig("energy_icnn_L2", "energy_icnn", (10, 10), "icnn"),
        CaseConfig("stiffness_baseline_plus_mlp_L2", "stiffness_mlp", (10, 10), "combined"),
    ]
