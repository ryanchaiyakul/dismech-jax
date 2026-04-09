"""Filesystem helpers for ablation outputs."""

import json
import os

import equinox as eqx
import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_case_outputs(outdir: str, model: eqx.Module, result: dict) -> None:
    ensure_dir(outdir)
    case_name = result["case"]["name"]

    np.savez(
        os.path.join(outdir, f"{case_name}.npz"),
        disps=result["disps"],
        pulled_node_x=result["pulled_node_x"],
        force_truth=result["force_truth"],
        pred_force=result["pred_force"],
        strains=result["strains"],
        pred_energy=result["pred_energy"],
        pred_stiffness=result["pred_stiffness"],
        train_mask=result["train_mask"],
        test_mask=result["test_mask"],
        train_hist=result["train_hist"],
        test_hist=result["test_hist"],
        train_mse=result["train_mse"],
        test_mse=result["test_mse"],
        generalization_gap=result["generalization_gap"],
        stiffness_sharpness=result["stiffness_sharpness"],
    )

    eqx.tree_serialise_leaves(os.path.join(outdir, f"{case_name}.eqx"), model)

    summary = {
        "case": result["case"],
        "train_mse": result["train_mse"],
        "test_mse": result["test_mse"],
        "generalization_gap": result["generalization_gap"],
        "stiffness_sharpness": result["stiffness_sharpness"],
        "meta": result["meta"],
    }
    with open(os.path.join(outdir, f"{case_name}.json"), "w") as f:
        json.dump(summary, f, indent=2)
