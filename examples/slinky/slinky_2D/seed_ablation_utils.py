import os
import json
from dataclasses import replace
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from run_architectures import (
    build_architecture_registry,
    run_one_architecture,
)


# =========================================================
# 1) Run seed ablation
# =========================================================
def run_seed_ablation(
    properties,
    train_file: str,
    valid_file: str,
    base_cfg,
    selected_architectures: list[str],
):
    """
    Run multiple seeds for each selected architecture.

    Parameters
    ----------
    properties : any
        Passed into run_one_architecture.
    train_file, valid_file : str
        Dataset files.
    base_cfg : SweepConfig
        Must contain all base settings. Uses base_cfg.seed_list.
    selected_architectures : list[str]
        Architecture names from build_architecture_registry().

    Returns
    -------
    all_results : dict
        all_results[arch_name] = [result_seed0, result_seed1, ...]
    """
    registry = build_architecture_registry()

    unknown = [name for name in selected_architectures if name not in registry]
    if len(unknown) > 0:
        raise ValueError(f"Unknown architecture names: {unknown}")

    if not hasattr(base_cfg, "seed_list"):
        raise AttributeError(
            "base_cfg must have attribute 'seed_list'. "
            "Add seed_list: tuple[int, ...] to SweepConfig."
        )

    all_results = {}

    for arch_name in selected_architectures:
        spec = registry[arch_name]
        arch_results = []

        print("\n" + "=" * 110)
        print(f"SEED ABLATION for architecture: {arch_name}")
        print("=" * 110)

        for seed in base_cfg.seed_list:
            cfg = replace(base_cfg, seed=int(seed))

            print(f"\n--- Running seed {seed} for {arch_name} ---\n")

            result = run_one_architecture(
                properties=properties,
                train_file=train_file,
                valid_file=valid_file,
                spec=spec,
                cfg=cfg,
            )
            arch_results.append(result)

        all_results[arch_name] = arch_results

    return all_results


# =========================================================
# 2) Summaries
# =========================================================
def get_seed_summary(seed_results: list[dict]) -> dict:
    final_train = np.array([r["train_hist"][-1] for r in seed_results], dtype=float)
    final_valid = np.array([r["valid_hist"][-1] for r in seed_results], dtype=float)
    seeds = np.array([r["cfg"].seed for r in seed_results], dtype=int)

    best_idx = int(np.argmin(final_valid))

    return {
        "n_seeds": len(seed_results),
        "seeds": seeds,
        "final_train": final_train,
        "final_valid": final_valid,
        "best_idx": best_idx,
        "best_seed": int(seeds[best_idx]),
        "best_result": seed_results[best_idx],
        "mean_final_train": float(np.mean(final_train)),
        "std_final_train": float(np.std(final_train)),
        "mean_final_valid": float(np.mean(final_valid)),
        "std_final_valid": float(np.std(final_valid)),
        "median_final_valid": float(np.median(final_valid)),
        "min_final_valid": float(np.min(final_valid)),
        "max_final_valid": float(np.max(final_valid)),
    }


def print_seed_summary(all_seed_results: dict):
    print("\n" + "#" * 110)
    print("SEED ABLATION SUMMARY")
    print("#" * 110)

    for arch_name, seed_results in all_seed_results.items():
        s = get_seed_summary(seed_results)

        print(f"\nArchitecture: {arch_name}")
        print(f"  n_seeds           : {s['n_seeds']}")
        print(f"  best_seed         : {s['best_seed']}")
        print(f"  best final valid  : {s['min_final_valid']:.6e}")
        print(f"  worst final valid : {s['max_final_valid']:.6e}")
        print(f"  mean final valid  : {s['mean_final_valid']:.6e}")
        print(f"  std  final valid  : {s['std_final_valid']:.6e}")
        print(f"  median final valid: {s['median_final_valid']:.6e}")
        print(f"  mean final train  : {s['mean_final_train']:.6e}")
        print(f"  std  final train  : {s['std_final_train']:.6e}")


def save_seed_summary_json(all_seed_results: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    payload = {}
    for arch_name, seed_results in all_seed_results.items():
        s = get_seed_summary(seed_results)
        payload[arch_name] = {
            "n_seeds": int(s["n_seeds"]),
            "seeds": [int(x) for x in s["seeds"]],
            "final_train": [float(x) for x in s["final_train"]],
            "final_valid": [float(x) for x in s["final_valid"]],
            "best_idx": int(s["best_idx"]),
            "best_seed": int(s["best_seed"]),
            "mean_final_train": float(s["mean_final_train"]),
            "std_final_train": float(s["std_final_train"]),
            "mean_final_valid": float(s["mean_final_valid"]),
            "std_final_valid": float(s["std_final_valid"]),
            "median_final_valid": float(s["median_final_valid"]),
            "min_final_valid": float(s["min_final_valid"]),
            "max_final_valid": float(s["max_final_valid"]),
        }

    with open(os.path.join(output_dir, "seed_summary.json"), "w") as f:
        json.dump(payload, f, indent=2)


# =========================================================
# 3) Plot helpers
# =========================================================
def plot_seed_loss_envelope(
    seed_results: list[dict],
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = False,
    which: str = "valid",   # "train" or "valid"
):
    if which not in ("train", "valid"):
        raise ValueError("which must be 'train' or 'valid'")

    losses = []
    seeds = []

    for r in seed_results:
        losses.append(r[f"{which}_hist"])
        seeds.append(r["cfg"].seed)

    losses = np.asarray(losses, dtype=float)
    seeds = np.asarray(seeds, dtype=int)

    q05 = np.percentile(losses, 5, axis=0)
    q25 = np.percentile(losses, 25, axis=0)
    q50 = np.percentile(losses, 50, axis=0)
    q75 = np.percentile(losses, 75, axis=0)
    q95 = np.percentile(losses, 95, axis=0)

    final_losses = losses[:, -1]
    best_idx = int(np.argmin(final_losses))
    best_seed = int(seeds[best_idx])

    fig, ax = plt.subplots(figsize=(8.0, 5.4))

    for i in range(losses.shape[0]):
        ax.plot(losses[i], linewidth=1.0, alpha=0.18)

    ax.fill_between(np.arange(losses.shape[1]), q05, q95, alpha=0.18, label="5-95%")
    ax.fill_between(np.arange(losses.shape[1]), q25, q75, alpha=0.28, label="25-75%")

    ax.plot(q50, linestyle="--", linewidth=2.2, label="Median across seeds")
    ax.plot(
        losses[best_idx],
        linewidth=2.5,
        label=f"Best seed = {best_seed} (final {which} = {final_losses[best_idx]:.3e})",
    )

    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(f"{which.capitalize()} MSE loss")
    ax.set_title(title if title is not None else f"{which.capitalize()} loss across seeds")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_seed_prediction_envelope(
    seed_results: list[dict],
    split: str = "valid",
    component: str = "x",
    traj_idx: int = 0,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = False,
    x_idx: int = 4,
    z_idx: int = 6,
):
    """
    Percentile envelope version for prediction curves across seeds.
    """
    if split not in ("train", "valid"):
        raise ValueError("split must be 'train' or 'valid'")
    if component not in ("x", "z"):
        raise ValueError("component must be 'x' or 'z'")

    comp_idx = x_idx if component == "x" else z_idx
    pred_key = f"{split}_pred"
    truth_key = f"{split}_truth"

    seeds = np.array([r["cfg"].seed for r in seed_results], dtype=int)
    preds = np.asarray([r[pred_key][traj_idx, :, comp_idx] for r in seed_results], dtype=float)
    truth = np.asarray(seed_results[0][truth_key][traj_idx, :, comp_idx], dtype=float)

    final_valid = np.asarray([r["valid_hist"][-1] for r in seed_results], dtype=float)
    best_idx = int(np.argmin(final_valid))
    best_seed = int(seeds[best_idx])

    q05 = np.percentile(preds, 5, axis=0)
    q25 = np.percentile(preds, 25, axis=0)
    q50 = np.percentile(preds, 50, axis=0)
    q75 = np.percentile(preds, 75, axis=0)
    q95 = np.percentile(preds, 95, axis=0)

    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    ax.fill_between(np.arange(preds.shape[1]), q05, q95, alpha=0.18, label="5-95%")
    ax.fill_between(np.arange(preds.shape[1]), q25, q75, alpha=0.28, label="25-75%")
    ax.plot(q50, linestyle="--", linewidth=2.0, label="Median across seeds")
    ax.plot(preds[best_idx], linewidth=2.6, label=f"Best seed = {best_seed}")
    ax.plot(truth, color="black", linewidth=2.3, label="Ground truth")

    ax.set_xlabel("lambda index")
    ax.set_ylabel(f"{component}-component position (m)")
    ax.set_title(
        title if title is not None else
        f"{split.capitalize()} envelope across seeds | traj {traj_idx} | {component}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_seed_prediction_all_curves(
    seed_results: list[dict],
    split: str = "valid",
    component: str = "x",
    traj_idx: int = 0,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = False,
    x_idx: int = 4,
    z_idx: int = 6,
):
    """
    All seeds in light alpha, best seed bold, truth in black.
    """
    if split not in ("train", "valid"):
        raise ValueError("split must be 'train' or 'valid'")
    if component not in ("x", "z"):
        raise ValueError("component must be 'x' or 'z'")

    comp_idx = x_idx if component == "x" else z_idx
    pred_key = f"{split}_pred"
    truth_key = f"{split}_truth"

    seeds = np.array([r["cfg"].seed for r in seed_results], dtype=int)
    preds = np.asarray([r[pred_key][traj_idx, :, comp_idx] for r in seed_results], dtype=float)
    truth = np.asarray(seed_results[0][truth_key][traj_idx, :, comp_idx], dtype=float)

    final_valid = np.asarray([r["valid_hist"][-1] for r in seed_results], dtype=float)
    best_idx = int(np.argmin(final_valid))
    best_seed = int(seeds[best_idx])

    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    for i in range(preds.shape[0]):
        ax.plot(preds[i], linewidth=1.0, alpha=0.18)

    ax.plot(
        preds[best_idx],
        linewidth=2.8,
        label=f"Best seed = {best_seed} (test MSE = {final_valid[best_idx]:.3e})",
    )
    ax.plot(truth, color="black", linewidth=2.3, label="Ground truth")

    ax.set_xlabel("lambda index")
    ax.set_ylabel(f"{component}-component position (m)")
    ax.set_title(
        title if title is not None else
        f"{split.capitalize()} all-seed curves | traj {traj_idx} | {component}"
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_seed_final_loss_bar(
    all_seed_results: dict,
    save_path: Optional[str] = None,
    title: str = "Final validation loss across seeds",
    show: bool = False,
):
    arch_names = list(all_seed_results.keys())
    means = []
    stds = []
    bests = []

    for arch_name in arch_names:
        seed_results = all_seed_results[arch_name]
        final_valid = np.asarray([r["valid_hist"][-1] for r in seed_results], dtype=float)
        means.append(np.mean(final_valid))
        stds.append(np.std(final_valid))
        bests.append(np.min(final_valid))

    x = np.arange(len(arch_names))

    fig, ax = plt.subplots(figsize=(max(9.5, 0.8 * len(arch_names)), 5.8))
    ax.bar(x, means, yerr=stds, capsize=4, alpha=0.85, label="Mean ± std")
    ax.scatter(x, bests, marker="x", s=70, label="Best seed")

    ax.set_yscale("log")
    ax.set_ylabel("Final validation loss")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(arch_names, rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# =========================================================
# 4) Make all standard plots
# =========================================================
def make_seed_ablation_plots(
    all_seed_results: dict,
    output_dir: str,
    traj_idx: int = 0,
    x_idx: int = 4,
    z_idx: int = 6,
):
    os.makedirs(output_dir, exist_ok=True)

    for arch_name, seed_results in all_seed_results.items():
        arch_dir = os.path.join(output_dir, arch_name)
        os.makedirs(arch_dir, exist_ok=True)

        plot_seed_loss_envelope(
            seed_results,
            which="train",
            save_path=os.path.join(arch_dir, "train_loss_across_seeds.png"),
            title=f"{arch_name} | train loss across seeds",
            show=False,
        )

        plot_seed_loss_envelope(
            seed_results,
            which="valid",
            save_path=os.path.join(arch_dir, "valid_loss_across_seeds.png"),
            title=f"{arch_name} | valid loss across seeds",
            show=False,
        )

        for component in ("x", "z"):
            plot_seed_prediction_envelope(
                seed_results,
                split="valid",
                component=component,
                traj_idx=traj_idx,
                x_idx=x_idx,
                z_idx=z_idx,
                save_path=os.path.join(arch_dir, f"valid_{component}_envelope.png"),
                title=f"{arch_name} | valid envelope | {component} | traj {traj_idx}",
                show=False,
            )

            plot_seed_prediction_all_curves(
                seed_results,
                split="valid",
                component=component,
                traj_idx=traj_idx,
                x_idx=x_idx,
                z_idx=z_idx,
                save_path=os.path.join(arch_dir, f"valid_{component}_all_seeds.png"),
                title=f"{arch_name} | valid all-seed curves | {component} | traj {traj_idx}",
                show=False,
            )

    plot_seed_final_loss_bar(
        all_seed_results,
        save_path=os.path.join(output_dir, "final_valid_loss_summary.png"),
        title="Final validation loss across seeds",
        show=False,
    )

    save_seed_summary_json(all_seed_results, output_dir=output_dir)