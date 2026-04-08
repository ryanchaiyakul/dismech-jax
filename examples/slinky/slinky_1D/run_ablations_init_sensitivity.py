import os
import json
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from Energy_NN import MLP_Stiffness, ICNN_Energy
import dismech_jax as djx


# =========================================================
# 1) Slinky toy system with exact prescribed displacement
# =========================================================
class Slinky1D(djx.System):
    """Simple 2-node 1D spring under exact displacement control."""

    l_k: jax.Array
    x_left: jax.Array

    def get_q(self, disp: jax.Array, q0: jax.Array) -> jax.Array:
        q = q0.at[0].set(self.x_left)
        q = q.at[1].set(disp)
        return q

    def get_eps(self, q: jax.Array) -> jax.Array:
        return (q[1] - q[0]) / self.l_k - 1.0

    def get_E(self, disp: jax.Array, q: jax.Array, model: eqx.Module, aux: None) -> jax.Array:
        eps = self.get_eps(q)
        return model(jnp.array([eps]))

    def get_F(self, disp: jax.Array, q: jax.Array, model: eqx.Module, aux: None) -> jax.Array:
        return jnp.zeros_like(q)

    def get_H(self, disp: jax.Array, q: jax.Array, model: eqx.Module, aux: None) -> jax.Array:
        return jnp.eye(q.shape[0])

    def get_reaction_force(self, disp: jax.Array, q: jax.Array, model: eqx.Module, aux: None) -> jax.Array:
        dEdq = jax.grad(self.get_E, argnums=1)(disp, q, model, aux)
        return dEdq[1]


# =========================================================
# 2) Configs
# =========================================================
@dataclass(frozen=True)
class CaseConfig:
    name: str
    family: str
    hidden_sizes: tuple
    which_case: str
    K_initial: float = 0.1
    weight_scale: float = 1.0


def selected_cases():
    return [
        CaseConfig("energy_icnn_L2", "energy_icnn", (10, 10), "icnn"),
        CaseConfig("stiffness_baseline_plus_mlp_L2", "stiffness_mlp", (10, 10), "combined"),
    ]


def make_model(case: CaseConfig, key: jax.Array):
    kwargs = dict(
        key=key,
        K_initial=case.K_initial,
        which_case=case.which_case,
        weight_scale=case.weight_scale,
    )

    if case.family == "energy_icnn":
        return ICNN_Energy(hidden_sizes=case.hidden_sizes, **kwargs)

    if case.family == "stiffness_mlp":
        return MLP_Stiffness(hidden_sizes=case.hidden_sizes, **kwargs)

    raise ValueError(f"Unknown family: {case.family}")


# =========================================================
# 3) Data loading
# =========================================================
def load_problem(data_path="experiment_data/pulling_phase_data.npz", test_range=(0.2, 0.8)):
    data = np.load(data_path)

    force_truth = np.abs(jnp.array(data["F"]))
    disps = jnp.array(data["disps"])
    initial_last_node_x = float(data["initial_last_node_x"])

    x_left = 0.0
    q0 = jnp.array([x_left, initial_last_node_x])
    l_k0 = q0[1] - q0[0]

    slinky = Slinky1D(
        l_k=jnp.array(l_k0),
        x_left=jnp.array(x_left),
    )

    disp_min = disps.min()
    disp_max = disps.max()
    split_lo = disp_min + test_range[0] * (disp_max - disp_min)
    split_hi = disp_min + test_range[1] * (disp_max - disp_min)

    train_mask = (disps < split_lo) | (disps >= split_hi)
    test_mask = (disps >= split_lo) & (disps < split_hi)

    return {
        "slinky": slinky,
        "q0": q0,
        "force_truth": force_truth,
        "disps": disps,
        "pulled_node_x": disps,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "meta": {
            "initial_last_node_x": initial_last_node_x,
            "rest_length": float(l_k0),
            "disp_split_lo": float(split_lo),
            "disp_split_hi": float(split_hi),
        },
    }


# =========================================================
# 4) Prediction
# =========================================================
def predict_force(slinky, q0, model, disp_vals):
    def one_force(disp):
        q = slinky.get_q(disp, q0)
        return slinky.get_reaction_force(disp, q, model, None)
    return jax.vmap(one_force)(disp_vals)

# =========================================================
# 5) Training
# =========================================================

def train_one_case(
    case: CaseConfig,
    problem,
    seed=0,
    lr=1e-3,
    num_epochs=10000,
    log_freq=500,
):
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
    
    def train_loss(model):
        pred_force = predict_force(slinky, q0, model, train_disps)
        return jnp.mean((train_force_truth - pred_force) ** 2)

    def test_loss(model):
        pred_force = predict_force(slinky, q0, model, test_disps)
        return jnp.mean((test_force_truth - pred_force) ** 2)

    schedule = optax.piecewise_constant_schedule(
        init_value=lr,
        boundaries_and_scales={7500: 0.1},
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(schedule)
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    @eqx.filter_jit
    def train_step(carry, _):
        model, opt_state = carry
        loss_val, grads = eqx.filter_value_and_grad(train_loss)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return (model, opt_state), loss_val

    def train_loop(model, opt_state):
        def scan_fn(carry, i):
            next_carry, train_loss_val = train_step(carry, None)
            model_next, _ = next_carry
            test_loss_val = test_loss(model_next)

            def log_loss(_):
                jax.debug.print(
                    "[{case}] seed={seed}, epoch={epoch}, train={train}, test={test}",
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
            (model, opt_state),
            jnp.arange(num_epochs + 1),
        )
        return final_model, loss_history

    final_model, loss_history = train_loop(model, opt_state)
    pred_full_force = predict_force(slinky, q0, final_model, disps)

    train_mse = float(jnp.mean((force_truth[train_mask] - pred_full_force[train_mask]) ** 2))
    test_mse = float(jnp.mean((force_truth[test_mask] - pred_full_force[test_mask]) ** 2))

    return final_model, {
        "case": asdict(case),
        "seed": seed,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "pulled_node_x": np.asarray(problem["pulled_node_x"]),
        "force_truth": np.asarray(force_truth),
        "pred_force": np.asarray(pred_full_force),
        "train_mask": np.asarray(train_mask),
        "test_mask": np.asarray(test_mask),
        "train_hist": np.asarray(loss_history)[0,:], # CORRECTED INDEXING
        "test_hist": np.asarray(loss_history)[1,:], # CORRECTED INDEXING
    }


# =========================================================
# 6) Plot helpers
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_case_arrays(results):
    results = sorted(results, key=lambda r: r["seed"])
    x = results[0]["pulled_node_x"]
    y_true = results[0]["force_truth"]
    train_mask = results[0]["train_mask"].astype(bool)
    test_mask = results[0]["test_mask"].astype(bool)
    preds = np.stack([r["pred_force"] for r in results], axis=0)
    test_mse = np.array([r["test_mse"] for r in results], dtype=float)
    best_idx = int(np.argmin(test_mse))
    best = results[best_idx]
    median_curve = np.median(preds, axis=0)
    q05 = np.percentile(preds, 5, axis=0)
    q25 = np.percentile(preds, 25, axis=0)
    q75 = np.percentile(preds, 75, axis=0)
    q95 = np.percentile(preds, 95, axis=0)
    ymin = np.min(preds, axis=0)
    ymax = np.max(preds, axis=0)
    var_curve = np.var(preds, axis=0)
    return {
        "x": x,
        "y_true": y_true,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "preds": preds,
        "test_mse": test_mse,
        "best": best,
        "median_curve": median_curve,
        "q05": q05,
        "q25": q25,
        "q75": q75,
        "q95": q95,
        "ymin": ymin,
        "ymax": ymax,
        "var_curve": var_curve,
        "results": results,
    }


def add_gt(ax, x, y_true, train_mask, test_mask):
    ax.plot(
        x[train_mask], y_true[train_mask],
        linestyle="None", marker="o", markersize=6,
        markerfacecolor="black", markeredgecolor="black",
        label="GT train samples", zorder=5,
    )
    ax.plot(
        x[test_mask], y_true[test_mask],
        linestyle="None", marker="o", markersize=6,
        markerfacecolor="white", markeredgecolor="black",
        label="GT test samples", zorder=5,
    )


def plot_minmax_envelope(results, save_path, title=None):
    d = get_case_arrays(results)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    ax.fill_between(
        d["x"], d["ymin"], d["ymax"],
        alpha=0.25, label=f"Min-max envelope ({len(d['results'])} seeds)", zorder=1
    )
    ax.plot(
        d["x"], d["best"]["pred_force"],
        linewidth=2.8,
        label=f"Best seed = {d['best']['seed']} (test MSE = {d['best']['test_mse']:.3e})",
        zorder=3,
    )

    add_gt(ax, d["x"], d["y_true"], d["train_mask"], d["test_mask"])
    ax.set_xlabel("Pulled node x")
    ax.set_ylabel("Force")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_percentile_envelope(results, save_path, title=None):
    d = get_case_arrays(results)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    ax.fill_between(
        d["x"], d["q05"], d["q95"],
        alpha=0.18, label="5-95 percentile band", zorder=1
    )
    ax.fill_between(
        d["x"], d["q25"], d["q75"],
        alpha=0.30, label="25-75 percentile band", zorder=2
    )
    ax.plot(
        d["x"], d["median_curve"],
        linestyle="--", linewidth=2.3, label="Median across seeds", zorder=3
    )
    ax.plot(
        d["x"], d["best"]["pred_force"],
        linewidth=2.8,
        label=f"Best seed = {d['best']['seed']} (test MSE = {d['best']['test_mse']:.3e})",
        zorder=4,
    )

    add_gt(ax, d["x"], d["y_true"], d["train_mask"], d["test_mask"])
    ax.set_xlabel("Pulled node x")
    ax.set_ylabel("Force")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_all_seeds(results, save_path, title=None):
    d = get_case_arrays(results)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    for r in d["results"]:
        ax.plot(
            d["x"], r["pred_force"],
            linewidth=1.0, alpha=0.18, zorder=1
        )

    ax.plot(
        d["x"], d["median_curve"],
        linestyle="--", linewidth=2.2, label="Median across seeds", zorder=3
    )
    ax.plot(
        d["x"], d["best"]["pred_force"],
        linewidth=2.8,
        label=f"Best seed = {d['best']['seed']} (test MSE = {d['best']['test_mse']:.3e})",
        zorder=4,
    )

    add_gt(ax, d["x"], d["y_true"], d["train_mask"], d["test_mask"])
    ax.set_xlabel("Pulled node x")
    ax.set_ylabel("Force")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_variance_curve(results, save_path, title=None):
    d = get_case_arrays(results)
    fig, ax = plt.subplots(figsize=(8.5, 5.0))

    ax.plot(d["x"], d["var_curve"], linewidth=2.5)
    ax.set_xlabel("Pulled node x")
    ax.set_ylabel("Variance across seeds")
    ax.grid(True, alpha=0.3)
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_test_mse_hist(results, save_path, title=None):
    d = get_case_arrays(results)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))

    ax.hist(d["test_mse"], bins=15, edgecolor="black", alpha=0.8)
    ax.axvline(np.min(d["test_mse"]), linestyle="--", linewidth=2, label="Best seed")
    ax.axvline(np.median(d["test_mse"]), linestyle=":", linewidth=2, label="Median")
    ax.set_xlabel("Test MSE")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if title is not None:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_summary_panel(results, save_path, title=None):
    d = get_case_arrays(results)

    # =========================
    # loss-envelope statistics
    # =========================
    train_hist = np.stack([r["train_hist"] for r in results], axis=0)
    test_hist  = np.stack([r["test_hist"] for r in results], axis=0)
    epochs = np.arange(train_hist.shape[1])

    train_median = np.median(train_hist, axis=0)
    train_p05 = np.percentile(train_hist, 5, axis=0)
    train_p95 = np.percentile(train_hist, 95, axis=0)

    test_median = np.median(test_hist, axis=0)
    test_p05 = np.percentile(test_hist, 5, axis=0)
    test_p95 = np.percentile(test_hist, 95, axis=0)

    best = min(results, key=lambda r: r["test_mse"])
    best_train = best["train_hist"]
    best_test = best["test_hist"]

    # =========================
    # figure: 1 x 2
    # =========================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax1, ax2 = axes

    # -------------------------------------------------
    # panel 1: all seed curves
    # -------------------------------------------------
    for r in d["results"]:
        ax1.plot(
            d["x"], r["pred_force"],
            linewidth=0.9, alpha=0.15
        )

    ax1.plot(
        d["x"], d["median_curve"],
        "--", linewidth=2.0, label="Median"
    )
    ax1.plot(
        d["x"], d["best"]["pred_force"],
        linewidth=2.4, label=f"Best seed={d['best']['seed']}"
    )
    add_gt(ax1, d["x"], d["y_true"], d["train_mask"], d["test_mask"])
    ax1.set_xlabel("Pulled node x")
    ax1.set_ylabel("Force")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    # -------------------------------------------------
    # panel 2: loss envelope
    # -------------------------------------------------
    ax2.fill_between(
        epochs, train_p05, train_p95,
        alpha=0.20, label="Train (5–95%)"
    )
    ax2.plot(
        epochs, train_median,
        linestyle="--", linewidth=2.0,
        label="Train median"
    )

    ax2.fill_between(
        epochs, test_p05, test_p95,
        alpha=0.20, label="Test (5–95%)"
    )
    ax2.plot(
        epochs, test_median,
        linewidth=2.5,
        label="Test median"
    )

    ax2.plot(
        epochs, best_train,
        linestyle=":", linewidth=2.0,
        label=f"Best seed train (seed={best['seed']})"
    )
    ax2.plot(
        epochs, best_test,
        linestyle="-", linewidth=2.8,
        label="Best seed test"
    )

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MSE loss")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    if title is not None:
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_loss_envelope(results, save_path, title=None):
            """
            Plots train/test loss across epochs with shaded regions over seeds.
            """

            # stack histories: shape = (num_seeds, num_epochs)
            train_hist = np.stack([r["train_hist"] for r in results], axis=0)
            test_hist  = np.stack([r["test_hist"]  for r in results], axis=0)

            epochs = np.arange(train_hist.shape[1])

            # statistics
            train_median = np.median(train_hist, axis=0)
            train_p05 = np.percentile(train_hist, 5, axis=0)
            train_p95 = np.percentile(train_hist, 95, axis=0)

            test_median = np.median(test_hist, axis=0)
            test_p05 = np.percentile(test_hist, 5, axis=0)
            test_p95 = np.percentile(test_hist, 95, axis=0)

            # best seed (by final test MSE)
            best = min(results, key=lambda r: r["test_mse"])
            best_train = best["train_hist"]
            best_test = best["test_hist"]

            fig, ax = plt.subplots(figsize=(8.5, 5.5))

            # --- train ---
            ax.fill_between(
                epochs, train_p05, train_p95,
                alpha=0.20, label="Train (5–95%)"
            )
            ax.plot(
                epochs, train_median,
                linestyle="--", linewidth=2.0,
                label="Train median"
            )

            # --- test ---
            ax.fill_between(
                epochs, test_p05, test_p95,
                alpha=0.20, label="Test (5–95%)"
            )
            ax.plot(
                epochs, test_median,
                linewidth=2.5,
                label="Test median"
            )

            # --- best seed ---
            ax.plot(
                epochs, best_train,
                linestyle=":", linewidth=2.0,
                label=f"Best seed train (seed={best['seed']})"
            )
            ax.plot(
                epochs, best_test,
                linestyle="-", linewidth=2.8,
                label=f"Best seed test"
            )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("MSE loss")
            ax.set_yscale("log")  # important for readability
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)

            if title is not None:
                ax.set_title(title)

            fig.tight_layout()
            fig.savefig(save_path, dpi=300)
            plt.close(fig)

# =========================================================
# 6) Main
# =========================================================
def main():
    out_root = "seed_envelope_two_models"
    run_dir = os.path.join(out_root, "runs")
    fig_dir = os.path.join(out_root, "figures")
    ensure_dir(run_dir)
    ensure_dir(fig_dir)

    problem = load_problem(
        data_path="experiment_data/pulling_phase_data.npz",
        test_range=(0.2, 0.8),
    )

    cases = selected_cases()
    seeds = list(range(50))

    all_results = {case.name: [] for case in cases}
    summary_rows = []

    for case in cases:
        print("=" * 90)
        print(f"Running case: {case.name}")

        for seed in seeds:
            print(f"  seed = {seed}")
            model, result = train_one_case(
                case=case,
                problem=problem,
                seed=seed,
                lr=1e-3,
                num_epochs=10000,
                log_freq=500,
            )

            all_results[case.name].append(result)
            summary_rows.append(
                {
                    "name": case.name,
                    "seed": seed,
                    "train_mse": result["train_mse"],
                    "test_mse": result["test_mse"],
                }
            )

            stem = f"{case.name}__seed{seed:03d}"
            np.savez(
                os.path.join(run_dir, f"{stem}.npz"),
                pulled_node_x=result["pulled_node_x"],
                force_truth=result["force_truth"],
                pred_force=result["pred_force"],
                train_mask=result["train_mask"],
                test_mask=result["test_mask"],
                train_hist=result["train_hist"],
                test_hist=result["test_hist"],
                train_mse=result["train_mse"],
                test_mse=result["test_mse"],
                seed=result["seed"],
            )
            eqx.tree_serialise_leaves(os.path.join(run_dir, f"{stem}.eqx"), model)

        case_results = all_results[case.name]

        plot_minmax_envelope(
            case_results,
            save_path=os.path.join(fig_dir, f"{case.name}_minmax_envelope.png"),
            title=f"{case.name}: min-max seed envelope",
        )
        plot_percentile_envelope(
            case_results,
            save_path=os.path.join(fig_dir, f"{case.name}_percentile_envelope.png"),
            title=f"{case.name}: percentile seed envelope",
        )
        plot_all_seeds(
            case_results,
            save_path=os.path.join(fig_dir, f"{case.name}_all_seed_curves.png"),
            title=f"{case.name}: all seed predictions",
        )
        plot_variance_curve(
            case_results,
            save_path=os.path.join(fig_dir, f"{case.name}_variance_curve.png"),
            title=f"{case.name}: variance across seeds",
        )
        plot_test_mse_hist(
            case_results,
            save_path=os.path.join(fig_dir, f"{case.name}_test_mse_hist.png"),
            title=f"{case.name}: test MSE distribution across seeds",
        )
        plot_summary_panel(
            case_results,
            save_path=os.path.join(fig_dir, f"{case.name}_summary_panel.png"),
            title=f"{case.name}: initialization sensitivity summary",
        )
        plot_loss_envelope(
            case_results,
            save_path=os.path.join(fig_dir, f"{case.name}_loss_envelope.png"),
            title=f"{case.name}: loss across seeds",
        )
        

    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)

    print("\nDone.")
    print(f"Saved runs to:    {run_dir}")
    print(f"Saved figures to: {fig_dir}")


if __name__ == "__main__":
    main()