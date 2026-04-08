import os
import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from Energy_NN import MLP_Energy, ICNN_Energy, MLP_Stiffness, Signed_MLP_Stiffness
import dismech_jax as djx


# =========================================================
# 1) Slinky toy system with exact prescribed displacement
# =========================================================
class Slinky1D(djx.System):
    """Simple 2-node 1D spring under exact displacement control."""

    l_k: jax.Array
    x_left: jax.Array

    def get_q(self, disp: jax.Array, q0: jax.Array) -> jax.Array:
        """
        disp is the exact prescribed right-node x-position.
        If your stored data is displacement relative to initial position
        instead of absolute position, replace q.at[1].set(disp) with
        q.at[1].set(q0[1] + disp).
        """
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


def build_case_list() -> List[CaseConfig]:
    cases = [
        # ---------------------------------------------
        # Energy parameterizations
        # ---------------------------------------------
        CaseConfig("energy_baseline", "energy_mlp", (), "baseline"),
        CaseConfig("energy_mlp_L1", "energy_mlp", (10,), "mlp"),
        CaseConfig("energy_mlp_L2", "energy_mlp", (10, 10), "mlp"),
        CaseConfig("energy_baseline_plus_mlp_L1", "energy_mlp", (10,), "combined"),
        CaseConfig("energy_baseline_plus_mlp_L2", "energy_mlp", (10, 10), "combined"),
        CaseConfig("energy_icnn_L1", "energy_icnn", (10,), "icnn"),
        CaseConfig("energy_icnn_L2", "energy_icnn", (10, 10), "icnn"),
        CaseConfig("energy_baseline_plus_icnn_L1", "energy_icnn", (10,), "combined"),
        CaseConfig("energy_baseline_plus_icnn_L2", "energy_icnn", (10, 10), "combined"),

        # ---------------------------------------------
        # Stiffness parameterizations
        # ---------------------------------------------
        CaseConfig("stiffness_baseline", "stiffness_mlp", (), "only_baseline"),
        CaseConfig("stiffness_only_mlp_L1", "stiffness_mlp", (10,), "only_mlp"),
        CaseConfig("stiffness_only_mlp_L2", "stiffness_mlp", (10, 10), "only_mlp"),
        CaseConfig("stiffness_baseline_plus_mlp_L1", "stiffness_mlp", (10,), "combined"),
        CaseConfig("stiffness_baseline_plus_mlp_L2", "stiffness_mlp", (10, 10), "combined"),

        # Optional signed-correction stiffness variants
        CaseConfig("signed_stiffness_only_mlp_L1", "signed_stiffness_mlp", (10,), "only_mlp"),
        CaseConfig("signed_stiffness_only_mlp_L2", "signed_stiffness_mlp", (10, 10), "only_mlp"),
        CaseConfig("signed_stiffness_baseline_plus_mlp_L1", "signed_stiffness_mlp", (10,), "combined"),
        CaseConfig("signed_stiffness_baseline_plus_mlp_L2", "signed_stiffness_mlp", (10, 10), "combined"),
    ]
    return cases


def make_model(case: CaseConfig, key: jax.Array):
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


# =========================================================
# 3) Data + split helpers
# =========================================================
def load_problem(data_path="experiment_data/pulling_phase_data.npz", test_range=(0.2, 0.8)):
    data = np.load(data_path)

    force_truth = np.abs(jnp.array(data["F"]))
    disps = jnp.array(data["disps"])  # exact prescribed right-node positions

    initial_last_node_x = float(data["initial_last_node_x"])

    num_steps = force_truth.shape[0]

    x_left = 0.0
    q0 = jnp.array([x_left, initial_last_node_x])
    l_k0 = q0[1] - q0[0]

    slinky = Slinky1D(
        l_k=jnp.array(l_k0),
        x_left=jnp.array(x_left),
    )

    # split based on actual displacement range
    disp_min = disps.min()
    disp_max = disps.max()
    split_lo = disp_min + test_range[0] * (disp_max - disp_min)
    split_hi = disp_min + test_range[1] * (disp_max - disp_min)

    train_mask = (disps < split_lo) | (disps >= split_hi)
    test_mask = (disps >= split_lo) & (disps < split_hi)

    qs = jax.vmap(lambda d: slinky.get_q(d, q0))(disps)
    strains = jax.vmap(slinky.get_eps)(qs)

    return {
        "slinky": slinky,
        "q0": q0,
        "force_truth": force_truth,
        "disps": disps,
        "pulled_node_x": disps,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "strains": strains,
        "meta": {
            "initial_last_node_x": initial_last_node_x,
            "rest_length": float(l_k0),
            "num_steps": int(num_steps),
            "test_range_fraction": list(test_range),
            "disp_split_lo": float(split_lo),
            "disp_split_hi": float(split_hi),
        },
    }


# =========================================================
# 4) Metrics / derivatives
# =========================================================
def predict_force(slinky, q0, model, disp_vals):
    def one_force(disp):
        q = slinky.get_q(disp, q0)
        return slinky.get_reaction_force(disp, q, model, None)
    return jax.vmap(one_force)(disp_vals)


def predict_energy(model, strains):
    return jax.vmap(lambda eps: model(jnp.array([eps])))(strains)


def predict_effective_stiffness(model, strains):
    def scalar_energy(e):
        return model(jnp.array([e]))
    d2E = jax.grad(jax.grad(scalar_energy))
    return jax.vmap(d2E)(strains)


def summary_sharpness(stiffness_vals):
    s = np.asarray(stiffness_vals)
    if len(s) < 2:
        return 0.0
    ds = np.diff(s)
    return float(np.mean(np.abs(ds)))


# =========================================================
# 5) Training
# =========================================================
def train_one_case(
    case: CaseConfig,
    problem: Dict[str, Any],
    seed: int = 42,
    lr: float = 1e-2,
    num_epochs: int = 10000,
    log_freq: int = 500,
):
    slinky = problem["slinky"]
    q0 = problem["q0"]
    disps = problem["disps"]
    force_truth = problem["force_truth"]
    train_mask = problem["train_mask"]
    test_mask = problem["test_mask"]
    strains = problem["strains"]

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
    optimizer = optax.adam(learning_rate=schedule)
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
                    "[{case}] Epoch: {x}, Train Loss: {y}, Test Loss: {z}",
                    case=case.name,
                    x=i,
                    y=train_loss_val,
                    z=test_loss_val,
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
    pred_energy = predict_energy(final_model, strains)
    pred_stiffness = predict_effective_stiffness(final_model, strains)

    train_mse = float(jnp.mean((force_truth[train_mask] - pred_full_force[train_mask]) ** 2))
    test_mse = float(jnp.mean((force_truth[test_mask] - pred_full_force[test_mask]) ** 2))
    gap = float(test_mse - train_mse)
    sharpness = summary_sharpness(np.asarray(pred_stiffness))

    loss_history_np = np.asarray(loss_history)   # shape: (num_epochs+1, 2)
    train_hist = loss_history_np[0, :] # CORRECTED
    test_hist = loss_history_np[1, :] # CORRECTED

    result = {
        "case": asdict(case),
        "train_mse": train_mse,
        "test_mse": test_mse,
        "generalization_gap": gap,
        "stiffness_sharpness": sharpness,
        "disps": np.asarray(disps),
        "pulled_node_x": np.asarray(problem["pulled_node_x"]),
        "force_truth": np.asarray(force_truth),
        "pred_force": np.asarray(pred_full_force),
        "strains": np.asarray(strains),
        "pred_energy": np.asarray(pred_energy),
        "pred_stiffness": np.asarray(pred_stiffness),
        "train_mask": np.asarray(train_mask),
        "test_mask": np.asarray(test_mask),
        "train_hist": np.asarray(train_hist),
        "test_hist": np.asarray(test_hist),
        "meta": problem["meta"],
    }

    return final_model, result


# =========================================================
# 6) Save helpers
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_case_outputs(outdir, model, result):
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


# =========================================================
# 7) Plot helpers
# =========================================================
def _add_test_force_band(ax, reference_result, label="Test region (GT force range)"):
    y_true = np.asarray(reference_result["force_truth"])
    test_mask = np.asarray(reference_result["test_mask"]).astype(bool)

    if np.any(test_mask):
        y_test = y_true[test_mask]
        y0 = float(np.min(y_test))
        y1 = float(np.max(y_test))
        ax.axhspan(y0, y1, alpha=0.12, color="gray", label=label, zorder=0)


def _style_for_index(i):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    markers = ["s", "^", "D", "v", "P", "X"]
    linestyles = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
    return {
        "color": colors[i % len(colors)],
        "marker": markers[i % len(markers)],
        "linestyle": linestyles[i % len(linestyles)],
    }


def plot_force_displacement_overlay(results_by_name, save_path, selected_names=None, title=None):
    if selected_names is None:
        selected_names = list(results_by_name.keys())

    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    ref = results_by_name[selected_names[0]]
    x_ref = np.asarray(ref["pulled_node_x"])
    y_true = np.asarray(ref["force_truth"])
    train_mask = np.asarray(ref["train_mask"]).astype(bool)
    test_mask = np.asarray(ref["test_mask"]).astype(bool)

    _add_test_force_band(ax, ref)

    ax.plot(
        x_ref[train_mask],
        y_true[train_mask],
        linestyle="None",
        marker="o",
        markersize=6,
        markerfacecolor="black",
        markeredgecolor="black",
        label="GT train samples",
        zorder=5,
    )
    ax.plot(
        x_ref[test_mask],
        y_true[test_mask],
        linestyle="None",
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor="black",
        label="GT test samples",
        zorder=5,
    )

    for i, name in enumerate(selected_names):
        r = results_by_name[name]
        x = np.asarray(r["pulled_node_x"])
        y_pred = np.asarray(r["pred_force"])
        style = _style_for_index(i)
        ax.plot(
            x,
            y_pred,
            linewidth=2,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=5,
            markevery=max(1, len(x) // 12),
            label=name,
            zorder=3,
        )

    ax.set_xlabel("Pulled node x")
    ax.set_ylabel("Force")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_energy_strain_overlay(results_by_name, save_path, selected_names=None, title=None):
    if selected_names is None:
        selected_names = list(results_by_name.keys())

    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    for i, name in enumerate(selected_names):
        r = results_by_name[name]
        eps = np.asarray(r["strains"])
        E = np.asarray(r["pred_energy"])
        order = np.argsort(eps)
        style = _style_for_index(i)
        ax.plot(
            eps[order],
            E[order],
            linewidth=2,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4,
            markevery=max(1, len(eps) // 12),
            label=name,
        )

    ax.set_xlabel("Strain")
    ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_stiffness_strain_overlay(results_by_name, save_path, selected_names=None, title=None):
    if selected_names is None:
        selected_names = list(results_by_name.keys())

    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    for i, name in enumerate(selected_names):
        r = results_by_name[name]
        eps = np.asarray(r["strains"])
        K = np.asarray(r["pred_stiffness"])
        order = np.argsort(eps)
        style = _style_for_index(i)
        ax.plot(
            eps[order],
            K[order],
            linewidth=2,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4,
            markevery=max(1, len(eps) // 12),
            label=f"{name} (sharp={r['stiffness_sharpness']:.2e})",
        )

    ax.set_xlabel("Strain")
    ax.set_ylabel(r"Effective stiffness $d^2E/d\epsilon^2$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_loss_histories_overlay(results_by_name, save_path, selected_names=None, title=None):
    if selected_names is None:
        selected_names = list(results_by_name.keys())

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.2), sharex=True)
    ax_train, ax_test = axes

    for i, name in enumerate(selected_names):
        r = results_by_name[name]
        style = _style_for_index(i)
        ax_train.plot(
            r["train_hist"],
            linewidth=2,
            color=style["color"],
            linestyle=style["linestyle"],
            label=name,
        )
        ax_test.plot(
            r["test_hist"],
            linewidth=2,
            color=style["color"],
            linestyle=style["linestyle"],
            label=name,
        )

    ax_train.set_yscale("log")
    ax_test.set_yscale("log")
    ax_train.set_ylabel("Train force MSE")
    ax_test.set_ylabel("Test force MSE")
    ax_test.set_xlabel("Epoch")
    ax_train.grid(True, alpha=0.3)
    ax_test.grid(True, alpha=0.3)
    ax_train.legend(loc="best", fontsize=8)

    if title is not None:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_summary_bars(summary_rows, save_path, title=None):
    names = [row["name"] for row in summary_rows]
    train_mse = np.array([row["train_mse"] for row in summary_rows])
    test_mse = np.array([row["test_mse"] for row in summary_rows])
    sharpness = np.array([row["stiffness_sharpness"] for row in summary_rows])

    x = np.arange(len(names))
    width = 0.38

    fig, axes = plt.subplots(2, 1, figsize=(max(14, 0.75 * len(names)), 10), sharex=True)

    axes[0].bar(x - width / 2, train_mse, width, label="Train MSE")
    axes[0].bar(x + width / 2, test_mse, width, label="Test MSE")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Force MSE")
    axes[0].set_title("Ablation summary: train vs test MSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(x, sharpness)
    axes[1].set_ylabel("Sharpness proxy")
    axes[1].set_title("Energy landscape sharpness proxy")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=40, ha="right")

    if title is not None:
        fig.suptitle(title, y=0.995, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


# =========================================================
# 8) Main driver
# =========================================================
def main():
    out_root = "ablation_outputs"
    case_outdir = os.path.join(out_root, "cases")
    fig_outdir = os.path.join(out_root, "figures")
    ensure_dir(case_outdir)
    ensure_dir(fig_outdir)

    problem = load_problem(data_path="experiment_data/pulling_phase_data.npz", test_range=(0.2, 0.8))
    cases = build_case_list()

    results_by_name = {}
    summary_rows = []

    for i, case in enumerate(cases):
        print("=" * 80)
        print(f"Running case {i + 1}/{len(cases)}: {case.name}")
        final_model, result = train_one_case(
            case=case,
            problem=problem,
            seed=42,
            lr=1e-2,
            num_epochs=10000,
            log_freq=500,
        )
        save_case_outputs(case_outdir, final_model, result)
        results_by_name[case.name] = result
        summary_rows.append(
            {
                "name": case.name,
                "family": case.family,
                "hidden_sizes": case.hidden_sizes,
                "which_case": case.which_case,
                "train_mse": result["train_mse"],
                "test_mse": result["test_mse"],
                "generalization_gap": result["generalization_gap"],
                "stiffness_sharpness": result["stiffness_sharpness"],
            }
        )

    summary_rows = sorted(summary_rows, key=lambda x: x["test_mse"])
    np.savez(os.path.join(out_root, "summary_table.npz"), rows=np.array(summary_rows, dtype=object))

    with open(os.path.join(out_root, "summary_table.json"), "w") as f:
        json.dump(summary_rows, f, indent=2)

    ablation_A = [
        "energy_baseline",
        "energy_mlp_L2",
        "energy_icnn_L2",
        # "stiffness_only_mlp_L2",
        "stiffness_baseline_plus_mlp_L2",
    ]
    
    # ablation_A = [
    #     "energy_baseline",
    #     "energy_mlp_L1",
    #     "energy_icnn_L1",
    #     "stiffness_only_mlp_L1",
    #     "stiffness_baseline_plus_mlp_L1",
    # ]

    ablation_A = [name for name in ablation_A if name in results_by_name]

    plot_force_displacement_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "ablation_A_force_displacement_overlay.png"),
        selected_names=ablation_A,
        title="Ablation A: parameterization and architecture",
    )
    plot_stiffness_strain_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "ablation_A_stiffness_strain_overlay.png"),
        selected_names=ablation_A,
        title="Ablation A: parameterization and architecture",
    )
    plot_energy_strain_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "ablation_A_energy_strain_overlay.png"),
        selected_names=ablation_A,
        title="Ablation A: parameterization and architecture",
    )
    plot_loss_histories_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "ablation_A_loss_histories_overlay.png"),
        selected_names=ablation_A,
        title="Ablation A: training dynamics",
    )
    plot_summary_bars(
        [row for row in summary_rows if row["name"] in ablation_A],
        save_path=os.path.join(fig_outdir, "ablation_A_summary_bars.png"),
        title="Ablation A: summary metrics",
    )

    ablation_B = [
        "energy_icnn_L1",
        "energy_icnn_L2",
        "stiffness_only_mlp_L1",
        "stiffness_only_mlp_L2",
    ]
    ablation_B = [name for name in ablation_B if name in results_by_name]

    plot_force_displacement_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "ablation_B_force_displacement_overlay.png"),
        selected_names=ablation_B,
        title="Ablation B: depth effect",
    )
    plot_stiffness_strain_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "ablation_B_stiffness_strain_overlay.png"),
        selected_names=ablation_B,
        title="Ablation B: depth effect",
    )
    plot_energy_strain_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "ablation_B_energy_strain_overlay.png"),
        selected_names=ablation_B,
        title="Ablation B: depth effect",
    )
    plot_loss_histories_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "ablation_B_loss_histories_overlay.png"),
        selected_names=ablation_B,
        title="Ablation B: training dynamics",
    )
    plot_summary_bars(
        [row for row in summary_rows if row["name"] in ablation_B],
        save_path=os.path.join(fig_outdir, "ablation_B_summary_bars.png"),
        title="Ablation B: summary metrics",
    )

    all_names = list(results_by_name.keys())
    plot_force_displacement_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "supp_force_displacement_all.png"),
        selected_names=all_names,
        title="Supplement: all cases",
    )
    plot_energy_strain_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "supp_energy_strain_all.png"),
        selected_names=all_names,
        title="Supplement: all cases",
    )
    plot_stiffness_strain_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "supp_stiffness_strain_all.png"),
        selected_names=all_names,
        title="Supplement: all cases",
    )
    plot_loss_histories_overlay(
        results_by_name,
        save_path=os.path.join(fig_outdir, "supp_loss_histories_all.png"),
        selected_names=all_names,
        title="Supplement: all cases",
    )

    print("\nDone. Outputs written to:")
    print(f"  {out_root}/summary_table.json")
    print(f"  {fig_outdir}/")
    print(f"  {case_outdir}/")


if __name__ == "__main__":
    main()