import os
import json
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, List

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from Energy_NN import MLP_Energy, ICNN_Energy, MLP_Stiffness, Signed_MLP_Stiffness
import dismech_jax as djx


# =========================================================
# 1) Slinky toy system
# =========================================================
class Slinky1D(djx.System):
    """Simple 2-node 1D spring under displacement control."""

    l_k: jax.Array
    x_left: jax.Array
    x_right_initial: jax.Array
    x_right_final: jax.Array

    def get_q(self, _lambda: jax.Array, q0: jax.Array) -> jax.Array:
        x_right = (1.0 - _lambda) * self.x_right_initial + _lambda * self.x_right_final
        q = q0.at[0].set(self.x_left)
        q = q.at[1].set(x_right)
        return q

    def get_eps(self, q: jax.Array) -> jax.Array:
        return (q[1] - q[0]) / self.l_k - 1.0

    def get_E(self, _lambda: jax.Array, q: jax.Array, model: eqx.Module, aux: None) -> jax.Array:
        eps = self.get_eps(q)
        return model(jnp.array([eps]))

    def get_F(self, _lambda: jax.Array, q: jax.Array, model: eqx.Module, aux: None) -> jax.Array:
        return jnp.zeros_like(q)

    def get_H(self, _lambda: jax.Array, q: jax.Array, model: eqx.Module, aux: None) -> jax.Array:
        return jnp.eye(q.shape[0])

    def get_reaction_force(self, _lambda: jax.Array, q: jax.Array, model: eqx.Module, aux: None) -> jax.Array:
        dEdq = jax.grad(self.get_E, argnums=1)(_lambda, q, model, aux)
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
    K_initial: float = 1.0
    weight_scale: float = 0.01


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
    kwargs = dict(key=key, K_initial=case.K_initial, which_case=case.which_case, weight_scale=case.weight_scale)

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
def load_problem(data_path="slinky_pulling_force_data.npz", test_range=(0.2, 0.8)):
    data = np.load(data_path)

    force_truth = np.abs(jnp.array(data["F"]))
    initial_last_node_x = float(data["initial_last_node_x"])
    final_last_node_x = float(data["final_last_node_x"])

    num_steps = force_truth.shape[0]
    lambdas = jnp.linspace(0.0, 1.0, num_steps)

    x_left = 0.0
    q0 = jnp.array([x_left, initial_last_node_x])
    l_k0 = q0[1] - q0[0]

    slinky = Slinky1D(
        l_k=jnp.array(l_k0),
        x_left=jnp.array(x_left),
        x_right_initial=jnp.array(initial_last_node_x),
        x_right_final=jnp.array(final_last_node_x),
    )

    train_mask = (lambdas < test_range[0]) | (lambdas >= test_range[1])
    test_mask = (lambdas >= test_range[0]) & (lambdas < test_range[1])

    pulled_node_x = (1.0 - lambdas) * initial_last_node_x + lambdas * final_last_node_x
    qs = jax.vmap(lambda lam: slinky.get_q(lam, q0))(lambdas)
    strains = jax.vmap(slinky.get_eps)(qs)

    return {
        "slinky": slinky,
        "q0": q0,
        "force_truth": force_truth,
        "lambdas": lambdas,
        "pulled_node_x": pulled_node_x,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "strains": strains,
        "meta": {
            "initial_last_node_x": initial_last_node_x,
            "final_last_node_x": final_last_node_x,
            "rest_length": float(l_k0),
            "num_steps": int(num_steps),
            "test_range": list(test_range),
        },
    }


# =========================================================
# 4) Metrics / derivatives
# =========================================================
def predict_force(slinky, q0, model, lambda_vals):
    def one_force(_lambda):
        q = slinky.get_q(_lambda, q0)
        return slinky.get_reaction_force(_lambda, q, model, None)
    return jax.vmap(one_force)(lambda_vals)


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
    lambdas = problem["lambdas"]
    force_truth = problem["force_truth"]
    train_mask = problem["train_mask"]
    test_mask = problem["test_mask"]
    strains = problem["strains"]

    train_lambdas = lambdas[train_mask]
    test_lambdas = lambdas[test_mask]
    train_force_truth = force_truth[train_mask]
    test_force_truth = force_truth[test_mask]

    model = make_model(case, jax.random.PRNGKey(seed))

    def train_loss(model):
        pred_force = predict_force(slinky, q0, model, train_lambdas)
        return jnp.mean((train_force_truth - pred_force) ** 2)

    def test_loss(model):
        pred_force = predict_force(slinky, q0, model, test_lambdas)
        return jnp.mean((test_force_truth - pred_force) ** 2)

    schedule = optax.piecewise_constant_schedule(
        init_value=lr,
        # boundaries_and_scales={3000: 0.1, 6000: 0.1},
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

    pred_full_force = predict_force(slinky, q0, final_model, lambdas)
    pred_energy = predict_energy(final_model, strains)
    pred_stiffness = predict_effective_stiffness(final_model, strains)

    train_mse = float(jnp.mean((force_truth[train_mask] - pred_full_force[train_mask]) ** 2))
    test_mse = float(jnp.mean((force_truth[test_mask] - pred_full_force[test_mask]) ** 2))
    gap = float(test_mse - train_mse)
    sharpness = summary_sharpness(np.asarray(pred_stiffness))

    loss_history_np = np.asarray(loss_history)
    train_hist = loss_history_np[0,:]
    test_hist = loss_history_np[1,:]

    result = {
        "case": asdict(case),
        "train_mse": train_mse,
        "test_mse": test_mse,
        "generalization_gap": gap,
        "stiffness_sharpness": sharpness,
        "lambdas": np.asarray(lambdas),
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
        lambdas=result["lambdas"],
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
def plot_force_displacement_grid(results_by_name, save_path, selected_names=None):
    if selected_names is None:
        selected_names = list(results_by_name.keys())

    n = len(selected_names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, name in zip(axes, selected_names):
        r = results_by_name[name]
        x = r["pulled_node_x"]
        y_true = r["force_truth"]
        y_pred = r["pred_force"]
        train_mask = r["train_mask"].astype(bool)
        test_mask = r["test_mask"].astype(bool)

        ax.plot(x, y_true, linewidth=2, label="Ground truth")
        ax.plot(x, y_pred, "--", linewidth=2, label="Prediction")
        ax.plot(x[train_mask], y_true[train_mask], "o", markersize=5, label="Train region")
        ax.plot(x[test_mask], y_true[test_mask], "s", markersize=5, label="Test region")
        ax.set_title(name)
        ax.set_xlabel("Pulled node x")
        ax.set_ylabel("Force")
        ax.grid(True, alpha=0.3)

    for ax in axes[n:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)



def plot_energy_strain_grid(results_by_name, save_path, selected_names=None):
    if selected_names is None:
        selected_names = list(results_by_name.keys())

    n = len(selected_names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, name in zip(axes, selected_names):
        r = results_by_name[name]
        eps = r["strains"]
        E = r["pred_energy"]
        train_mask = r["train_mask"].astype(bool)
        test_mask = r["test_mask"].astype(bool)

        ax.plot(eps, E, linewidth=2, label="Learned energy")
        ax.scatter(eps[train_mask], E[train_mask], s=25, label="Train region")
        ax.scatter(eps[test_mask], E[test_mask], s=25, marker="s", label="Test region")
        ax.set_title(name)
        ax.set_xlabel("Strain")
        ax.set_ylabel("Energy")
        ax.grid(True, alpha=0.3)

    for ax in axes[n:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)



def plot_stiffness_strain_grid(results_by_name, save_path, selected_names=None):
    if selected_names is None:
        selected_names = list(results_by_name.keys())

    n = len(selected_names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, name in zip(axes, selected_names):
        r = results_by_name[name]
        eps = r["strains"]
        K = r["pred_stiffness"]
        ax.plot(eps, K, linewidth=2)
        ax.set_title(f"{name}\nsharpness={r['stiffness_sharpness']:.3e}")
        ax.set_xlabel("Strain")
        ax.set_ylabel(r"Effective stiffness $d^2E/d\epsilon^2$")
        ax.grid(True, alpha=0.3)

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)



def plot_loss_histories(results_by_name, save_path, selected_names=None):
    if selected_names is None:
        selected_names = list(results_by_name.keys())

    n = len(selected_names)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.2 * nrows), squeeze=False)
    axes = axes.ravel()

    for ax, name in zip(axes, selected_names):
        r = results_by_name[name]
        ax.plot(r["train_hist"], label="Train")
        ax.plot(r["test_hist"], label="Test")
        ax.set_yscale("log")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Force MSE")
        ax.grid(True, alpha=0.3)

    for ax in axes[n:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=200)
    plt.close(fig)



def plot_summary_bars(summary_rows, save_path):
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
    axes[1].set_xticklabels(names, rotation=60, ha="right")

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

    problem = load_problem(data_path="slinky_pulling_force_data.npz", test_range=(0.2, 0.8))
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

    # -----------------------------
    # Paper-style selected comparisons
    # -----------------------------
    selected_main = [
        "energy_mlp_L1",
        "energy_icnn_L2",
        "energy_baseline_plus_icnn_L2",
        "stiffness_only_mlp_L1",
        "stiffness_only_mlp_L2",
        "stiffness_baseline_plus_mlp_L1",
        "stiffness_baseline_plus_mlp_L2",
    ]
    selected_main = [name for name in selected_main if name in results_by_name]

    plot_force_displacement_grid(
        results_by_name,
        save_path=os.path.join(fig_outdir, "force_displacement_selected.png"),
        selected_names=selected_main,
    )

    plot_energy_strain_grid(
        results_by_name,
        save_path=os.path.join(fig_outdir, "energy_strain_selected.png"),
        selected_names=selected_main,
    )

    plot_stiffness_strain_grid(
        results_by_name,
        save_path=os.path.join(fig_outdir, "stiffness_strain_selected.png"),
        selected_names=selected_main,
    )

    plot_loss_histories(
        results_by_name,
        save_path=os.path.join(fig_outdir, "loss_histories_selected.png"),
        selected_names=selected_main,
    )

    plot_summary_bars(
        summary_rows,
        save_path=os.path.join(fig_outdir, "summary_bars.png"),
    )

    # Full grids for all cases
    all_names = list(results_by_name.keys())
    plot_force_displacement_grid(
        results_by_name,
        save_path=os.path.join(fig_outdir, "force_displacement_all.png"),
        selected_names=all_names,
    )
    plot_energy_strain_grid(
        results_by_name,
        save_path=os.path.join(fig_outdir, "energy_strain_all.png"),
        selected_names=all_names,
    )
    plot_stiffness_strain_grid(
        results_by_name,
        save_path=os.path.join(fig_outdir, "stiffness_strain_all.png"),
        selected_names=all_names,
    )
    plot_loss_histories(
        results_by_name,
        save_path=os.path.join(fig_outdir, "loss_histories_all.png"),
        selected_names=all_names,
    )

    print("\nDone. Outputs written to:")
    print(f"  {out_root}/summary_table.json")
    print(f"  {fig_outdir}/")
    print(f"  {case_outdir}/")


if __name__ == "__main__":
    main()
