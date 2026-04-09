"""Plots for multi-seed initialization sensitivity."""

import os

import numpy as np
import matplotlib.pyplot as plt


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
        x[train_mask],
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
        x[test_mask],
        y_true[test_mask],
        linestyle="None",
        marker="o",
        markersize=6,
        markerfacecolor="white",
        markeredgecolor="black",
        label="GT test samples",
        zorder=5,
    )

def get_strain_quantity_arrays(results, quantity_key):
    """
    Collect a strain-based predicted quantity across seeds.

    Assumes each result dict contains:
        - "seed"
        - "strains"
        - quantity_key, e.g. "pred_energy" or "pred_stiffness"

    Returns percentile envelopes after sorting by strain.
    """
    results = sorted(results, key=lambda r: r["seed"])

    # Use strain grid from first result
    eps0 = np.asarray(results[0]["strains"]).reshape(-1)
    order = np.argsort(eps0)
    eps = eps0[order]

    vals = []
    for r in results:
        eps_r = np.asarray(r["strains"]).reshape(-1)
        q_r = np.asarray(r[quantity_key]).reshape(-1)

        # sort each seed by its own strain array
        ord_r = np.argsort(eps_r)
        eps_r = eps_r[ord_r]
        q_r = q_r[ord_r]

        # if strain grids are effectively same, just reorder
        if len(eps_r) == len(eps) and np.allclose(eps_r, eps, atol=1e-10, rtol=1e-8):
            vals.append(q_r)
        else:
            # interpolate to common strain grid if needed
            vals.append(np.interp(eps, eps_r, q_r))

    vals = np.stack(vals, axis=0)

    return {
        "eps": eps,
        "vals": vals,
        "median": np.median(vals, axis=0),
        "q05": np.percentile(vals, 5, axis=0),
        "q25": np.percentile(vals, 25, axis=0),
        "q75": np.percentile(vals, 75, axis=0),
        "q95": np.percentile(vals, 95, axis=0),
        "ymin": np.min(vals, axis=0),
        "ymax": np.max(vals, axis=0),
        "results": results,
        "best": min(results, key=lambda r: r["test_mse"]) if "test_mse" in results[0] else results[0],
    }
def plot_energy_strain_all_seeds(results, save_path, title=None):
    d = get_strain_quantity_arrays(results, "pred_energy")
    if d is None:
        print(f"[plot_energy_strain_all_seeds] Skipping {save_path}: missing 'strains' or 'pred_energy'")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    # all seeds (light)
    for r in d["results"]:
        eps = np.asarray(r["strains"]).reshape(-1)
        e_pred = np.asarray(r["pred_energy"]).reshape(-1)
        order = np.argsort(eps)
        ax.plot(
            eps[order],
            e_pred[order],
            linewidth=1.0,
            alpha=0.18,
            zorder=1,
        )

    # best seed (bold)
    best = d["best"]
    eps_best = np.asarray(best["strains"]).reshape(-1)
    e_best = np.asarray(best["pred_energy"]).reshape(-1)
    order_best = np.argsort(eps_best)

    ax.plot(
        eps_best[order_best],
        e_best[order_best],
        linewidth=2.8,
        label=f"Best seed = {best['seed']} (test MSE = {best['test_mse']:.3e})",
        zorder=4,
    )

    ax.set_xlabel("Strain")
    ax.set_ylabel("Energy")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_stiffness_strain_all_seeds(results, save_path, title=None):
    d = get_strain_quantity_arrays(results, "pred_stiffness")
    if d is None:
        print(f"[plot_stiffness_strain_all_seeds] Skipping {save_path}: missing 'strains' or 'pred_stiffness'")
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    # all seeds (light)
    for r in d["results"]:
        eps = np.asarray(r["strains"]).reshape(-1)
        k_pred = np.asarray(r["pred_stiffness"]).reshape(-1)
        order = np.argsort(eps)
        ax.plot(
            eps[order],
            k_pred[order],
            linewidth=1.0,
            alpha=0.18,
            zorder=1,
        )

    # best seed (bold)
    best = d["best"]
    eps_best = np.asarray(best["strains"]).reshape(-1)
    k_best = np.asarray(best["pred_stiffness"]).reshape(-1)
    order_best = np.argsort(eps_best)

    ax.plot(
        eps_best[order_best],
        k_best[order_best],
        linewidth=2.8,
        label=f"Best seed = {best['seed']} (test MSE = {best['test_mse']:.3e})",
        zorder=4,
    )

    ax.set_xlabel("Strain")
    ax.set_ylabel(r"Effective stiffness $d^2E/d\epsilon^2$")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_minmax_envelope(results, save_path, title=None):
    d = get_case_arrays(results)
    fig, ax = plt.subplots(figsize=(8.5, 5.8))

    ax.fill_between(
        d["x"],
        d["ymin"],
        d["ymax"],
        alpha=0.25,
        label=f"Min-max envelope ({len(d['results'])} seeds)",
        zorder=1,
    )
    ax.plot(
        d["x"],
        d["best"]["pred_force"],
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

    ax.fill_between(d["x"], d["q05"], d["q95"], alpha=0.18, label="5-95 percentile band", zorder=1)
    ax.fill_between(d["x"], d["q25"], d["q75"], alpha=0.30, label="25-75 percentile band", zorder=2)
    ax.plot(
        d["x"],
        d["median_curve"],
        linestyle="--",
        linewidth=2.3,
        label="Median across seeds",
        zorder=3,
    )
    ax.plot(
        d["x"],
        d["best"]["pred_force"],
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
        ax.plot(d["x"], r["pred_force"], linewidth=1.0, alpha=0.18, zorder=1)

    ax.plot(
        d["x"],
        d["median_curve"],
        linestyle="--",
        linewidth=2.2,
        label="Median across seeds",
        zorder=3,
    )
    ax.plot(
        d["x"],
        d["best"]["pred_force"],
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

    train_hist = np.stack([r["train_hist"] for r in results], axis=0)
    test_hist = np.stack([r["test_hist"] for r in results], axis=0)
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    ax1, ax2 = axes

    for r in d["results"]:
        ax1.plot(d["x"], r["pred_force"], linewidth=0.9, alpha=0.15)

    ax1.plot(d["x"], d["median_curve"], "--", linewidth=2.0, label="Median")
    ax1.plot(d["x"], d["best"]["pred_force"], linewidth=2.4, label=f"Best seed={d['best']['seed']}")
    add_gt(ax1, d["x"], d["y_true"], d["train_mask"], d["test_mask"])
    ax1.set_xlabel("Pulled node x")
    ax1.set_ylabel("Force")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.fill_between(epochs, train_p05, train_p95, alpha=0.20, label="Train (5–95%)")
    ax2.plot(epochs, train_median, linestyle="--", linewidth=2.0, label="Train median")

    ax2.fill_between(epochs, test_p05, test_p95, alpha=0.20, label="Test (5–95%)")
    ax2.plot(epochs, test_median, linewidth=2.5, label="Test median")

    ax2.plot(epochs, best_train, linestyle=":", linewidth=2.0, label=f"Best seed train (seed={best['seed']})")
    ax2.plot(epochs, best_test, linestyle="-", linewidth=2.8, label="Best seed test")

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
    train_hist = np.stack([r["train_hist"] for r in results], axis=0)
    test_hist = np.stack([r["test_hist"] for r in results], axis=0)

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

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    ax.fill_between(epochs, train_p05, train_p95, alpha=0.20, label="Train (5–95%)")
    ax.plot(epochs, train_median, linestyle="--", linewidth=2.0, label="Train median")

    ax.fill_between(epochs, test_p05, test_p95, alpha=0.20, label="Test (5–95%)")
    ax.plot(epochs, test_median, linewidth=2.5, label="Test median")

    ax.plot(epochs, best_train, linestyle=":", linewidth=2.0, label=f"Best seed train (seed={best['seed']})")
    ax.plot(epochs, best_test, linestyle="-", linewidth=2.8, label="Best seed test")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_all_init_figures(case_results, fig_dir, case_name: str) -> None:
    plot_minmax_envelope(
        case_results,
        save_path=os.path.join(fig_dir, f"{case_name}_minmax_envelope.png"),
        title=f"{case_name}: min-max seed envelope",
    )
    plot_percentile_envelope(
        case_results,
        save_path=os.path.join(fig_dir, f"{case_name}_percentile_envelope.png"),
        title=f"{case_name}: percentile seed envelope",
    )
    plot_all_seeds(
        case_results,
        save_path=os.path.join(fig_dir, f"{case_name}_all_seed_curves.png"),
        title=f"{case_name}: all seed predictions",
    )
    plot_variance_curve(
        case_results,
        save_path=os.path.join(fig_dir, f"{case_name}_variance_curve.png"),
        title=f"{case_name}: variance across seeds",
    )
    plot_test_mse_hist(
        case_results,
        save_path=os.path.join(fig_dir, f"{case_name}_test_mse_hist.png"),
        title=f"{case_name}: test MSE distribution across seeds",
    )
    plot_summary_panel(
        case_results,
        save_path=os.path.join(fig_dir, f"{case_name}_summary_panel.png"),
        title=f"{case_name}: initialization sensitivity summary",
    )
    plot_loss_envelope(
        case_results,
        save_path=os.path.join(fig_dir, f"{case_name}_loss_envelope.png"),
        title=f"{case_name}: loss across seeds",
    )

    # new
    plot_energy_strain_all_seeds(
        case_results,
        save_path=os.path.join(fig_dir, f"{case_name}_energy_strain_envelope.png"),
        title=f"{case_name}: energy vs strain across seeds",
    )
    plot_stiffness_strain_all_seeds(
        case_results,
        save_path=os.path.join(fig_dir, f"{case_name}_stiffness_strain_envelope.png"),
        title=f"{case_name}: stiffness vs strain across seeds",
    )
