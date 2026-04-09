"""Plots for single-seed experiment-data ablations."""

import numpy as np
import matplotlib.pyplot as plt


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
        e_pred = np.asarray(r["pred_energy"])
        order = np.argsort(eps)
        style = _style_for_index(i)
        ax.plot(
            eps[order],
            e_pred[order],
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
        k_pred = np.asarray(r["pred_stiffness"])
        order = np.argsort(eps)
        style = _style_for_index(i)
        ax.plot(
            eps[order],
            k_pred[order],
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
