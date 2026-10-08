"""Check and render one homotopy-path sample (.npz from ribbon_data).

    uv run scripts/render_ribbon.py data/<name>/samples/<sample>.npz [--out prefix] [--no-gif]

Prints per-frame checks (clamps, rigid end pose, global force balance, at-rest
metrics) and writes <prefix>.gif (the deforming shell) and <prefix>_final.png.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from ribbon_data import load


def rot_x(a: float) -> np.ndarray:
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def check(s: dict) -> list[tuple]:
    x0, U, tri = s["x0"], s["U"].astype(np.float64), s["elements"]
    X = x0[None] + U
    left, right = s["clamp_left"], s["clamp_right"]
    area = 0.5 * np.linalg.norm(
        np.cross(x0[tri[:, 1]] - x0[tri[:, 0]], x0[tri[:, 2]] - x0[tri[:, 0]]), axis=1
    )
    F_grav = s["density"] * s["thickness"] * area.sum() * np.asarray(s["gravity"])
    b, a = s["width_plus"][-1], s["width_minus"][-1]
    rows = []
    for k, f in enumerate(s["f"]):
        Xr = (
            (x0[right] - s["rp_x0"]) @ rot_x(f * s["theta"]).T
            + s["rp_x0"]
            + f * s["endDisp"]
        )
        F_sum = s["RF"][k].sum(0) + s["rp_RF"][k] + F_grav
        dv = X[k, b] - X[k, a]
        rows.append(
            (
                f,
                np.abs(U[k][left]).max(),
                np.abs(X[k][right] - Xr).max(),
                np.linalg.norm(F_sum) / max(np.linalg.norm(s["rp_RF"][k]), 1e-30),
                s["ALLSE"][k],
                s["ke_over_se"][k],
                s["max_speed"][k],
                np.degrees(np.arctan2(dv[2], dv[1])),
            )
        )
    return rows


def render(s: dict, prefix: str, gif: bool = True) -> None:
    x0, tri, f = s["x0"], s["elements"], s["f"]
    X = x0[None] + s["U"].astype(np.float64)
    area = 0.5 * np.linalg.norm(
        np.cross(x0[tri[:, 1]] - x0[tri[:, 0]], x0[tri[:, 2]] - x0[tri[:, 0]]), axis=1
    )
    dens = s["ELSE"] / area[None]
    lo, hi = np.log10(np.percentile(dens[dens > 0], 1)), np.log10(dens.max())
    pts = np.concatenate([x0, X.reshape(-1, 3)])
    ctr, half = 0.5 * (pts.max(0) + pts.min(0)), 0.5 * np.ptp(pts, 0).max() * 0.8

    fig = plt.figure(figsize=(15, 7.5))
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.7, 1], left=0.0, right=0.97, wspace=0.05, hspace=0.35
    )
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    axE, axF = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    fe = np.concatenate([[0], f])
    axE.plot(fe, np.concatenate([[0], s["ALLSE"]]) * 1e3, "k.-")
    axE.set_ylabel("strain energy [mJ]")
    axE.set_title("settled states along the path (end of each hold)")
    for i, lab in enumerate("xyz"):
        axF.plot(f, s["rp_RF"][:, i], label=f"F{lab} [N]")
    axF.plot(f, s["rp_RM"][:, 0] * 100, "k--", label="Mx [N·cm]")
    axF.set_xlabel("load fraction f")
    axF.set_title("reaction at the moving end")
    axF.legend(fontsize=8, ncol=4)
    for ax in (axE, axF):
        ax.set_xlim(0, 1.02)
    markE, markF = axE.axvline(0, c="r"), axF.axvline(0, c="r")
    cmap = plt.get_cmap("viridis")
    c = s["centerline"]

    def draw(k):  # k = 0 is the flat reference, k >= 1 is frame k-1
        ax3.cla()
        Xk = x0 if k == 0 else X[k - 1]
        fk = 0.0 if k == 0 else f[k - 1]
        col = cmap(
            0.0
            if k == 0
            else (np.log10(np.maximum(dens[k - 1], 10**lo)) - lo) / (hi - lo)
        )
        ax3.plot_trisurf(Xk[:, 0], Xk[:, 1], Xk[:, 2], triangles=tri, color="w")
        poly = ax3.collections[-1]
        poly.set_facecolors(col if np.ndim(col) == 2 else [col])
        poly.set_alpha(0.9)
        poly.set_edgecolor("none")
        ax3.plot(*Xk[c].T, "r-", lw=2)
        ax3.scatter(*Xk[s["clamp_left"]].T, s=1, c="k")
        ax3.scatter(*Xk[s["clamp_right"]].T, s=1, c="m")
        ax3.set_xlim(ctr[0] - half, ctr[0] + half)
        ax3.set_ylim(ctr[1] - half, ctr[1] + half)
        ax3.set_zlim(ctr[2] - half, ctr[2] + half)
        ax3.set_box_aspect((1, 1, 1))
        ax3.view_init(elev=22, azim=-60)
        ax3.set_title(
            f"f = {fk:.2f}   end twist = {np.degrees(fk * s['theta']):.1f}°   "
            f"end disp/L = {np.round(fk * s['endDisp'] / s['l'], 2)}\n"
            "red: centerline · color: log energy density",
            fontsize=10,
        )
        markE.set_xdata([fk])
        markF.set_xdata([fk])

    if gif:
        FuncAnimation(fig, draw, frames=len(f) + 1).save(
            f"{prefix}.gif", writer=PillowWriter(fps=3)
        )
    draw(len(f))
    fig.savefig(f"{prefix}_final.png", dpi=110)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("sample", type=Path)
    p.add_argument("--out", help="output prefix (default: next to the sample)")
    p.add_argument("--no-gif", action="store_true")
    args = p.parse_args()
    s = load(args.sample)
    print(
        f"{s['source']}: {len(s['f'])} frames, completed={s['completed']}, "
        f"endDisp/l={np.round(s['endDisp'] / s['l'], 3)}, theta={np.degrees(s['theta']):.1f} deg"
    )
    print(
        f"{'f':>5} {'L clamp':>9} {'R rigid':>9} {'|sumF|/|Fend|':>13} {'ALLSE [J]':>10} "
        f"{'KE/SE':>8} {'max|v| m/s':>10} {'twist':>7}"
    )
    for r in check(s):
        print(
            "{:5.2f} {:9.1e} {:9.1e} {:13.1e} {:10.3e} {:8.1e} {:10.1e} {:7.1f}".format(
                *r
            )
        )
    prefix = args.out or str(args.sample.with_suffix(""))
    render(s, prefix, gif=not args.no_gif)
    print(f"saved {prefix}{'.gif and ' + prefix if not args.no_gif else ''}_final.png")


if __name__ == "__main__":
    main()
