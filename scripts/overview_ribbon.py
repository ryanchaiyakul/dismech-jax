"""Grid overview of several homotopy-path samples in a dataset.

    uv run scripts/overview_ribbon.py data/<name> [--n 12] [--out overview.png]

Each panel shows the final shell (color: log strain-energy density) with the
centerline at every frame of the path overlaid (color: load fraction f).
Samples are picked at evenly spaced quantiles of final strain energy.
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ribbon_data import centerline_and_frame, load, sample_id


def pick(root: Path, n: int) -> list[dict]:
    rows = [r for r in csv.DictReader(open(root / "index.csv")) if r["completed"] == "True"]
    rows.sort(key=lambda r: float(r["final_ALLSE"]))
    idx = np.unique(np.round(np.linspace(0, len(rows) - 1, n)).astype(int))
    return [rows[i] for i in idx]


def panel(ax, s: dict) -> None:
    x0, tri = s["x0"], s["elements"]
    X = x0 + s["U"][-1].astype(np.float64)
    area = 0.5 * np.linalg.norm(
        np.cross(x0[tri[:, 1]] - x0[tri[:, 0]], x0[tri[:, 2]] - x0[tri[:, 0]]), axis=1
    )
    dens = np.log10(np.maximum(s["ELSE"][-1] / area, 1e-12))
    lo, hi = np.percentile(dens, [2, 100])
    ax.plot_trisurf(X[:, 0], X[:, 1], X[:, 2], triangles=tri, color="w")
    poly = ax.collections[-1]
    poly.set_facecolors(plt.get_cmap("viridis")(np.clip((dens - lo) / (hi - lo), 0, 1)))
    poly.set_alpha(0.75)
    poly.set_edgecolor("none")
    c, _ = centerline_and_frame(s)
    c = np.concatenate([x0[s["centerline"]][None], c])
    fs = np.concatenate([[0.0], s["f"]])
    cm = plt.get_cmap("autumn_r")
    for k in range(len(c)):
        ax.plot(*c[k].T, c=cm(fs[k]), lw=1.0 if k < len(c) - 1 else 2.0)
    pts = np.concatenate([x0, X])
    ctr, half = 0.5 * (pts.max(0) + pts.min(0)), 0.5 * np.ptp(pts, 0).max() * 0.75
    for i, setlim in enumerate((ax.set_xlim, ax.set_ylim, ax.set_zlim)):
        setlim(ctr[i] - half, ctr[i] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-60)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    d = np.round(s["endDisp"] / s["l"], 2)
    ax.set_title(
        f"{sample_id(s)}\nd/L={d}  θ={np.degrees(s['theta']):.0f}°\n"
        f"SE={s['ALLSE'][-1] * 1e3:.2f} mJ",
        fontsize=8,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dataset", type=Path)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    rows = pick(args.dataset, args.n)
    nr = -(-len(rows) // args.cols)
    fig = plt.figure(figsize=(4 * args.cols, 4.2 * nr))
    for i, r in enumerate(rows):
        ax = fig.add_subplot(nr, args.cols, i + 1, projection="3d")
        panel(ax, load(args.dataset / "samples" / r["sample"]))
    fig.suptitle(
        f"{args.dataset.name}: final shells (color: log energy density), "
        "centerlines along path (yellow f=0 → red f=1)"
    )
    fig.tight_layout()
    out = args.out or args.dataset / "overview.png"
    fig.savefig(out, dpi=100)
    print(f"saved {out}")
    for r in rows:
        print(r["sample"])


if __name__ == "__main__":
    main()
