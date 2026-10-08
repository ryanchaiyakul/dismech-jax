"""RibbonFEMData homotopy-path samples: JSON (from the cluster) -> NPZ, and loading.

One `.npz` per path. Each path is a sequence of settled shell states at load
fractions f = 1/N, ..., 1 (f = 0, the flat ribbon, is not stored: U = 0 there).
The right clamp moves rigidly: rotation f*theta about the x axis through `rp_x0`
(the centerline), then translation f*endDisp. The left clamp is fixed.

Arrays (K frames, n shell nodes, m triangles):
    x0 (n,3)            reference node positions [m] (includes a tiny z imperfection)
    elements (m,3)      triangle connectivity as row indices into x0
    f (K,)              load fraction of each frame
    U (K,n,3)           displacements; deformed positions are x0 + U
    RF (K,n,3)          reaction forces at constrained nodes [N]
    ELSE (K,m)          element strain energy [J]
    rp_U, rp_UR, rp_RF, rp_RM (K,3)   moving-end displacement, rotation, force, moment
    ALLSE, ALLKE, ALLWK, ALLVD (K,)   whole-model energies at the end of each hold
    ke_over_se, max_speed (K,)        at-rest metrics (kinetic/strain energy, max nodal speed)
    centerline (r,)     node indices along y = w/2, sorted by x
    width_minus, width_plus (r,)      node indices two rows either side, same x
    clamp_left, clamp_right (n,) bool clamped node masks
Scalars: l, w, thickness, young, poisson, density, n_rows, clamp_fraction,
theta, hold_time, completed; endDisp (3,), gravity (3,), rp_x0 (3,), source (str).
"""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

DENSITY = 1000.0
GRAVITY = (0.0, 0.0, 1.0e-2)  # *DLOAD GRAV in topLevel_path.inp
ENERGIES = ("ALLSE", "ALLKE", "ALLWK", "ALLVD")


def _rows(x0: np.ndarray, w: float, n_rows: int, k: int) -> np.ndarray:
    rows = np.round((x0[:, 1] - x0[:, 1].min()) / (w / n_rows)).astype(int)
    i = np.where(rows == k)[0]
    return i[np.argsort(x0[i, 0])]


def json_to_npz(src: Path, dst: Path) -> dict:
    """Convert one `<job>_path.json` to `.npz`. Returns a summary row for the index."""
    d = json.loads(Path(src).read_text())
    frames = d["frames"]
    x0 = np.asarray(d["x0"], np.float64)
    idx = {lab: i for i, lab in enumerate(d["node_labels"])}
    elements = np.asarray([[idx[n] for n in e] for e in d["elements"]], np.int32)
    length, w, n_rows, clamp = d["l"], d["w"], int(d["n_rows"]), d["clamp_fraction"]

    def stack(key, dtype=np.float32):
        return np.asarray([fr[key] for fr in frames], dtype)

    def rp(key):
        return np.asarray([fr["rp"][key] or [np.nan] * 3 for fr in frames], np.float64)

    energy = {
        k: np.asarray([fr["energy"].get(k, np.nan) for fr in frames], np.float64)
        for k in ENERGIES
    }
    with np.errstate(divide="ignore", invalid="ignore"):
        ke_over_se = energy["ALLKE"] / energy["ALLSE"]
    out: dict[str, Any] = dict(
        x0=x0,
        elements=elements,
        node_labels=np.asarray(d["node_labels"], np.int32),
        f=stack("f", np.float64),
        U=stack("U"),
        RF=stack("RF"),
        ELSE=stack("ELSE"),
        rp_U=rp("U"),
        rp_UR=rp("UR"),
        rp_RF=rp("RF"),
        rp_RM=rp("RM"),
        **energy,
        ke_over_se=ke_over_se,
        max_speed=stack("max_speed", np.float64),
        centerline=_rows(x0, w, n_rows, n_rows // 2),
        width_minus=_rows(x0, w, n_rows, n_rows // 2 - 2),
        width_plus=_rows(x0, w, n_rows, n_rows // 2 + 2),
        clamp_left=np.abs(x0[:, 0]) <= length * clamp,
        clamp_right=np.abs(x0[:, 0] - length) <= length * clamp,
        l=length,
        w=w,
        thickness=d["thickness"],
        young=d["young"],
        poisson=d["poisson"],
        density=DENSITY,
        gravity=np.asarray(GRAVITY),
        n_rows=n_rows,
        clamp_fraction=clamp,
        endDisp=np.asarray(d["endDisp"], np.float64),
        theta=float(d["theta"]),
        hold_time=float(d.get("hold_time", np.nan)),
        completed=bool(d["completed"]),
        rp_x0=np.asarray(d["rp_x0"], np.float64),
        source=Path(src).name,
    )
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    tmp.replace(dst)
    return summary(out, dst.name)


def summary(s: dict, name: str) -> dict:
    ke = np.nan_to_num(np.asarray(s["ke_over_se"], float), nan=np.inf)
    return {
        "sample": name,
        "frames": len(s["f"]),
        "completed": bool(s["completed"]),
        "endDisp_x": float(s["endDisp"][0]),
        "endDisp_y": float(s["endDisp"][1]),
        "endDisp_z": float(s["endDisp"][2]),
        "theta": float(s["theta"]),
        "max_ke_over_se": float(ke.max()) if len(ke) else np.inf,
        "max_speed": float(np.max(s["max_speed"])) if len(s["max_speed"]) else np.inf,
        "final_ALLSE": float(s["ALLSE"][-1]) if len(s["ALLSE"]) else np.nan,
    }


def load(path: Path) -> dict:
    """Load one sample as a dict of numpy arrays / python scalars."""
    with np.load(path, allow_pickle=False) as z:
        return {k: (z[k].item() if z[k].ndim == 0 else z[k]) for k in z.files}


def sample_id(s: dict) -> str:
    """Sample id from the source name: `job-ribbon-[<cluster job>-]<id>-l-...`."""
    return re.search(r"-(\d+)-l-", s["source"]).group(1)


def centerline_and_frame(s: dict) -> tuple[np.ndarray, np.ndarray]:
    """Deformed centerline positions (K, r, 3) and unit width directions (K, r, 3)."""
    X = s["x0"][None] + s["U"].astype(np.float64)
    c = X[:, s["centerline"]]
    d = X[:, s["width_plus"]] - X[:, s["width_minus"]]
    return c, d / np.linalg.norm(d, axis=-1, keepdims=True)
