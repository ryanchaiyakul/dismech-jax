import json
import numpy as np
from datetime import datetime


def extract_directbc_dataset(
    input_json_path,
    output_npz_path,
    drop_first_point=True,
    reverse_trajectory=True,
):
    """
    Build direct-BC dataset with format:
        qs      : (n_traj, T, dof)
        xb      : (n_traj, T, n_b)
        idx_b   : (n_b,)
        lambdas : (T,)

    Desired DOF layout:
        qs = [marker4_x, marker4_y, marker4_z, 0,
              marker2_x, marker2_y, marker2_z, 0,
              ee_x, ee_y, ee_z]

    Boundary condition:
        xb = [ee_x, ee_y, ee_z]

    Assumes a single trajectory in the JSON file, so n_traj = 1.
    """

    # =========================================================
    # Load JSON lines
    # =========================================================
    data = []
    with open(input_json_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if len(data) == 0:
        raise ValueError("No valid JSON samples found.")

    print(f"Loaded {len(data)} samples")

    # =========================================================
    # Parse timestamps (optional, not used for saving)
    # =========================================================
    timestamps = []
    for d in data:
        try:
            timestamps.append(datetime.fromisoformat(d["timestamp"]))
        except Exception:
            timestamps.append(None)
    timestamps = np.array(timestamps)

    # =========================================================
    # Extract raw EE pose (in mm)
    # =========================================================
    ee_x_raw, ee_y_raw, ee_z_raw = [], [], []
    for d in data:
        ee = d.get("ee_pose", {})
        ee_x_raw.append(ee.get("x_mm", np.nan))
        ee_y_raw.append(ee.get("y_mm", np.nan))
        ee_z_raw.append(ee.get("z_mm", np.nan))

    ee_x_raw = np.asarray(ee_x_raw, dtype=float)
    ee_y_raw = np.asarray(ee_y_raw, dtype=float)
    ee_z_raw = np.asarray(ee_z_raw, dtype=float)

    # =========================================================
    # Extract marker data
    # =========================================================
    marker_names = set()
    for d in data:
        marker_names.update(d.get("markers", {}).keys())
    marker_names = sorted(marker_names)

    required_markers = ["marker_4", "marker_2", "marker_0"]
    for m in required_markers:
        if m not in marker_names:
            raise ValueError(f"Required marker '{m}' not found in data.")

    marker_data = {
        name: {"x": [], "y": [], "z": []}
        for name in marker_names
    }

    for d in data:
        markers = d.get("markers", {})
        for name in marker_names:
            if name in markers:
                marker_data[name]["x"].append(markers[name].get("x", np.nan))
                marker_data[name]["y"].append(markers[name].get("y", np.nan))
                marker_data[name]["z"].append(markers[name].get("z", np.nan))
            else:
                marker_data[name]["x"].append(np.nan)
                marker_data[name]["y"].append(np.nan)
                marker_data[name]["z"].append(np.nan)

    for name in marker_names:
        for c in ["x", "y", "z"]:
            marker_data[name][c] = np.asarray(marker_data[name][c], dtype=float)

    # =========================================================
    # Define origin from marker_4 mean position
    # =========================================================
    ref_name = "marker_4"
    origin_x = np.nanmean(marker_data[ref_name]["x"])
    origin_y = np.nanmean(marker_data[ref_name]["y"])
    origin_z = np.nanmean(marker_data[ref_name]["z"])

    print("Origin from marker_4 mean:")
    print("x0 =", origin_x)
    print("y0 =", origin_y)
    print("z0 =", origin_z)

    # =========================================================
    # Remove global offset from all markers
    # =========================================================
    marker_data_centered = {}
    for name in marker_names:
        marker_data_centered[name] = {
            "x": marker_data[name]["x"] - origin_x,
            "y": marker_data[name]["y"] - origin_y,
            "z": marker_data[name]["z"] - origin_z,
        }

    # =========================================================
    # Coordinate transform
    #
    # Based on your pipeline intent:
    #   new_x = -old_x
    #   new_z = -old_y
    #   new_y = old_z
    #
    # This avoids the overwrite bug in the original code.
    # =========================================================
    for name in marker_names:
        old_x = marker_data_centered[name]["x"].copy()
        old_y = marker_data_centered[name]["y"].copy()
        old_z = marker_data_centered[name]["z"].copy()

        marker_data_centered[name]["x"] = -old_x
        marker_data_centered[name]["y"] = old_z
        marker_data_centered[name]["z"] = -old_y

    # =========================================================
    # EE transform
    #
    # Your pipeline used:
    #   ee_x = -ee_y_raw
    #   ee_z = ee_z_raw
    #   ee_y = ee_x_raw
    #
    # Keep that convention, then convert mm -> m
    # =========================================================
    ee_x = -ee_y_raw / 1000.0
    ee_y =  ee_x_raw / 1000.0
    ee_z =  ee_z_raw / 1000.0

    # =========================================================
    # Align EE trajectory so first EE point matches transformed marker_0 first point
    # =========================================================
    ee_x = ee_x - ee_x[0] + marker_data_centered["marker_0"]["x"][0]
    ee_y = ee_y - ee_y[0] + marker_data_centered["marker_0"]["y"][0]
    ee_z = ee_z - ee_z[0] + marker_data_centered["marker_0"]["z"][0]

    # =========================================================
    # Remove first point to avoid initial jump
    # =========================================================
    if drop_first_point:
        for name in marker_names:
            marker_data_centered[name]["x"] = marker_data_centered[name]["x"][1:]
            marker_data_centered[name]["y"] = marker_data_centered[name]["y"][1:]
            marker_data_centered[name]["z"] = marker_data_centered[name]["z"][1:]

        ee_x = ee_x[1:]
        ee_y = ee_y[1:]
        ee_z = ee_z[1:]

    # =========================================================
    # Reverse trajectory direction if desired
    # =========================================================
    if reverse_trajectory:
        for name in marker_names:
            marker_data_centered[name]["x"] = marker_data_centered[name]["x"][::-1]
            marker_data_centered[name]["y"] = marker_data_centered[name]["y"][::-1]
            marker_data_centered[name]["z"] = marker_data_centered[name]["z"][::-1]

        ee_x = ee_x[::-1]
        ee_y = ee_y[::-1]
        ee_z = ee_z[::-1]

    # =========================================================
    # Sanity check lengths
    # =========================================================
    T = len(ee_x)
    for m in required_markers:
        for c in ["x", "y", "z"]:
            if len(marker_data_centered[m][c]) != T:
                raise ValueError(f"Length mismatch in {m}_{c}")
    
    # =========================================================
    # Build qs
    # DOF layout:
    # [x0,y0,z0,th0, x1,y1,z1,th1, x2,y2,z2]
    #
    # node 0 fixed at origin
    # node 1 from marker_2, planar => y=0
    # node 2 from EE, planar => y=0
    # theta DOFs = 0
    # =========================================================
    T = len(ee_x)

    qs = np.column_stack([
        np.zeros(T),  # x0
        np.zeros(T),  # y0
        np.zeros(T),  # z0
        np.zeros(T),  # th0

        marker_data_centered["marker_2"]["x"],  # x1
        np.zeros(T),                            # y1
        marker_data_centered["marker_2"]["z"],  # z1
        np.zeros(T),                            # th1

        ee_x,                                   # x2
        np.zeros(T),                            # y2
        ee_z,                                   # z2
    ])  # shape (T, 11)


    # =========================================================
    # Direct boundary conditions on all fixed/constrained DOFs
    # idx_b = [0,1,2,3,7,8,9,10]
    #
    # xb entries correspond in this exact order:
    # [x0, y0, z0, th0, th1, x2, y2, z2]
    # =========================================================
    xb = np.column_stack([
        np.zeros(T),  # x0
        np.zeros(T),  # y0
        np.zeros(T),  # z0
        np.zeros(T),  # th0
        np.zeros(T),  # th1
        ee_x,         # x2
        np.zeros(T),  # y2
        ee_z,         # z2
    ])  # shape (T, 8)

    idx_b = np.array([0, 1, 2, 3, 7, 8, 9, 10], dtype=int)


    # =========================================================
    # Lambdas
    # =========================================================
    lambdas = np.linspace(0.0, 1.0, T)


    # =========================================================
    # Add trajectory dimension
    # =========================================================
    qs = qs[None, :, :]   # (1, T, 11)
    xb = xb[None, :, :]   # (1, T, 8)


    # =========================================================
    # Save
    # =========================================================
    np.savez(
        output_npz_path,
        qs=qs,
        xb=xb,
        idx_b=idx_b,
        lambdas=lambdas,
    )

    print("qs shape:", qs.shape)
    print("xb shape:", xb.shape)
    print("idx_b:", idx_b)
    print("lambdas shape:", lambdas.shape)

    return qs, xb, idx_b, lambdas

import numpy as np
import matplotlib.pyplot as plt


def plot_qs_trajectories(qs, traj_idx=0, title=None, show_points=True):
    """
    Plot the final planar trajectories encoded in qs.

    Expected DOF layout for one trajectory:
        [x0, y0, z0, th0, x1, y1, z1, th1, x2, y2, z2]

    We only visualize x-z since the motion is planar and y is zero.

    Parameters
    ----------
    qs : array-like
        Shape (n_traj, T, 11) or (T, 11)
    traj_idx : int
        Which trajectory to plot if qs has shape (n_traj, T, 11)
    title : str or None
        Plot title
    show_points : bool
        Whether to mark sampled time points
    """
    qs = np.asarray(qs)

    if qs.ndim == 2:
        q = qs
    elif qs.ndim == 3:
        q = qs[traj_idx]
    else:
        raise ValueError("qs must have shape (T,11) or (n_traj,T,11)")

    if q.shape[1] != 11:
        raise ValueError(f"Expected qs last dimension = 11, got {q.shape[1]}")

    # Extract node coordinates
    x0, z0 = q[:, 0], q[:, 2]
    x1, z1 = q[:, 4], q[:, 6]
    x2, z2 = q[:, 8], q[:, 10]

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Trajectories
    ax.plot(x0, z0, linewidth=2.0, label="Node 0 (fixed)")
    ax.plot(x1, z1, linewidth=2.0, label="Node 1 (marker_2)")
    ax.plot(x2, z2, linewidth=2.5, label="Node 2 (end effector)")

    if show_points:
        ax.scatter(x0, z0, s=18)
        ax.scatter(x1, z1, s=18)
        ax.scatter(x2, z2, s=18)

    # Mark start and end
    ax.scatter(x0[0], z0[0], s=80, marker="o", label="Start")
    ax.scatter(x2[-1], z2[-1], s=90, marker="x", label="End")

    # Draw rod shapes at first and last step
    ax.plot([x0[0], x1[0], x2[0]], [z0[0], z1[0], z2[0]], "--", linewidth=1.5, alpha=0.8)
    ax.plot([x0[-1], x1[-1], x2[-1]], [z0[-1], z1[-1], z2[-1]], "--", linewidth=1.5, alpha=0.8)

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    if title is None:
        title = f"Planar trajectories for qs (trajectory {traj_idx})"
    ax.set_title(title)

    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_qs_snapshots(qs, traj_idx=0, title=None, every_step=1):
    """
    Plot the rod configuration at multiple time steps.
    """
    qs = np.asarray(qs)

    if qs.ndim == 2:
        q = qs
    elif qs.ndim == 3:
        q = qs[traj_idx]
    else:
        raise ValueError("qs must have shape (T,11) or (n_traj,T,11)")

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    for t in range(0, q.shape[0], every_step):
        x = [q[t, 0], q[t, 4], q[t, 8]]
        z = [q[t, 2], q[t, 6], q[t, 10]]
        ax.plot(x, z, "-o", alpha=0.25)

    # Highlight first and last
    ax.plot([q[0, 0], q[0, 4], q[0, 8]],
            [q[0, 2], q[0, 6], q[0, 10]],
            "-o", linewidth=2.5, label="Start")

    ax.plot([q[-1, 0], q[-1, 4], q[-1, 8]],
            [q[-1, 2], q[-1, 6], q[-1, 10]],
            "-o", linewidth=2.5, label="End")

    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    if title is None:
        title = f"Rod snapshots from qs (trajectory {traj_idx})"
    ax.set_title(title)

    ax.legend()
    plt.tight_layout()
    plt.show()