import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# =========================================================
# Helper: linear resampling to fixed number of steps
# =========================================================
def resample_1d(arr, T_new):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == T_new:
        return arr.copy()
    s_old = np.linspace(0.0, 1.0, len(arr))
    s_new = np.linspace(0.0, 1.0, T_new)
    return np.interp(s_new, s_old, arr)


# =========================================================
# Step 1: load + transform raw data only
# =========================================================
def load_and_transform_raw(
    input_json_path,
    drop_first=1,
    reverse_trajectory=True,
):
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

    # ---------------------------------------------------------
    # timestamps
    # ---------------------------------------------------------
    timestamps = []
    for d in data:
        try:
            timestamps.append(datetime.fromisoformat(d["timestamp"]))
        except Exception:
            timestamps.append(None)
    timestamps = np.array(timestamps, dtype=object)

    # ---------------------------------------------------------
    # EE raw
    # ---------------------------------------------------------
    ee_x_raw, ee_y_raw, ee_z_raw = [], [], []
    for d in data:
        ee = d.get("ee_pose", {})
        ee_x_raw.append(ee.get("x_mm", np.nan))
        ee_y_raw.append(ee.get("y_mm", np.nan))
        ee_z_raw.append(ee.get("z_mm", np.nan))

    ee_x_raw = np.asarray(ee_x_raw, dtype=float)
    ee_y_raw = np.asarray(ee_y_raw, dtype=float)
    ee_z_raw = np.asarray(ee_z_raw, dtype=float)

    # ---------------------------------------------------------
    # markers
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # origin from marker_4 mean
    # ---------------------------------------------------------
    ref_name = "marker_4"
    origin_x = np.nanmean(marker_data[ref_name]["x"])
    origin_y = np.nanmean(marker_data[ref_name]["y"])
    origin_z = np.nanmean(marker_data[ref_name]["z"])

    print("Origin from marker_4 mean:")
    print("x0 =", origin_x)
    print("y0 =", origin_y)
    print("z0 =", origin_z)

    # ---------------------------------------------------------
    # center markers
    # ---------------------------------------------------------
    marker_data_centered = {}
    for name in marker_names:
        marker_data_centered[name] = {
            "x": marker_data[name]["x"] - origin_x,
            "y": marker_data[name]["y"] - origin_y,
            "z": marker_data[name]["z"] - origin_z,
        }

    # ---------------------------------------------------------
    # marker transform
    #   new_x = -old_x
    #   new_y =  old_z
    #   new_z = -old_y
    # ---------------------------------------------------------
    for name in marker_names:
        old_x = marker_data_centered[name]["x"].copy()
        old_y = marker_data_centered[name]["y"].copy()
        old_z = marker_data_centered[name]["z"].copy()

        marker_data_centered[name]["x"] = -old_x
        marker_data_centered[name]["y"] = old_z
        marker_data_centered[name]["z"] = -old_y

    # ---------------------------------------------------------
    # EE transform and mm -> m
    # ---------------------------------------------------------
    ee_x = -ee_y_raw / 1000.0
    ee_y = ee_x_raw / 1000.0
    ee_z = ee_z_raw / 1000.0

    # ---------------------------------------------------------
    # align first EE point to transformed marker_0 first point
    # ---------------------------------------------------------
    ee_x = ee_x - ee_x[0] + marker_data_centered["marker_0"]["x"][0]
    ee_y = ee_y - ee_y[0] + marker_data_centered["marker_0"]["y"][0]
    ee_z = ee_z - ee_z[0] + marker_data_centered["marker_0"]["z"][0]

    # ---------------------------------------------------------
    # optional drop first point
    # ---------------------------------------------------------
    if drop_first != 0:
        timestamps = timestamps[drop_first:]
        for name in marker_names:
            marker_data_centered[name]["x"] = marker_data_centered[name]["x"][drop_first:]
            marker_data_centered[name]["y"] = marker_data_centered[name]["y"][drop_first:]
            marker_data_centered[name]["z"] = marker_data_centered[name]["z"][drop_first:]

        ee_x = ee_x[drop_first:]
        ee_y = ee_y[drop_first:]
        ee_z = ee_z[drop_first:]

    # ---------------------------------------------------------
    # optional reverse
    # ---------------------------------------------------------
    if reverse_trajectory:
        timestamps = timestamps[::-1]
        for name in marker_names:
            marker_data_centered[name]["x"] = marker_data_centered[name]["x"][::-1]
            marker_data_centered[name]["y"] = marker_data_centered[name]["y"][::-1]
            marker_data_centered[name]["z"] = marker_data_centered[name]["z"][::-1]

        ee_x = ee_x[::-1]
        ee_y = ee_y[::-1]
        ee_z = ee_z[::-1]

    return {
        "timestamps": timestamps,
        "markers": marker_data_centered,
        "ee_x": ee_x,
        "ee_y": ee_y,
        "ee_z": ee_z,
    }


# =========================================================
# Step 2: split full time series into fixed-length chunks
# =========================================================
def segment_into_fixed_chunks(
    n_samples,
    chunk_len=80,
    stride=None,
    drop_last=True,
):
    """
    Split a long time series into fixed-length chunks.

    Parameters
    ----------
    n_samples : int
        Total number of samples.
    chunk_len : int
        Number of raw samples per chunk.
    stride : int or None
        Step between chunk starts. If None, uses non-overlapping chunks.
    drop_last : bool
        Whether to discard the final incomplete chunk.

    Returns
    -------
    segments : list of (start, end)
    """
    if chunk_len < 2:
        raise ValueError("chunk_len must be at least 2")

    if stride is None:
        stride = chunk_len

    if stride < 1:
        raise ValueError("stride must be at least 1")

    segments = []
    start = 0
    while start < n_samples:
        end = start + chunk_len

        if end <= n_samples:
            segments.append((start, end))
        else:
            if not drop_last and (n_samples - start) >= 2:
                segments.append((start, n_samples))
            break

        start += stride

    if len(segments) == 0:
        raise ValueError(
            f"No chunks created. n_samples={n_samples}, "
            f"chunk_len={chunk_len}, stride={stride}"
        )

    return segments


# =========================================================
# Step 3: build multi-trajectory dataset
# =========================================================
def build_multitraj_directbc_dataset(
    transformed,
    segments,
    output_npz_path=None,
    T_common=40,
):
    marker2_x = transformed["markers"]["marker_2"]["x"]
    marker2_z = transformed["markers"]["marker_2"]["z"]
    ee_x = transformed["ee_x"]
    ee_z = transformed["ee_z"]

    qs_list = []
    xb_list = []

    for (s, e) in segments:
        m2x = marker2_x[s:e]
        m2z = marker2_z[s:e]
        ex = ee_x[s:e]
        ez = ee_z[s:e]

        # resample each chunk to common length
        m2x = resample_1d(m2x, T_common)
        m2z = resample_1d(m2z, T_common)
        ex = resample_1d(ex, T_common)
        ez = resample_1d(ez, T_common)

        # qs = [x0,y0,z0,th0, x1,y1,z1,th1, x2,y2,z2]
        qs_i = np.column_stack([
            np.zeros(T_common),   # x0
            np.zeros(T_common),   # y0
            np.zeros(T_common),   # z0
            np.zeros(T_common),   # th0

            m2x,                  # x1
            np.zeros(T_common),   # y1
            m2z,                  # z1
            np.zeros(T_common),   # th1

            ex,                   # x2
            np.zeros(T_common),   # y2
            ez,                   # z2
        ])

        # xb for idx_b = [0,1,2,3,7,8,9,10]
        xb_i = np.column_stack([
            np.zeros(T_common),   # x0
            np.zeros(T_common),   # y0
            np.zeros(T_common),   # z0
            np.zeros(T_common),   # th0
            np.zeros(T_common),   # th1
            ex,                   # x2
            np.zeros(T_common),   # y2
            ez,                   # z2
        ])

        qs_list.append(qs_i)
        xb_list.append(xb_i)

    if len(qs_list) == 0:
        raise ValueError("No valid chunks found.")

    qs = np.stack(qs_list, axis=0)   # (n_traj, T_common, 11)
    xb = np.stack(xb_list, axis=0)   # (n_traj, T_common, 8)
    idx_b = np.array([0, 1, 2, 3, 7, 8, 9, 10], dtype=int)
    lambdas = np.linspace(0.0, 1.0, T_common)

    if output_npz_path is not None:
        np.savez(
            output_npz_path,
            qs=qs,
            xb=xb,
            idx_b=idx_b,
            lambdas=lambdas,
        )

    return qs, xb, idx_b, lambdas


# =========================================================
# Step 4: split trajectories into train/test
# =========================================================
def save_train_test_split(
    qs,
    xb,
    idx_b,
    lambdas,
    train_ids,
    test_ids,
    train_path,
    test_path,
):
    np.savez(
        train_path,
        qs=qs[train_ids],
        xb=xb[train_ids],
        idx_b=idx_b,
        lambdas=lambdas,
    )

    np.savez(
        test_path,
        qs=qs[test_ids],
        xb=xb[test_ids],
        idx_b=idx_b,
        lambdas=lambdas,
    )


# =========================================================
# Plot: full transformed EE trajectory
# =========================================================
def plot_full_ee_trajectory(
    transformed,
    figsize=(10, 6),
    title="End Effector Trajectory (x vs z)",
    save_path=None,
):
    ee_x = transformed["ee_x"]
    ee_z = transformed["ee_z"]

    plt.figure(figsize=figsize)
    plt.plot(ee_x, ee_z, color="red", linewidth=1.5, label="EE trajectory")
    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =========================================================
# Plot: chunks with IDs
# =========================================================
def plot_segmented_chunks(
    transformed,
    segments,
    figsize=(10, 6),
    title="Detected chunks",
    save_path=None,
):
    ee_x = transformed["ee_x"]
    ee_z = transformed["ee_z"]

    plt.figure(figsize=figsize)
    plt.plot(ee_x, ee_z, color="lightgray", linewidth=1.0, label="Full trajectory")

    for i, (s, e) in enumerate(segments):
        x = ee_x[s:e]
        z = ee_z[s:e]
        plt.plot(x, z, linewidth=2.0)
        mid = len(x) // 2
        plt.text(x[mid], z[mid], str(i), fontsize=8)

    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =========================================================
# Plot: train/test color-coded chunks
# =========================================================
def plot_chunks_train_test(
    transformed,
    segments,
    train_ids=None,
    test_ids=None,
    show_ids=True,
    figsize=(10, 6),
    title="EE Trajectory with Train/Test Chunks",
    save_path=None,
):
    ee_x = transformed["ee_x"]
    ee_z = transformed["ee_z"]

    train_set = set([] if train_ids is None else list(np.asarray(train_ids).astype(int)))
    test_set = set([] if test_ids is None else list(np.asarray(test_ids).astype(int)))

    print("plotting with:")
    print("  train_set =", sorted(train_set))
    print("  test_set  =", sorted(test_set))

    plt.figure(figsize=figsize)
    plt.plot(ee_x, ee_z, color="lightgray", linewidth=1.0, label="Full trajectory")

    train_done = False
    test_done = False

    for i, (s, e) in enumerate(segments):
        x = ee_x[s:e]
        z = ee_z[s:e]

        if i in train_set:
            color = "blue"
            label = "Train" if not train_done else None
            train_done = True
        elif i in test_set:
            color = "red"
            label = "Test" if not test_done else None
            test_done = True
        else:
            color = "black"
            label = None

        plt.plot(x, z, color=color, linewidth=2.0, alpha=0.95, label=label)
        plt.scatter(x[0], z[0], color=color, s=18, marker="o")
        plt.scatter(x[-1], z[-1], color=color, s=18, marker="x")

        if show_ids:
            mid = len(x) // 2
            plt.text(x[mid], z[mid], str(i), fontsize=8, color=color)

    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title(title)
    plt.grid(True)
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =========================================================
# Plot: rod snapshots from qs
# =========================================================
def plot_rod_snapshots_from_qs(
    qs,
    traj_ids=None,
    plot_start=True,
    plot_end=True,
    every_step=None,
    figsize=(10, 6),
    title="Rod snapshots from qs",
    save_path=None,
):
    """
    qs shape: (n_traj, T, 11)

    DOF layout:
    [x0,y0,z0,th0, x1,y1,z1,th1, x2,y2,z2]
    We plot x-z only.
    """
    qs = np.asarray(qs)
    n_traj, T, dof = qs.shape
    if dof != 11:
        raise ValueError(f"Expected dof=11, got {dof}")

    if traj_ids is None:
        traj_ids = np.arange(n_traj)
    traj_ids = np.asarray(traj_ids, dtype=int)

    plt.figure(figsize=figsize)

    for tid in traj_ids:
        steps = np.arange(T)
        if every_step is not None and every_step > 1:
            steps = steps[::every_step]

        for t in steps:
            q = qs[tid, t]

            x = np.array([q[0], q[4], q[8]])
            z = np.array([q[2], q[6], q[10]])

            alpha = 0.18 if (t != 0 and t != T - 1) else 1.0
            plt.plot(x, z, "-o", alpha=alpha)

        if plot_start:
            q0 = qs[tid, 0]
            x0 = np.array([q0[0], q0[4], q0[8]])
            z0 = np.array([q0[2], q0[6], q0[10]])
            plt.plot(x0, z0, "-o", linewidth=2.5, label=f"traj {tid} start")

        if plot_end:
            q1 = qs[tid, -1]
            x1 = np.array([q1[0], q1[4], q1[8]])
            z1 = np.array([q1[2], q1[6], q1[10]])
            plt.plot(x1, z1, "-o", linewidth=2.5, label=f"traj {tid} end")

    plt.xlabel("x (m)")
    plt.ylabel("z (m)")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.legend(fontsize=8, ncol=2)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# =========================================================
# Main extraction function
# =========================================================
def extract_multitraj_dataset(
    input_json_path,
    output_all_path,
    output_train_path,
    output_test_path,
    T_common=40,
    plot=True,
    reverse_trajectory=True,
    drop_first=1,
    chunk_len=80,
    stride=None,
    drop_last=True,
    train_ids=None,   # ✅ NEW
    test_ids=None,    # ✅ NEW
):
    transformed = load_and_transform_raw(
        input_json_path,
        drop_first=drop_first,
        reverse_trajectory=reverse_trajectory,
    )

    n_samples = len(transformed["ee_x"])

    segments = segment_into_fixed_chunks(
        n_samples=n_samples,
        chunk_len=chunk_len,
        stride=stride,
        drop_last=drop_last,
    )

    print("Detected chunks:")
    for i, (s, e) in enumerate(segments):
        print(f"  traj {i}: [{s}, {e}) length={e-s}")

    qs, xb, idx_b, lambdas = build_multitraj_directbc_dataset(
        transformed,
        segments,
        output_npz_path=output_all_path,
        T_common=T_common,
    )

    n_traj = qs.shape[0]
    all_ids = np.arange(n_traj)

    # ---------------------------------------------------------
    # ✅ Custom split logic
    # ---------------------------------------------------------
    if train_ids is None and test_ids is None:
        # default split
        train_ids = all_ids[::2]
        test_ids = all_ids[1::2]

    elif train_ids is not None and test_ids is None:
        train_ids = np.asarray(train_ids, dtype=int)
        test_ids = np.setdiff1d(all_ids, train_ids)

    elif train_ids is None and test_ids is not None:
        test_ids = np.asarray(test_ids, dtype=int)
        train_ids = np.setdiff1d(all_ids, test_ids)

    else:
        train_ids = np.asarray(train_ids, dtype=int)
        test_ids = np.asarray(test_ids, dtype=int)

    # ---------------------------------------------------------
    # Safety checks
    # ---------------------------------------------------------
    if np.intersect1d(train_ids, test_ids).size > 0:
        raise ValueError("train_ids and test_ids overlap!")

    if len(train_ids) == 0 or len(test_ids) == 0:
        raise ValueError("Train or test set is empty!")

    print("n_traj =", n_traj)
    print("train_ids =", train_ids)
    print("test_ids  =", test_ids)

    # ---------------------------------------------------------
    # Save datasets
    # ---------------------------------------------------------
    save_train_test_split(
        qs, xb, idx_b, lambdas,
        train_ids=train_ids,
        test_ids=test_ids,
        train_path=output_train_path,
        test_path=output_test_path,
    )

    print("qs shape =", qs.shape)
    print("xb shape =", xb.shape)

    # ---------------------------------------------------------
    # Plot
    # ---------------------------------------------------------
    if plot:
        plot_full_ee_trajectory(transformed)
        plot_segmented_chunks(transformed, segments)
        plot_chunks_train_test(
            transformed,
            segments,
            train_ids=train_ids,
            test_ids=test_ids,
            show_ids=True,
        )

    return {
        "qs": qs,
        "xb": xb,
        "idx_b": idx_b,
        "lambdas": lambdas,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "segments": segments,
    }


# =========================================================
# Example usage
# =========================================================
if __name__ == "__main__":
    result = extract_multitraj_dataset(
        input_json_path="your_input.json",
        output_all_path="all_chunks.npz",
        output_train_path="train_chunks.npz",
        output_test_path="test_chunks.npz",
        T_common=40,
        plot=True,
        reverse_trajectory=False,
        drop_first=1,
        chunk_len=80,
        stride=80,      # non-overlapping chunks
        drop_last=True,
    )