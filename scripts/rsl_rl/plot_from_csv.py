# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause
"""Plot per-step time-series from the CSV exported by play_p2p_time_single3_all.py.

The CSV schema (one row per step):
  episode, time_s, err_x_cm, err_y_cm, err_yaw_deg, height_cm, roll_deg, pitch_deg,
  noise_model, noise_k, sensor_noise, sensor_k

Usage:
  # one figure per episode (like the in-sim plot), saved as PNGs:
  python plot_from_csv.py --csv drift_k2.csv

  # only specific episodes:
  python plot_from_csv.py --csv drift_k2.csv --episodes 0 3 5

  # overlay all episodes' distance error on a single axis:
  python plot_from_csv.py --csv drift_k2.csv --overlay

No Isaac / torch needed -- pure numpy + matplotlib.
"""

import argparse
import csv
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path):
    """Return dict: episode -> dict of column arrays, plus the noise meta string."""
    episodes = defaultdict(lambda: defaultdict(list))
    ep_target_time = {}  # episode -> target_time_s (if present)
    meta = ""
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ep = int(row["episode"])
            episodes[ep]["time"].append(float(row["time_s"]))
            episodes[ep]["err_x"].append(float(row["err_x_cm"]))
            episodes[ep]["err_y"].append(float(row["err_y_cm"]))
            episodes[ep]["err_yaw"].append(float(row["err_yaw_deg"]))
            episodes[ep]["height"].append(float(row["height_cm"]))
            episodes[ep]["roll"].append(float(row["roll_deg"]))
            episodes[ep]["pitch"].append(float(row["pitch_deg"]))
            if "target_time_s" in row and row["target_time_s"] not in ("", None):
                ep_target_time[ep] = float(row["target_time_s"])
            if not meta:
                nm = row.get("noise_model", "?")
                nk = row.get("noise_k", "?")
                sn = row.get("sensor_noise", "none")
                sk = row.get("sensor_k", "0")
                meta = f"pose={nm} k={nk} | sensor={sn} k={sk}"
    # to numpy
    out = {}
    for ep, cols in episodes.items():
        out[ep] = {k: np.array(v) for k, v in cols.items()}
        if ep in ep_target_time:
            out[ep]["target_time"] = ep_target_time[ep]
    return out, meta


def plot_one_episode(ep, d, meta, out_path, dist_target_time=None):
    """Two-panel figure for a single episode: position/height, and orientation."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # vertical line at the target time (if present in the data)
    tt = d.get("target_time", None)

    ax1.plot(d["time"], d["err_x"], label="Error X (cm)", linewidth=2)
    ax1.plot(d["time"], d["err_y"], label="Error Y (cm)", linewidth=2)
    ax1.plot(d["time"], d["height"], label="Body Height (cm)", linewidth=2, color="green")
    if tt is not None:
        ax1.axvline(tt, color="red", linestyle=":", linewidth=2, label=f"Target time ({tt:.2f}s)")
    ax1.axhline(0, color="black", linewidth=1)
    ax1.set_ylabel("cm")
    ax1.set_title(f"Episode {ep} — Position error & body height\n{meta}", fontsize=12)
    ax1.legend(loc="best"); ax1.grid(alpha=0.3)

    ax2.plot(d["time"], d["err_yaw"], label="Error Yaw (deg)", linewidth=2, linestyle="--")
    ax2.plot(d["time"], d["roll"], label="Roll (deg)", linewidth=2, color="purple")
    ax2.plot(d["time"], d["pitch"], label="Pitch (deg)", linewidth=2, color="brown")
    if tt is not None:
        ax2.axvline(tt, color="red", linestyle=":", linewidth=2)
    ax2.axhline(0, color="black", linewidth=1)
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("degrees")
    ax2.set_title("Yaw error, roll, pitch")
    ax2.legend(loc="best"); ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def plot_overlay(episodes, meta, out_path):
    """Overlay every episode's distance error (sqrt(x^2+y^2)) on one axis."""
    fig, ax = plt.subplots(figsize=(11, 6))
    for ep in sorted(episodes):
        d = episodes[ep]
        dist = np.hypot(d["err_x"], d["err_y"])
        ax.plot(d["time"], dist, linewidth=1, alpha=0.7, label=f"ep {ep}")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("distance error (cm)")
    ax.set_title(f"Distance error over time — all episodes\n{meta}")
    ax.grid(alpha=0.3)
    if len(episodes) <= 12:
        ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV exported by the play script.")
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="Only plot these episode indices (default: all).")
    ap.add_argument("--overlay", action="store_true",
                    help="Also make a single overlay plot of all episodes' distance error.")
    ap.add_argument("--out_dir", default=None,
                    help="Directory for PNGs (default: alongside the CSV).")
    args = ap.parse_args()

    episodes, meta = load_csv(args.csv)
    if not episodes:
        print(f"[plot] no data in {args.csv}")
        return

    base = os.path.splitext(os.path.basename(args.csv))[0]
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)

    want = args.episodes if args.episodes is not None else sorted(episodes)
    for ep in want:
        if ep not in episodes:
            print(f"[plot] episode {ep} not in CSV; skipping")
            continue
        out_path = os.path.join(out_dir, f"{base}_episode_{ep:03d}.png")
        plot_one_episode(ep, episodes[ep], meta, out_path)

    if args.overlay:
        plot_overlay(episodes, meta, os.path.join(out_dir, f"{base}_overlay.png"))

    print(f"[plot] done ({len(want)} episode figures).")


if __name__ == "__main__":
    main()
