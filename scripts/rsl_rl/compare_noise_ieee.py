# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause
"""Compare CLEAN vs NOISED P2P results (paired, same commands).

Both CSVs share the schema written by collect_p2p_noise.py:
  cmd_x, cmd_y, cmd_yaw, target_time, err_x, err_y, err_yaw, noise_model, noise_k

Because episode i used the identical command in both runs (clean and noised),
this is a PAIRED comparison: how much does estimator noise on the pose-error
observation degrade the closed-loop P2P policy's final accuracy?

Usage:
  python compare_noise.py --clean results_p2p_clean.csv --noise results_p2p_gaussian_k1.csv \
      --out_prefix noise_gaussian_k1
"""

import argparse
import csv
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        rows = [row for row in r if row]
    # tolerate 7-col (no meta), 9-col (pose-noise meta), or 11-col (+sensor meta)
    data = np.array([[float(x) for x in row[:7]] for row in rows])
    r0 = rows[0]
    pose_model = r0[7] if len(r0) > 7 else "?"
    pose_k = r0[8] if len(r0) > 8 else "?"
    sensor_noise = r0[9] if len(r0) > 9 else "none"
    sensor_k = r0[10] if len(r0) > 10 else "0"
    # Build a label that reflects whatever noise was ACTUALLY active:
    #  - sensor noise active -> "<channels> sk<k>"  (e.g. "all sk3")
    #  - pose-error noise active -> "<model> k<k>"  (e.g. "gaussian k1")
    #  - neither -> "clean"
    def _active(v):
        return v not in ("none", "0", "0.000", "?", "")
    if _active(sensor_noise) and _active(sensor_k):
        label = f"{sensor_noise} sk{float(sensor_k):g}"
    elif _active(pose_model) and _active(pose_k):
        label = f"{pose_model} k{float(pose_k):g}"
    else:
        label = "clean"
    return data, label


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def derive(d):
    ex, ey, eyaw = d[:, 4], d[:, 5], wrap(d[:, 6])
    return {
        "cmd_dist": np.hypot(d[:, 0], d[:, 1]),
        "target_time": d[:, 3],
        "err_dist": np.hypot(ex, ey),
        "abs_yaw": np.abs(eyaw),
    }


def stats_line(name, err_cm, yaw_deg):
    print(f"  {name:<8} dist(cm): mean {err_cm.mean():.2f} median {np.median(err_cm):.2f} "
          f"p90 {np.percentile(err_cm,90):.2f} max {err_cm.max():.2f} | "
          f"yaw(deg): mean {yaw_deg.mean():.2f} median {np.median(yaw_deg):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", default="results_p2p_clean.csv")
    ap.add_argument("--noise", default="results_p2p_noise.csv")
    ap.add_argument("--out_prefix", default="noise_compare")
    ap.add_argument("--cmd_tol", type=float, default=1e-3)
    ap.add_argument("--single_col", action="store_true", default=False,
                    help="Fit the figure into a single IEEE column (3.5 in) using a vertical 4-panel stack "
                         "instead of the default double-column 2x2 layout.")
    args = ap.parse_args()

    craw, clabel = load(args.clean)
    nraw, nlabel = load(args.noise)
    n = min(len(craw), len(nraw))
    if len(craw) != len(nraw):
        print(f"[WARN] different counts clean={len(craw)} noise={len(nraw)}; using first {n}")
    craw, nraw = craw[:n], nraw[:n]

    # pairing check on the command columns
    md = np.abs(craw[:, :4] - nraw[:, :4]).max()
    paired = md <= args.cmd_tol
    print(f"[{'OK' if paired else 'WARN'}] command columns "
          f"{'match' if paired else 'DIFFER'} (max diff {md:.2e})")
    print(f"[INFO] clean=({clabel}) noise=({nlabel}) | {n} episodes\n")

    c, nz = derive(craw), derive(nraw)
    c_cm, n_cm = c["err_dist"] * 100, nz["err_dist"] * 100
    c_deg, n_deg = np.degrees(c["abs_yaw"]), np.degrees(nz["abs_yaw"])

    print("=== absolute accuracy ===")
    stats_line("CLEAN", c_cm, c_deg)
    stats_line(f"NOISE", n_cm, n_deg)

    if paired:
        d_dist = n_cm - c_cm            # positive => noise made it worse
        d_yaw = n_deg - c_deg
        print("\n=== paired degradation (noise - clean), positive => noise worse ===")
        print(f"  dist (cm): mean {d_dist.mean():+.2f} median {np.median(d_dist):+.2f} | "
              f"noise worse in {(d_dist>0).sum()}/{n}")
        print(f"  yaw (deg): mean {d_yaw.mean():+.2f} median {np.median(d_yaw):+.2f} | "
              f"noise worse in {(d_yaw>0).sum()}/{n}")

    # ---- IEEE STYLE ----
    COL_W = 3.5    # IEEE single-column width (inches)
    DCOL_W = 7.16  # IEEE double-column width (inches)
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    # ---------------- figure ----------------
    if args.single_col:
        # single-column: vertical 4-panel stack (3.5" wide)
        fig, axes = plt.subplots(4, 1, figsize=(COL_W, COL_W * 2.6))
        ax = axes.reshape(4)  # flat array for indexing
    else:
        # double-column: 2x2 grid (7.16" wide)
        fig, axes = plt.subplots(2, 2, figsize=(DCOL_W, DCOL_W * 0.62))
        ax = axes.ravel()

    # (a) distance-error CDF
    a = ax[0]
    for arr, lab, col in [(c_cm, "clean", "C0"), (n_cm, nlabel, "C3")]:
        xs = np.sort(arr); ys = np.arange(1, len(xs)+1)/len(xs)
        a.plot(xs, ys, label=lab, color=col)
    a.set_xlabel("Distance error (cm)"); a.set_ylabel("Cumulative fraction")
    a.set_title("(a) Distance-error CDF"); a.legend(loc="lower right")

    # (b) distance-error histogram
    a = ax[1]
    bins = np.linspace(0, max(c_cm.max(), n_cm.max()), 25)
    a.hist(c_cm, bins=bins, alpha=0.6, label="clean", color="C0")
    a.hist(n_cm, bins=bins, alpha=0.6, label=nlabel, color="C3")
    a.set_xlabel("Distance error (cm)"); a.set_ylabel("Episodes")
    a.set_title("(b) Distance error distribution"); a.legend(loc="best")

    # (c) paired scatter clean vs noise
    a = ax[2]
    if paired:
        lim = max(c_cm.max(), n_cm.max())
        a.scatter(c_cm, n_cm, s=8, alpha=0.6, color="C0", edgecolors="none")
        a.plot([0, lim], [0, lim], "k--", linewidth=0.7, label="equal")
        a.set_xlabel("Clean dist. err (cm)"); a.set_ylabel("Noised dist. err (cm)")
        a.set_title("(c) Paired error"); a.legend(loc="best")
    else:
        a.text(0.5, 0.5, "not paired", ha="center", fontsize=8); a.set_axis_off()

    # (d) yaw-error histogram
    a = ax[3]
    ybins = np.linspace(0, max(c_deg.max(), n_deg.max()), 25)
    a.hist(c_deg, bins=ybins, alpha=0.6, label="clean", color="C0")
    a.hist(n_deg, bins=ybins, alpha=0.6, label=nlabel, color="C3")
    a.set_xlabel("|Yaw| error (deg)"); a.set_ylabel("Episodes")
    a.set_title("(d) Yaw error distribution"); a.legend(loc="best")

    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}.pdf")
    fig.savefig(f"{args.out_prefix}.png")
    plt.close(fig)
    print(f"\n[plot] wrote {args.out_prefix}.pdf and {args.out_prefix}.png")


if __name__ == "__main__":
    main()
