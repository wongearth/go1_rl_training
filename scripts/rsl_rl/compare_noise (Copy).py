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
    # tolerate the 7-col (no noise meta) or 9-col schema
    data = np.array([[float(x) for x in row[:7]] for row in rows])
    model = rows[0][7] if len(rows[0]) > 7 else "?"
    k = rows[0][8] if len(rows[0]) > 8 else "?"
    return data, model, k


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
    args = ap.parse_args()

    craw, cm, ck = load(args.clean)
    nraw, nm, nk = load(args.noise)
    n = min(len(craw), len(nraw))
    if len(craw) != len(nraw):
        print(f"[WARN] different counts clean={len(craw)} noise={len(nraw)}; using first {n}")
    craw, nraw = craw[:n], nraw[:n]

    # pairing check on the command columns
    md = np.abs(craw[:, :4] - nraw[:, :4]).max()
    paired = md <= args.cmd_tol
    print(f"[{'OK' if paired else 'WARN'}] command columns "
          f"{'match' if paired else 'DIFFER'} (max diff {md:.2e})")
    print(f"[INFO] clean model={cm} k={ck} | noise model={nm} k={nk} | {n} episodes\n")

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

    # ---------------- figure ----------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (0,0) distance-error CDF
    a = ax[0, 0]
    for arr, lab, col in [(c_cm, "clean", "C0"), (n_cm, f"{nm} k{nk}", "C3")]:
        xs = np.sort(arr); ys = np.arange(1, len(xs)+1)/len(xs)
        a.plot(xs, ys, label=lab, color=col, linewidth=2)
    a.set_xlabel("distance error (cm)"); a.set_ylabel("cumulative fraction")
    a.set_title("Distance-error CDF (left/up = better)"); a.legend(); a.grid(alpha=0.3)

    # (0,1) distance-error histogram
    a = ax[0, 1]
    bins = np.linspace(0, max(c_cm.max(), n_cm.max()), 30)
    a.hist(c_cm, bins=bins, alpha=0.6, label="clean", color="C0")
    a.hist(n_cm, bins=bins, alpha=0.6, label=f"{nm} k{nk}", color="C3")
    a.set_xlabel("distance error (cm)"); a.set_ylabel("episodes")
    a.set_title("Distance error distribution"); a.legend(); a.grid(alpha=0.3)

    # (1,0) paired scatter clean vs noise
    a = ax[1, 0]
    if paired:
        lim = max(c_cm.max(), n_cm.max())
        a.scatter(c_cm, n_cm, s=18, alpha=0.7)
        a.plot([0, lim], [0, lim], "k--", linewidth=1, label="equal")
        a.set_xlabel("clean dist err (cm)"); a.set_ylabel("noised dist err (cm)")
        a.set_title("Paired error (above line = noise worse)")
        a.legend(); a.grid(alpha=0.3)
    else:
        a.text(0.5, 0.5, "not paired", ha="center"); a.set_axis_off()

    # (1,1) yaw-error histogram
    a = ax[1, 1]
    ybins = np.linspace(0, max(c_deg.max(), n_deg.max()), 30)
    a.hist(c_deg, bins=ybins, alpha=0.6, label="clean", color="C0")
    a.hist(n_deg, bins=ybins, alpha=0.6, label=f"{nm} k{nk}", color="C3")
    a.set_xlabel("|yaw| error (deg)"); a.set_ylabel("episodes")
    a.set_title("Yaw error distribution"); a.legend(); a.grid(alpha=0.3)

    fig.tight_layout()
    out = f"{args.out_prefix}.png"
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f"\n[plot] wrote {out}")


if __name__ == "__main__":
    main()
