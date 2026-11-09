# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause
"""Compare velocity-based P2P vs timed-P2P result CSVs.

Both result files must share the schema written by the play scripts:
    cmd_x, cmd_y, cmd_yaw, target_time, err_x, err_y, err_yaw   (SI: m, rad)

Because episode i used the IDENTICAL command in both runs, this does a PAIRED
comparison. It first asserts the command columns match row-for-row (this catches
the un-paired failure mode, e.g. if the P2P run was launched without
--inject_commands), then reports absolute, percentage, and paired-difference
statistics plus a multi-panel figure.

Usage:
    python compare_results.py --vel results_velocity.csv --p2p results_p2p.csv \
        --out_prefix compare --dist_floor 0.3

No Isaac / torch needed -- pure numpy + matplotlib.
"""

import argparse
import csv
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLS = ["cmd_x", "cmd_y", "cmd_yaw", "target_time", "err_x", "err_y", "err_yaw"]


def load_results(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        # Accept files that have the 7 core columns first, ignoring any extra
        # trailing columns (e.g. noise metadata written by the noise collector).
        if header[:len(COLS)] != COLS:
            raise AssertionError(
                f"{path}: unexpected header {header}\nexpected first columns {COLS}")
        data = np.array([[float(x) for x in row[:len(COLS)]] for row in r if row])
    return data


def wrap_to_pi(a):
    """Wrap angle(s) to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def derived(data):
    """Return a dict of derived per-episode quantities from a raw results array."""
    cmd_x, cmd_y, cmd_yaw = data[:, 0], data[:, 1], data[:, 2]
    target_time = data[:, 3]
    err_x, err_y, err_yaw = data[:, 4], data[:, 5], wrap_to_pi(data[:, 6])

    cmd_dist = np.hypot(cmd_x, cmd_y)
    err_dist = np.hypot(err_x, err_y)              # Euclidean position error (m)
    return {
        "cmd_x": cmd_x, "cmd_y": cmd_y, "cmd_yaw": cmd_yaw,
        "cmd_dist": cmd_dist, "target_time": target_time,
        "err_x": err_x, "err_y": err_y, "err_yaw": err_yaw,
        "err_dist": err_dist,
        "abs_err_x": np.abs(err_x), "abs_err_y": np.abs(err_y),
        "abs_err_yaw": np.abs(err_yaw),
    }


def summarize(name, d):
    """Print an absolute-error summary table (cm / deg)."""
    def stats(a):
        return (np.mean(a), np.median(a), np.std(a),
                np.percentile(a, 90), np.max(a))
    print(f"\n=== {name} : absolute error ({len(d['err_dist'])} episodes) ===")
    print(f"{'metric':<16}{'mean':>9}{'median':>9}{'std':>9}{'p90':>9}{'max':>9}")
    rows = [
        ("dist err (cm)", d["err_dist"] * 100),
        ("|x| err (cm)", d["abs_err_x"] * 100),
        ("|y| err (cm)", d["abs_err_y"] * 100),
        ("|yaw| err (deg)", np.degrees(d["abs_err_yaw"])),
    ]
    for label, a in rows:
        m, med, sd, p90, mx = stats(a)
        print(f"{label:<16}{m:>9.2f}{med:>9.2f}{sd:>9.2f}{p90:>9.2f}{mx:>9.2f}")


def pct_of_distance(d, floor):
    """Position error as % of commanded distance, only where cmd_dist >= floor.

    Percentage relative to a near-zero command is meaningless/explosive, so
    episodes below the floor are excluded (and the count reported).
    """
    mask = d["cmd_dist"] >= floor
    pct = 100.0 * d["err_dist"][mask] / d["cmd_dist"][mask]
    return pct, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vel", default="results_velocity.csv")
    ap.add_argument("--p2p", default="results_p2p.csv")
    ap.add_argument("--out_prefix", default="compare")
    ap.add_argument("--dist_floor", type=float, default=0.3,
                    help="Min commanded distance (m) for percentage metric; below this, %% is excluded.")
    ap.add_argument("--cmd_tol", type=float, default=1e-3,
                    help="Max allowed mismatch in command columns before warning (m/rad).")
    ap.add_argument("--single_col", action="store_true", default=False,
                    help="Fit figures into a single IEEE column (3.5 in) instead of double-column (7.16 in).")
    args = ap.parse_args()

    vel_raw = load_results(args.vel)
    p2p_raw = load_results(args.p2p)

    # ---- pairing check ----
    n = min(len(vel_raw), len(p2p_raw))
    if len(vel_raw) != len(p2p_raw):
        print(f"[WARN] Different episode counts: vel={len(vel_raw)} p2p={len(p2p_raw)}. "
              f"Comparing the first {n} rows.")
    vel_raw, p2p_raw = vel_raw[:n], p2p_raw[:n]

    cmd_diff = np.abs(vel_raw[:, :4] - p2p_raw[:, :4])
    max_cmd_diff = cmd_diff.max()
    if max_cmd_diff > args.cmd_tol:
        n_bad = int((cmd_diff.max(axis=1) > args.cmd_tol).sum())
        print("\n" + "!" * 70)
        print(f"[PAIRING WARNING] Command columns differ (max diff {max_cmd_diff:.4f} > tol {args.cmd_tol}).")
        print(f"  {n_bad}/{n} episodes have mismatched commands.")
        print("  The two runs are NOT paired -- did the P2P run use --inject_commands?")
        print("  Absolute stats are still valid; PAIRED differences below are NOT meaningful.")
        print("!" * 70)
        paired_ok = False
    else:
        print(f"[OK] Commands match row-for-row (max diff {max_cmd_diff:.2e}). Paired comparison valid.")
        paired_ok = True

    vel = derived(vel_raw)
    p2p = derived(p2p_raw)

    # ---- absolute summaries ----
    summarize("VELOCITY (cubic spline)", vel)
    summarize("P2P (timed policy)", p2p)

    # ---- percentage of commanded distance ----
    vel_pct, mask = pct_of_distance(vel, args.dist_floor)
    p2p_pct, _ = pct_of_distance(p2p, args.dist_floor)
    n_excl = int((~mask).sum())
    print(f"\n=== position error as %% of commanded distance (cmd_dist >= {args.dist_floor} m) ===")
    print(f"  excluded {n_excl}/{n} near-zero-distance episodes")
    print(f"  VELOCITY : mean {vel_pct.mean():.1f}%%  median {np.median(vel_pct):.1f}%%  p90 {np.percentile(vel_pct,90):.1f}%%")
    print(f"  P2P      : mean {p2p_pct.mean():.1f}%%  median {np.median(p2p_pct):.1f}%%  p90 {np.percentile(p2p_pct,90):.1f}%%")

    # ---- paired differences (P2P - VEL), same episode ----
    if paired_ok:
        d_dist = (p2p["err_dist"] - vel["err_dist"]) * 100  # cm; negative => P2P better
        d_yaw = np.degrees(p2p["abs_err_yaw"] - vel["abs_err_yaw"])
        print("\n=== paired difference  (P2P - VELOCITY), negative => P2P better ===")
        print(f"  dist err diff (cm): mean {d_dist.mean():+.2f}  median {np.median(d_dist):+.2f}")
        print(f"     P2P better in {int((d_dist<0).sum())}/{n} episodes")
        print(f"  yaw  err diff (deg): mean {d_yaw.mean():+.2f}  median {np.median(d_yaw):+.2f}")
        print(f"     P2P better in {int((d_yaw<0).sum())}/{n} episodes")

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

    def save_fig(fig, name):
        fig.tight_layout()
        fig.savefig(f"{name}.pdf")
        fig.savefig(f"{name}.png")
        plt.close(fig)
        print(f"[plot] wrote {name}.pdf and {name}.png")

    # ==========================================================
    # FIGURE 1: 6-panel overview
    # ==========================================================
    if args.single_col:
        fig, axes = plt.subplots(6, 1, figsize=(COL_W, COL_W * 4.2))
        ax = axes.ravel()
    else:
        fig, axes = plt.subplots(2, 3, figsize=(DCOL_W, DCOL_W * 0.56))
        ax = axes.ravel()

    # (a) distance-error histogram
    a = ax[0]
    bins = np.linspace(0, max(vel["err_dist"].max(), p2p["err_dist"].max()) * 100, 25)
    a.hist(vel["err_dist"] * 100, bins=bins, alpha=0.55, label="Velocity", color="C0")
    a.hist(p2p["err_dist"] * 100, bins=bins, alpha=0.55, label="P2P-T", color="C1")
    a.set_xlabel("Distance error (cm)"); a.set_ylabel("Episodes")
    a.set_title("(a) Distance error distribution"); a.legend(loc="best")

    # (b) yaw-error histogram
    a = ax[1]
    ybins = np.linspace(0, max(np.degrees(vel["abs_err_yaw"]).max(),
                               np.degrees(p2p["abs_err_yaw"]).max()), 25)
    a.hist(np.degrees(vel["abs_err_yaw"]), bins=ybins, alpha=0.55, label="Velocity", color="C0")
    a.hist(np.degrees(p2p["abs_err_yaw"]), bins=ybins, alpha=0.55, label="P2P-T", color="C1")
    a.set_xlabel("|Yaw| error (deg)"); a.set_ylabel("Episodes")
    a.set_title("(b) Yaw error distribution"); a.legend(loc="best")

    # (c) CDF
    a = ax[2]
    for d, lab, c in [(vel, "Velocity", "C0"), (p2p, "P2P-T", "C1")]:
        xs = np.sort(d["err_dist"] * 100)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        a.plot(xs, ys, label=lab, color=c)
    a.set_xlabel("Distance error (cm)"); a.set_ylabel("Cumulative fraction")
    a.set_title("(c) Distance-error CDF"); a.legend(loc="lower right")

    # (d) paired scatter
    a = ax[3]
    if paired_ok:
        lim = max(vel["err_dist"].max(), p2p["err_dist"].max()) * 100
        a.scatter(vel["err_dist"] * 100, p2p["err_dist"] * 100,
                  s=8, alpha=0.6, color="C0", edgecolors="none")
        a.plot([0, lim], [0, lim], "k--", linewidth=0.7, label="equal")
        a.set_xlabel("Velocity dist. err (cm)"); a.set_ylabel("P2P dist. err (cm)")
        a.set_title("(d) Paired error"); a.legend(loc="best")
    else:
        a.text(0.5, 0.5, "not paired", ha="center", fontsize=8); a.set_axis_off()

    # (e) error vs commanded distance
    a = ax[4]
    a.scatter(vel["cmd_dist"], vel["err_dist"] * 100,
              s=6, alpha=0.5, label="Velocity", color="C0", edgecolors="none")
    a.scatter(p2p["cmd_dist"], p2p["err_dist"] * 100,
              s=6, alpha=0.5, label="P2P-T", color="C1", edgecolors="none")
    a.axvline(args.dist_floor, color="gray", linestyle=":", linewidth=0.7,
              label=f"pct floor {args.dist_floor} m")
    a.set_xlabel("Commanded distance (m)"); a.set_ylabel("Distance error (cm)")
    a.set_title("(e) Error vs commanded distance"); a.legend(loc="best")

    # (f) error vs target time
    a = ax[5]
    a.scatter(vel["target_time"], vel["err_dist"] * 100,
              s=6, alpha=0.5, label="Velocity", color="C0", edgecolors="none")
    a.scatter(p2p["target_time"], p2p["err_dist"] * 100,
              s=6, alpha=0.5, label="P2P-T", color="C1", edgecolors="none")
    a.set_xlabel("Target time (s)"); a.set_ylabel("Distance error (cm)")
    a.set_title("(f) Error vs target time"); a.legend(loc="best")

    save_fig(fig, f"{args.out_prefix}_overview")

    # ==========================================================
    # FIGURE 2: percentage-of-distance comparison
    # ==========================================================
    if args.single_col:
        fig2, ax2 = plt.subplots(figsize=(COL_W, COL_W * 0.72))
    else:
        fig2, ax2 = plt.subplots(figsize=(DCOL_W * 0.5, DCOL_W * 0.35))
    pbins = np.linspace(0, max(vel_pct.max(), p2p_pct.max()), 25)
    ax2.hist(vel_pct, bins=pbins, alpha=0.55, label="Velocity", color="C0")
    ax2.hist(p2p_pct, bins=pbins, alpha=0.55, label="P2P-T", color="C1")
    ax2.set_xlabel("Position error (% of commanded distance)")
    ax2.set_ylabel("Episodes")
    ax2.set_title(f"Percentage error (cmd\\_dist $\\geq$ {args.dist_floor} m, "
                  f"{n_excl} excl.)")
    ax2.legend(loc="best")
    save_fig(fig2, f"{args.out_prefix}_percentage")


if __name__ == "__main__":
    main()
