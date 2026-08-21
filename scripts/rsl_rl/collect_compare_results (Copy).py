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
        assert header == COLS, f"{path}: unexpected header {header}\nexpected {COLS}"
        data = np.array([[float(x) for x in row] for row in r if row])
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

    # ==========================================================
    # FIGURE 1: distributions + paired scatter + error-vs-distance
    # ==========================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (0,0) distance-error histogram, overlaid
    ax = axes[0, 0]
    bins = np.linspace(0, max(vel["err_dist"].max(), p2p["err_dist"].max()) * 100, 30)
    ax.hist(vel["err_dist"] * 100, bins=bins, alpha=0.55, label="Velocity", color="C0")
    ax.hist(p2p["err_dist"] * 100, bins=bins, alpha=0.55, label="P2P", color="C1")
    ax.set_xlabel("distance error (cm)"); ax.set_ylabel("episodes")
    ax.set_title("Distance error distribution"); ax.legend(); ax.grid(alpha=0.3)

    # (0,1) yaw-error histogram
    ax = axes[0, 1]
    ybins = np.linspace(0, max(np.degrees(vel["abs_err_yaw"]).max(),
                               np.degrees(p2p["abs_err_yaw"]).max()), 30)
    ax.hist(np.degrees(vel["abs_err_yaw"]), bins=ybins, alpha=0.55, label="Velocity", color="C0")
    ax.hist(np.degrees(p2p["abs_err_yaw"]), bins=ybins, alpha=0.55, label="P2P", color="C1")
    ax.set_xlabel("|yaw| error (deg)"); ax.set_ylabel("episodes")
    ax.set_title("Yaw error distribution"); ax.legend(); ax.grid(alpha=0.3)

    # (0,2) CDF of distance error -- clearest single "which is better" view
    ax = axes[0, 2]
    for d, lab, c in [(vel, "Velocity", "C0"), (p2p, "P2P", "C1")]:
        xs = np.sort(d["err_dist"] * 100)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, label=lab, color=c, linewidth=2)
    ax.set_xlabel("distance error (cm)"); ax.set_ylabel("cumulative fraction")
    ax.set_title("Distance-error CDF (left/up = better)"); ax.legend(); ax.grid(alpha=0.3)

    # (1,0) paired scatter: vel vs p2p distance error (only if paired)
    ax = axes[1, 0]
    if paired_ok:
        lim = max(vel["err_dist"].max(), p2p["err_dist"].max()) * 100
        ax.scatter(vel["err_dist"] * 100, p2p["err_dist"] * 100, s=18, alpha=0.7)
        ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="equal")
        ax.set_xlabel("Velocity dist err (cm)"); ax.set_ylabel("P2P dist err (cm)")
        ax.set_title("Paired distance error\n(below line = P2P better)")
        ax.legend(); ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "not paired\n(commands differ)", ha="center", va="center")
        ax.set_axis_off()

    # (1,1) error vs commanded distance -- shows if either degrades with reach length
    ax = axes[1, 1]
    ax.scatter(vel["cmd_dist"], vel["err_dist"] * 100, s=16, alpha=0.6, label="Velocity", color="C0")
    ax.scatter(p2p["cmd_dist"], p2p["err_dist"] * 100, s=16, alpha=0.6, label="P2P", color="C1")
    ax.axvline(args.dist_floor, color="gray", linestyle=":", label=f"pct floor {args.dist_floor} m")
    ax.set_xlabel("commanded distance (m)"); ax.set_ylabel("distance error (cm)")
    ax.set_title("Error vs commanded distance"); ax.legend(); ax.grid(alpha=0.3)

    # (1,2) error vs target time -- does a tighter time budget hurt?
    ax = axes[1, 2]
    ax.scatter(vel["target_time"], vel["err_dist"] * 100, s=16, alpha=0.6, label="Velocity", color="C0")
    ax.scatter(p2p["target_time"], p2p["err_dist"] * 100, s=16, alpha=0.6, label="P2P", color="C1")
    ax.set_xlabel("target time (s)"); ax.set_ylabel("distance error (cm)")
    ax.set_title("Error vs target time"); ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    out1 = f"{args.out_prefix}_overview.png"
    fig.savefig(out1, dpi=120); plt.close(fig)
    print(f"\n[plot] wrote {out1}")

    # ==========================================================
    # FIGURE 2: percentage-of-distance comparison (guarded)
    # ==========================================================
    fig2, ax = plt.subplots(figsize=(8, 5))
    pbins = np.linspace(0, max(vel_pct.max(), p2p_pct.max()), 30)
    ax.hist(vel_pct, bins=pbins, alpha=0.55, label="Velocity", color="C0")
    ax.hist(p2p_pct, bins=pbins, alpha=0.55, label="P2P", color="C1")
    ax.set_xlabel("position error (% of commanded distance)")
    ax.set_ylabel("episodes")
    ax.set_title(f"Percentage error (cmd_dist >= {args.dist_floor} m, "
                 f"{n_excl} episodes excluded)")
    ax.legend(); ax.grid(alpha=0.3)
    fig2.tight_layout()
    out2 = f"{args.out_prefix}_percentage.png"
    fig2.savefig(out2, dpi=120); plt.close(fig2)
    print(f"[plot] wrote {out2}")


if __name__ == "__main__":
    main()
