# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause
"""IEEE double-column figure generation for the P2P / velocity / noise studies.

IEEE two-column format sizing:
  - single column width : 3.5 in  (88.9 mm)  -> use for most figures
  - double column width : 7.16 in (~180 mm)  -> use for wide/multi-panel figures
  - body/caption font    : ~8 pt; figure text set to 8 pt so it matches when placed
  - lines/markers scaled down so they read correctly at print size

This produces vector PDF (preferred for LaTeX) AND PNG. Use the PDF in \\includegraphics.

Usage:
  # velocity vs P2P, single-column CDF + a compact 2-panel summary:
  python plot_ieee.py compare --vel results_velocity.csv --p2p results_p2p_clean.csv --out fig_compare

  # noise sweep overlay (clean + k1/k2/k3) as one single-column CDF:
  python plot_ieee.py noise_sweep --clean results_p2p_clean.csv \
      --noise results_p2p_allsensors_k1.csv results_p2p_allsensors_k2.csv results_p2p_allsensors_k3.csv \
      --labels "k=1" "k=2" "k=3" --out fig_noise

  # per-episode time series (single column), target-time line included:
  python plot_ieee.py timeseries --csv vel_run.csv --episode 0 --out fig_ts_ep0

Pure numpy + matplotlib; no Isaac needed.
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


# ----------------------------------------------------------------------------
# IEEE STYLE
# ----------------------------------------------------------------------------
COL_W = 3.5     # inches, IEEE single-column
DCOL_W = 7.16   # inches, IEEE double-column (full text width)

def set_ieee_style():
    """Apply IEEE-appropriate rcParams: small serif font, thin lines, tight."""
    plt.rcParams.update({
        # Fonts -- serif to match IEEE body text (Times-like). 8pt base.
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        # Lines / markers scaled for print
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        # Layout
        "figure.dpi": 150,
        "savefig.dpi": 600,          # high-res raster fallback
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "legend.frameon": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        # Embed fonts properly in PDF for IEEE submission
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, out_prefix):
    """Save both PDF (for LaTeX) and PNG (for quick view)."""
    fig.savefig(f"{out_prefix}.pdf")
    fig.savefig(f"{out_prefix}.png")
    plt.close(fig)
    print(f"[ieee] wrote {out_prefix}.pdf and {out_prefix}.png")


# ----------------------------------------------------------------------------
# DATA LOADING (tolerant of 7-col or noise-metadata schemas)
# ----------------------------------------------------------------------------
CORE = ["cmd_x", "cmd_y", "cmd_yaw", "target_time", "err_x", "err_y", "err_yaw"]

def load_results(path):
    with open(path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        if header[:len(CORE)] != CORE:
            raise SystemExit(f"{path}: unexpected header (need first cols {CORE})")
        data = np.array([[float(x) for x in row[:len(CORE)]] for row in r if row])
    return data

def dist_err_cm(data):
    return np.hypot(data[:, 4], data[:, 5]) * 100.0

def yaw_err_deg(data):
    a = (data[:, 6] + np.pi) % (2 * np.pi) - np.pi
    return np.abs(a) * 180.0 / np.pi

def load_timeseries(path):
    eps = defaultdict(lambda: defaultdict(list))
    tt = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            e = int(row["episode"])
            eps[e]["t"].append(float(row["time_s"]))
            eps[e]["ex"].append(float(row["err_x_cm"]))
            eps[e]["ey"].append(float(row["err_y_cm"]))
            eps[e]["eyaw"].append(float(row["err_yaw_deg"]))
            eps[e]["h"].append(float(row["pos_z"]))
            eps[e]["roll"].append(float(row["roll_deg"]))
            eps[e]["pitch"].append(float(row["pitch_deg"]))
            eps[e]["vx"].append(float(row["vx"]))
            eps[e]["vy"].append(float(row["vy"]))
            eps[e]["vz"].append(float(row["vz"]))
            eps[e]["wx"].append(float(row["wx"]))
            eps[e]["wy"].append(float(row["wy"]))
            eps[e]["wz"].append(float(row["wz"]))
            
            if row.get("target_time_s"):
                tt[e] = float(row["target_time_s"])
    out = {}
    for e, c in eps.items():
        out[e] = {k: np.array(v) for k, v in c.items()}
        if e in tt:
            out[e]["tt"] = tt[e]
    return out


# ----------------------------------------------------------------------------
# FIGURES
# ----------------------------------------------------------------------------
def fig_compare(args):
    """Velocity vs P2P: single-column CDF, plus a compact double-column 2-panel."""
    vel = load_results(args.vel)
    p2p = load_results(args.p2p)
    v_cm, p_cm = dist_err_cm(vel), dist_err_cm(p2p)
    v_deg, p_deg = yaw_err_deg(vel), yaw_err_deg(p2p)

    # --- single-column CDF (the headline figure) ---
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.72))
    for arr, lab, c in [(v_cm, "Velocity (open-loop)", "C0"),
                        (p_cm, "P2P (closed-loop)", "C3")]:
        xs = np.sort(arr); ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, label=lab, color=c)
    ax.set_xlabel("Final distance error (cm)")
    ax.set_ylabel("Cumulative fraction")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, f"{args.out}_cdf")

    # --- double-column 2-panel: distance CDF + yaw CDF ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DCOL_W, DCOL_W * 0.34))
    for arr, lab, c in [(v_cm, "Velocity", "C0"), (p_cm, "P2P", "C3")]:
        xs = np.sort(arr); ys = np.arange(1, len(xs)+1)/len(xs)
        a1.plot(xs, ys, label=lab, color=c)
    a1.set_xlabel("Distance error (cm)"); a1.set_ylabel("Cumulative fraction")
    a1.legend(loc="lower right"); a1.set_title("(a) Position accuracy")
    for arr, lab, c in [(v_deg, "Velocity", "C0"), (p_deg, "P2P", "C3")]:
        xs = np.sort(arr); ys = np.arange(1, len(xs)+1)/len(xs)
        a2.plot(xs, ys, label=lab, color=c)
    a2.set_xlabel("Yaw error (deg)"); a2.set_ylabel("Cumulative fraction")
    a2.legend(loc="lower right"); a2.set_title("(b) Heading accuracy")
    fig.tight_layout()
    save(fig, f"{args.out}_2panel")

    print(f"[stats] velocity dist mean {v_cm.mean():.1f}cm median {np.median(v_cm):.1f}cm")
    print(f"[stats] p2p      dist mean {p_cm.mean():.1f}cm median {np.median(p_cm):.1f}cm")


def fig_noise_sweep(args):
    """Overlay clean + several noise levels as one single-column CDF."""
    clean = load_results(args.clean)
    c_cm = dist_err_cm(clean)
    fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.72))
    xs = np.sort(c_cm); ys = np.arange(1, len(xs)+1)/len(xs)
    ax.plot(xs, ys, label="clean", color="C0")
    colors = ["C1", "C2", "C3", "C4", "C5"]
    labels = args.labels if args.labels else [os.path.basename(p) for p in args.noise]
    for i, (path, lab) in enumerate(zip(args.noise, labels)):
        arr = dist_err_cm(load_results(path))
        xs = np.sort(arr); ys = np.arange(1, len(xs)+1)/len(xs)
        ax.plot(xs, ys, label=lab, color=colors[i % len(colors)])
    ax.set_xlabel("Final distance error (cm)")
    ax.set_ylabel("Cumulative fraction")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, args.out)


def fig_timeseries(args):
    """Dual-panel time series: Errors (top) and Velocities (bottom) on shared axes."""
    eps = load_timeseries(args.csv)
    if args.episode not in eps:
        raise SystemExit(f"episode {args.episode} not in {args.csv}")
    d = eps[args.episode]
    tt = d.get("tt", None)

    t = d["t"]
    if tt is not None:
        mask = t <= (tt + args.margin)
    else:
        mask = np.ones_like(t, dtype=bool)

    def clip(key):
        return d[key][mask]

    tc = clip("t")
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(COL_W, COL_W * 1.15), sharex=True)

    # --- TOP PLOT: Position and Orientation Errors ---
    a1.plot(tc, clip("ex"), label=r"$e_x$ (cm)", color="C0")
    a1.plot(tc, clip("ey"), label=r"$e_y$ (cm)", color="C1")
    a1.plot(tc, clip("eyaw"), label=r"$e_{yaw}$ (deg)", color="C3", ls="--")
    
    # if tt is not None:
    #     a1.axvline(tt, color="black", ls=":", label=fr"$t_g$={tt:.2f}s")
    if tt is not None:
        # Define 'a1_trans' before placing text so we can position it relative to axes
        a1_trans = matplotlib.transforms.blended_transform_factory(a1.transData, a1.transAxes)
        a1.axvline(tt, color="black", ls=":", lw=1)
        # Place the text at x=tt and y=1.02 (just above the top spine of the plot)
        a1.text(tt, 1.02, fr"$t_g$={tt:.2f}s", color="black", transform=a1_trans,
                ha="center", va="bottom", fontsize=8)

    a1.axhline(0, color="k", lw=0.5)
    
    a1.set_ylabel("Error")
    # a1.legend(loc="best", ncol=3)
    if args.legend == "right":
        a1.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    elif args.legend == "below":
        a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    else:
        a1.legend(loc="best", ncol=3)

    # --- BOTTOM PLOT: Linear and Angular Velocities ---
    # Convert velocities: m/s -> cm/s, rad/s -> deg/s
    vx_cm = clip("vx") * 100.0
    vy_cm = clip("vy") * 100.0
    wz_deg = clip("wz") * (180.0 / math.pi)
    
    a2.plot(tc, vx_cm, label=r"$v_x$ (cm/s)", color="C0")
    a2.plot(tc, vy_cm, label=r"$v_y$ (cm/s)", color="C1")
    a2.plot(tc, wz_deg, label=r"$\omega_{z}$ (deg/s)", color="C3", ls="--")
    
    if tt is not None:
        a2.axvline(tt, color="black", ls=":")
    a2.axhline(0, color="k", lw=0.5)
    
    a2.set_xlabel("Time (s)")
    a2.set_ylabel("Velocity")
    # a2.legend(loc="best", ncol=3)
    if args.legend == "right":
        a2.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    elif args.legend == "below":
        # Requires a slightly larger negative Y offset to clear the X-axis label
        a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3) 
    else:
        a2.legend(loc="best", ncol=3)

    if tt is not None:
        a1.set_xlim(tc.min(), tt + args.margin)

    # Force visible, slightly darker grid lines to render strictly behind the data
    a1.grid(visible=True, which="major", axis="both", color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    a1.set_axisbelow(True)
    
    a2.grid(visible=True, which="major", axis="both", color="gray", linestyle="-", linewidth=0.5, alpha=0.5)
    a2.set_axisbelow(True)

    fig.tight_layout()
    save(fig, args.out)


def main():
    set_ieee_style()
    ap = argparse.ArgumentParser(description="IEEE double-column figures.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare", help="velocity vs P2P")
    c.add_argument("--vel", required=True)
    c.add_argument("--p2p", required=True)
    c.add_argument("--out", default="fig_compare")
    c.set_defaults(func=fig_compare)

    n = sub.add_parser("noise_sweep", help="clean + noise levels overlay")
    n.add_argument("--clean", required=True)
    n.add_argument("--noise", nargs="+", required=True)
    n.add_argument("--labels", nargs="*", default=None)
    n.add_argument("--out", default="fig_noise")
    n.set_defaults(func=fig_noise_sweep)

    t = sub.add_parser("timeseries", help="one episode time series")
    t.add_argument("--csv", required=True)
    t.add_argument("--episode", type=int, default=0)
    t.add_argument("--margin", type=float, default=0.2,
                   help="Extra seconds to show past the target time (default 0 = stop exactly at target).")
    t.add_argument("--out", default="fig_timeseries")
    t.add_argument("--legend", type=str, choices=["right", "below", "inside"], default="right",
                   help="Where to place the legend to avoid obstructing the graph.")
    t.set_defaults(func=fig_timeseries)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
