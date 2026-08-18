# Copyright (c) 2022-2025.
# SPDX-License-Identifier: BSD-3-Clause
"""Generate a shared, seeded command set for paired P2P comparison runs.

Writes a CSV with one row per episode:  cmd_x, cmd_y, cmd_yaw, target_time

Both the velocity-based P2P script and the timed-P2P script read this same
file so that episode i has the *identical* target in both runs, making the two
result files directly comparable row-for-row -- regardless of how each env
would otherwise sample targets internally.

Standalone: no Isaac / torch needed. Import-and-call OR run directly.

    python generate_commands.py --num_episodes 100 --seed 42 --out p2p_commands.csv

The commands are sampled as base-frame relative targets (dx, dy, dyaw) plus a
per-episode target_time, matching the ranges used by the play scripts.
"""

import argparse
import csv
import math
import os

import numpy as np


# Default sampling ranges -- keep these in sync with the play scripts' CLI defaults.
DEFAULT_X_RANGE = (-2.0, 2.0)
DEFAULT_Y_RANGE = (-2.0, 2.0)
DEFAULT_YAW_RANGE = (-math.pi, math.pi)
DEFAULT_TIME_RANGE = (2.5, 4.0)

CSV_HEADER = ["cmd_x", "cmd_y", "cmd_yaw", "target_time"]


def generate_commands(
    num_episodes: int,
    seed: int,
    x_range=DEFAULT_X_RANGE,
    y_range=DEFAULT_Y_RANGE,
    yaw_range=DEFAULT_YAW_RANGE,
    time_range=DEFAULT_TIME_RANGE,
):
    """Return a list of (cmd_x, cmd_y, cmd_yaw, target_time) tuples."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(num_episodes):
        cmd_x = float(rng.uniform(*x_range))
        cmd_y = float(rng.uniform(*y_range))
        cmd_yaw = float(rng.uniform(*yaw_range))
        target_time = float(rng.uniform(*time_range))
        rows.append((cmd_x, cmd_y, cmd_yaw, target_time))
    return rows


def write_commands_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        for r in rows:
            w.writerow([f"{v:.6f}" for v in r])


def load_commands_csv(path):
    """Read the commands CSV back into a list of (cmd_x, cmd_y, cmd_yaw, target_time)."""
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        assert header == CSV_HEADER, f"Unexpected header in {path}: {header} (expected {CSV_HEADER})"
        for line in r:
            if not line:
                continue
            rows.append(tuple(float(x) for x in line[:4]))
    return rows


def ensure_commands(path, num_episodes, seed, **ranges):
    """Create the commands CSV if it doesn't exist; otherwise leave it untouched.

    Returns the loaded command rows. This is the function the play scripts call:
    the FIRST script to run creates the file, every later script reuses it, so
    all runs share one identical command set.
    """
    if os.path.exists(path):
        rows = load_commands_csv(path)
        print(f"[commands] Reusing existing command file: {path}  ({len(rows)} episodes)")
        if len(rows) < num_episodes:
            raise RuntimeError(
                f"[commands] {path} has only {len(rows)} episodes but {num_episodes} requested. "
                f"Delete the file to regenerate, or lower --num_episodes."
            )
        return rows[:num_episodes]
    rows = generate_commands(num_episodes, seed, **ranges)
    write_commands_csv(path, rows)
    # re-read so the creating run uses the SAME 6-decimal values every later run
    # will read -> guarantees byte-identical commands across all scripts.
    rows = load_commands_csv(path)
    print(f"[commands] Created new command file: {path}  ({len(rows)} episodes, seed={seed})")
    return rows[:num_episodes]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate shared seeded command set for paired P2P runs.")
    p.add_argument("--num_episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="p2p_commands.csv")
    p.add_argument("--x_range", type=float, nargs=2, default=list(DEFAULT_X_RANGE))
    p.add_argument("--y_range", type=float, nargs=2, default=list(DEFAULT_Y_RANGE))
    p.add_argument("--yaw_range", type=float, nargs=2, default=list(DEFAULT_YAW_RANGE))
    p.add_argument("--time_range", type=float, nargs=2, default=list(DEFAULT_TIME_RANGE))
    p.add_argument("--force", action="store_true", help="Overwrite an existing file.")
    args = p.parse_args()

    if os.path.exists(args.out) and not args.force:
        print(f"[commands] {args.out} already exists; use --force to overwrite. Loading it instead.")
        rows = load_commands_csv(args.out)
    else:
        rows = generate_commands(
            args.num_episodes, args.seed,
            tuple(args.x_range), tuple(args.y_range),
            tuple(args.yaw_range), tuple(args.time_range),
        )
        write_commands_csv(args.out, rows)
        print(f"[commands] Wrote {len(rows)} episodes to {args.out} (seed={args.seed})")
