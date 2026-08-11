# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
Inspect per-link mass and inertia for a quadruped articulation.

Spawns a single robot in an empty stage, lets physics resolve, then dumps
per-body mass, center-of-mass, and diagonal inertia. Use this to verify that
the four legs (and the base) are inertially symmetric -- an asymmetry here is
the kind of thing that makes every trained policy load the same diagonal.

Run inside the Isaac Lab python environment, e.g.:

    # MC quadruped (your robot)
    python inspect_link_masses.py --robot mc

    # stock Unitree Go1 (from the isaaclab_assets library)
    python inspect_link_masses.py --robot go1

Optionally group the report by leg-corner so diagonal symmetry is obvious:

    python inspect_link_masses.py --robot mc --group-legs
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

# ---------------------------------------------------------------------------
# CLI + app launch (must happen before any isaaclab.* / omni.* imports)
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Inspect per-link mass/inertia of a quadruped.")
parser.add_argument(
    "--robot",
    type=str,
    default="mc",
    choices=["mc", "go1"],
    help="Which robot cfg to load: 'mc' (your MC quadruped) or 'go1' (stock Isaac Lab Unitree Go1).",
)
parser.add_argument(
    "--group-legs",
    action="store_true",
    help="Group the per-body report by leg index so diagonal symmetry is easy to eyeball.",
)
# append AppLauncher cli args (adds --headless etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# force headless -- we only need physics, not rendering
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------------------------------------------------------------------------
# Everything below runs with the sim app live
# ---------------------------------------------------------------------------
import os
import sys

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext


def get_robot_cfg(robot: str):
    """Return the ArticulationCfg for the requested robot."""
    if robot == "mc":
        # Resolve the MC cfg the same way your env file does: walk up from this
        # script to the workspace root that contains the `data` package, then import.
        # Adjust the number of parent hops if you place this script elsewhere.
        here = os.path.dirname(os.path.abspath(__file__))
        # Try a few plausible roots so this works regardless of where you drop the file.
        candidate_roots = [
            os.path.abspath(os.path.join(here, "../../../../")),
            os.path.abspath(os.path.join(here, "../../../")),
            os.path.abspath(os.path.join(here, "../../")),
            os.path.abspath(os.path.join(here, "..")),
            here,
        ]
        for root in candidate_roots:
            if os.path.isdir(os.path.join(root, "data", "robots", "mc_quadruped")):
                if root not in sys.path:
                    sys.path.append(root)
                break
        else:
            raise FileNotFoundError(
                "Could not locate 'data/robots/mc_quadruped' above this script. "
                "Edit candidate_roots in get_robot_cfg() to point at your workspace root."
            )
        from data.robots.mc_quadruped.mc_quadruped_cfg import MC_QUADRUPED_CFG

        return MC_QUADRUPED_CFG.replace(prim_path="/World/Robot")

    elif robot == "go1":
        # Stock Go1 ships in the isaaclab_assets package.
        from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG

        return UNITREE_GO1_CFG.replace(prim_path="/World/Robot")

    else:
        raise ValueError(f"Unknown robot '{robot}'")


def main():
    # ---- build a minimal scene: one ground plane, one robot -----------------
    sim_cfg = SimulationCfg(dt=0.005, device=args_cli.device if hasattr(args_cli, "device") else "cuda:0")
    sim = SimulationContext(sim_cfg)

    # ground
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/ground", ground_cfg)

    # light (headless still wants a valid stage)
    light_cfg = sim_utils.DomeLightCfg(intensity=1000.0)
    light_cfg.func("/World/Light", light_cfg)

    # robot
    robot_cfg = get_robot_cfg(args_cli.robot)
    robot = Articulation(robot_cfg)

    # ---- reset + settle so physics resolves masses/inertia ------------------
    sim.reset()
    for _ in range(2):
        sim.step()
        robot.update(sim_cfg.dt)

    # ---- pull physics-resolved body properties ------------------------------
    # root_physx_view exposes the values PhysX actually simulates (post-parse),
    # which is what you want -- not just what the USD authored.
    body_names = robot.body_names
    masses = robot.root_physx_view.get_masses()[0]            # (num_bodies,)
    coms = robot.root_physx_view.get_coms()[0]                # (num_bodies, 7): pos(3)+quat(4)
    inertias = robot.root_physx_view.get_inertias()[0]        # (num_bodies, 9): flattened 3x3

    total_mass = float(torch.sum(masses).item())

    # ---- report -------------------------------------------------------------
    print("\n" + "=" * 92)
    print(f"ROBOT: {args_cli.robot}    |    bodies: {len(body_names)}    |    total mass: {total_mass:.4f} kg")
    print("=" * 92)

    def print_row(name, i):
        m = float(masses[i].item())
        cx, cy, cz = (float(coms[i, j].item()) for j in range(3))
        # diagonal inertia terms (Ixx, Iyy, Izz) sit at flat indices 0, 4, 8
        ixx, iyy, izz = float(inertias[i, 0]), float(inertias[i, 4]), float(inertias[i, 8])
        print(
            f"  {name:<16} m={m:9.5f}  "
            f"com=({cx:+.4f},{cy:+.4f},{cz:+.4f})  "
            f"I_diag=({ixx:.5e}, {iyy:.5e}, {izz:.5e})"
        )

    if args_cli.group_legs:
        # group by trailing index so leg 0..3 line up, plus an "other/base" bucket
        buckets: dict[str, list[int]] = {}
        for i, name in enumerate(body_names):
            # take the trailing digit if present, else 'base'
            key = "".join(ch for ch in name if ch.isdigit())
            key = key if key != "" else "base"
            buckets.setdefault(key, []).append(i)
        for key in sorted(buckets.keys()):
            print(f"\n-- group '{key}' " + "-" * 70)
            for i in buckets[key]:
                print_row(body_names[i], i)
    else:
        print()
        for i, name in enumerate(body_names):
            print_row(name, i)

    # ---- symmetry hint: compare per-leg total mass --------------------------
    # crude but useful: sum mass of all bodies sharing each trailing leg index
    leg_mass: dict[str, float] = {}
    for i, name in enumerate(body_names):
        key = "".join(ch for ch in name if ch.isdigit())
        if key == "":
            continue
        # only keep single-digit leg indices 0..3
        if key in {"0", "1", "2", "3"}:
            leg_mass[key] = leg_mass.get(key, 0.0) + float(masses[i].item())
    if len(leg_mass) == 4:
        print("\n" + "-" * 92)
        print("Per-leg total mass (bodies sharing trailing index 0..3):")
        for k in sorted(leg_mass.keys()):
            print(f"  leg {k}: {leg_mass[k]:.5f} kg")
        vals = torch.tensor(list(leg_mass.values()))
        spread = float((vals.max() - vals.min()).item())
        print(f"  max-min spread across legs: {spread:.6e} kg", "  <-- should be ~0 for a symmetric robot")

    print("=" * 92 + "\n")

    simulation_app.close()


if __name__ == "__main__":
    main()
