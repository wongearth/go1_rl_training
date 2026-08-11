#!/usr/bin/env python3
"""
Print the MC quadruped's per-link masses and inertias AS THE SIMULATOR SEES THEM,
by spawning the articulation in a headless Isaac Sim and reading its physics data.

This avoids needing the raw USD `pxr` bindings and, more importantly, reports the
*resolved* values (USD + any ArticulationCfg overrides + PhysX parsing), which is
exactly what determines acceleration behavior during training.

Run:
    ./isaaclab.sh -p print_robot_masses.py
(from the IsaacLab root). Edit ROBOT_CFG_IMPORT below if your cfg path differs.
"""

from isaaclab.app import AppLauncher

# launch a headless sim app first (required before importing sim modules)
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation

# ---- import your robot cfg -------------------------------------------------
# adjust this import to wherever MC_QUADRUPED_CFG lives
import sys, os
sys.path.append(os.path.expanduser("~/go1_rl_training/source/mc_tasks"))
from data.robots.mc_quadruped.mc_quadruped_cfg import MC_QUADRUPED_CFG
# ---------------------------------------------------------------------------


def main():
    # minimal scene: one ground plane, one robot
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device="cpu")
    sim = sim_utils.SimulationContext(sim_cfg)

    # ground
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())

    # spawn robot at a single prim path
    robot_cfg = MC_QUADRUPED_CFG.replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    sim.reset()

    # physics data is now populated
    body_names = robot.body_names
    masses = robot.root_physx_view.get_masses()[0]        # (num_bodies,)
    inertias = robot.root_physx_view.get_inertias()[0]    # (num_bodies, 9) flattened 3x3

    print("\n" + "=" * 74)
    print(f"{'link':<28}{'mass (kg)':>12}{'Ixx':>12}{'Iyy':>12}{'Izz':>12}")
    print("-" * 74)
    total = 0.0
    min_mass = (1e9, "")
    for i, name in enumerate(body_names):
        m = float(masses[i])
        total += m
        if m < min_mass[0] and m > 0:
            min_mass = (m, name)
        I = inertias[i].reshape(3, 3)
        ixx, iyy, izz = float(I[0, 0]), float(I[1, 1]), float(I[2, 2])
        print(f"{name:<28}{m:>12.5f}{ixx:>12.3e}{iyy:>12.3e}{izz:>12.3e}")
    print("-" * 74)
    print(f"{'TOTAL':<28}{total:>12.5f}")
    print(f"\nLightest link: {min_mass[1]} = {min_mass[0]:.5f} kg")

    # joint-level: armature and joint properties
    print("\n" + "=" * 74)
    print("JOINT PROPERTIES")
    print("-" * 74)
    try:
        armature = robot.root_physx_view.get_dof_armatures()[0]
        stiffness = robot.data.joint_stiffness[0]
        damping = robot.data.joint_damping[0]
        joint_names = robot.joint_names
        print(f"{'joint':<24}{'armature':>12}{'stiffness':>12}{'damping':>12}")
        print("-" * 60)
        for i, jn in enumerate(joint_names):
            print(f"{jn:<24}{float(armature[i]):>12.5f}"
                  f"{float(stiffness[i]):>12.3f}{float(damping[i]):>12.3f}")
    except Exception as e:
        print(f"(could not read joint armature/gains: {e})")

    print("\nReference Go1 values (approx, for comparison):")
    print("  trunk mass ~4-6 kg, thigh ~0.6-1.0 kg, calf ~0.1-0.2 kg")
    print("  calf inertia ~1e-4 to 1e-3 kg·m²")
    print("  If your lightest link is << Go1's calf, that's the acceleration source.")
    print("  If armature is 0.0 everywhere, adding ~0.01-0.05 will damp accel spikes.")

    simulation_app.close()


if __name__ == "__main__":
    main()
