"""Standalone script to load a robot USD and print its body/link names and prim paths.

Usage (from your terminal, with your conda env activated):

    python check_body_names.py --usd_path /home/wong/go1_rl_training/source/mc_tasks/data/robots/mc_quadruped/quadruped_robot_01.usd

Add --headless if you don't need the GUI window.
"""

import argparse

from isaaclab.app import AppLauncher

# ------------------------------------------------------------------
# 1. CLI args + AppLauncher (must happen BEFORE any other isaaclab/isaacsim/omni import)
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Check body names/paths of a robot USD.")
parser.add_argument(
    "--usd_path",
    type=str,
    default="/home/wong/go1_rl_training/source/mc_tasks/data/robots/mc_quadruped/quadruped_robot_01.usd",
    help="Absolute path to the robot .usd file.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ------------------------------------------------------------------
# 2. Everything else must be imported AFTER the simulation app starts
# ------------------------------------------------------------------
import os  # noqa: E402

from pxr import Usd, UsdPhysics  # noqa: E402


def main():
    # Always resolve to an absolute path. Isaac Sim's asset resolver can
    # mangle relative paths (e.g. prepend its own search roots), which is
    # what causes errors like "mc_tasks/mc_tasks/data/...".
    usd_path = os.path.abspath(os.path.expanduser(args_cli.usd_path))
    print(f"\nResolved usd_path: {usd_path}")

    if not os.path.isfile(usd_path):
        print(f"ERROR: file does not exist at '{usd_path}'")
        print("Double check --usd_path (relative paths are resolved against your current working directory).")
        return

    print(f"Opening stage: {usd_path}\n")

    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"ERROR: could not open USD at '{usd_path}'")
        return

    print("=" * 70)
    print("ARTICULATION ROOT(S)")
    print("=" * 70)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            print(f"  {prim.GetPath()}  (type: {prim.GetTypeName()})")

    print()
    print("=" * 70)
    print("RIGID BODIES (i.e. links / bodies)")
    print("=" * 70)
    body_count = 0
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body_count += 1
            print(f"  [{body_count:2d}] name='{prim.GetName()}'  path={prim.GetPath()}")

    print()
    print("=" * 70)
    print("JOINTS")
    print("=" * 70)
    joint_count = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint_count += 1
            joint = UsdPhysics.Joint(prim)
            body0_targets = joint.GetBody0Rel().GetTargets()
            body1_targets = joint.GetBody1Rel().GetTargets()
            print(
                f"  [{joint_count:2d}] name='{prim.GetName()}'  path={prim.GetPath()}\n"
                f"        body0={body0_targets}  body1={body1_targets}"
            )

    print()
    print(f"Total rigid bodies: {body_count}")
    print(f"Total joints: {joint_count}\n")


if __name__ == "__main__":
    main()
    simulation_app.close()