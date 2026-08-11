"""Check stage units and robot mass/inertia properties.

A very common cause of instant NaN physics explosions with custom robot USDs
is a units mismatch (e.g. authored in cm but treated as meters), or a
degenerate (near-zero) mass/inertia on some link. This script reports both
without running any simulation steps.

Run:
    python check_units_and_mass.py --usd_path /path/to/quadruped_robot_01.usd
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check stage units and robot mass/inertia.")
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

from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402


def main():
    usd_path = os.path.abspath(os.path.expanduser(args_cli.usd_path))
    stage = Usd.Stage.Open(usd_path)

    print("\n" + "=" * 70)
    print("STAGE UNITS")
    print("=" * 70)
    meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
    print(f"  metersPerUnit  : {meters_per_unit}  (1.0 = authored in meters, 0.01 = authored in cm)")
    if abs(meters_per_unit - 1.0) > 1e-6:
        print("  *** WARNING: not 1.0 — this is a very common cause of physics blow-ups! ***")

    print()
    print("=" * 70)
    print("MASS / INERTIA PER RIGID BODY")
    print("=" * 70)
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.MassAPI):
            mass_api = UsdPhysics.MassAPI(prim)
            mass_attr = mass_api.GetMassAttr()
            mass = mass_attr.Get() if mass_attr else None
            diag_attr = mass_api.GetDiagonalInertiaAttr()
            diag_inertia = diag_attr.Get() if diag_attr else None
            flag = ""
            if mass is not None and (mass <= 0.0 or mass > 1000.0):
                flag = "  *** SUSPICIOUS MASS ***"
            print(f"  {prim.GetPath()}  mass={mass}  diagonal_inertia={diag_inertia}{flag}")
        elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
            # rigid body but no explicit mass authored -> physics will compute
            # mass from collision geometry + density, which can be wildly off
            # if the mesh scale doesn't match metersPerUnit.
            print(f"  {prim.GetPath()}  (RigidBodyAPI present, NO explicit MassAPI — mass will be auto-computed)")

    print()


if __name__ == "__main__":
    main()
    simulation_app.close()
