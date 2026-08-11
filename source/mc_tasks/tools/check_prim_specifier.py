"""Check whether specific prims are real definitions or dangling overrides,
and whether the reported 'non-existent path' warnings point at something
that genuinely has no geometry/type behind it.

Run:
    python check_prim_specifier.py --usd_path /path/to/quadruped_robot_01.usd
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check prim specifiers (def vs over) for suspect paths.")
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

from pxr import Usd  # noqa: E402

# Paths taken directly from the warning messages seen during training/play.
# Adjust the leading part if your actual stage root differs (e.g. /World/Robot/... vs
# /World/envs/env_0/Robot/...) — check_body_names.py output showed the real hierarchy.
SUSPECT_PATHS = [
    "/quadruped_robot_01/QuadrupedRobot/UpperLeg3/motor_wo_cover/rotor/Chamfer3",
    "/quadruped_robot_01/QuadrupedRobot/UpperLeg3/Asm_UpperLeg/UpperLeg2/Chamfer1",
]


def report(stage, path):
    prim = stage.GetPrimAtPath(path)
    print(f"\nPath: {path}")
    if not prim:
        print("  -> GetPrimAtPath returned None/invalid object entirely.")
        return
    print(f"  IsValid()    : {prim.IsValid()}")
    if not prim.IsValid():
        return
    print(f"  Specifier    : {prim.GetSpecifier()}  (Def = real definition, Over = override with nothing to override, Class = abstract)")
    print(f"  TypeName     : '{prim.GetTypeName()}'  (empty string is a strong red flag for a dangling over)")
    print(f"  HasAuthoredTypeName : {prim.HasAuthoredTypeName()}")
    print(f"  IsDefined()  : {prim.IsDefined()}")
    # List which layer(s) actually contribute a spec for this prim (composition arcs)
    stack = prim.GetPrimStack()
    print(f"  Contributing layers ({len(stack)}):")
    for spec in stack:
        print(f"    - {spec.layer.identifier}  (specifier={spec.specifier})")


def main():
    usd_path = os.path.abspath(os.path.expanduser(args_cli.usd_path))
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"ERROR: could not open '{usd_path}'")
        return

    for path in SUSPECT_PATHS:
        report(stage, path)

    print("\n" + "=" * 70)
    print("Also scanning the WHOLE stage for any prim with an empty/missing")
    print("type name despite being Def'd — these are the prims most likely to")
    print("throw 'attribute/type not found' warnings during rendering.")
    print("=" * 70)
    count = 0
    for prim in stage.Traverse():
        if prim.GetSpecifier() == Sdf.SpecifierDef and not prim.GetTypeName():
            count += 1
            if count <= 30:  # cap output
                print(f"  {prim.GetPath()}")
    print(f"\nTotal Def'd prims with no type name: {count}")


if __name__ == "__main__":
    main()
    simulation_app.close()
