"""Remove the unused native IsaacContactSensor / IsaacImuSensor prims.

These are leftover from the native Isaac Sim sensor schema and are NOT what
your training pipeline actually uses (that goes through Isaac Lab's own
ContactSensorCfg, a separate mechanism). Confirmed unreferenced by any
reward/termination/event code in this project before writing this script.

Saves to a NEW file (quadruped_robot_01_cleaned.usd) — your original is
left untouched.

Run:
    python remove_unused_sensors.py --usd_path /path/to/quadruped_robot_01.usd
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument(
    "--usd_path",
    type=str,
    default="/home/wong/go1_rl_training/source/mc_tasks/data/robots/mc_quadruped/quadruped_robot_01.usd",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Usd  # noqa: E402

REMOVE_TYPES = ("IsaacContactSensor", "IsaacImuSensor")


def main():
    usd_path = os.path.abspath(os.path.expanduser(args_cli.usd_path))
    stage = Usd.Stage.Open(usd_path)

    to_remove = [prim.GetPath() for prim in stage.Traverse() if prim.GetTypeName() in REMOVE_TYPES]

    if not to_remove:
        print("No matching sensor prims found — nothing to remove.")
        return

    print("Removing the following prims:")
    for path in to_remove:
        print(f"  {path}")

    for path in to_remove:
        stage.RemovePrim(path)

    out_path = usd_path.replace(".usd", "_cleaned.usd")
    stage.GetRootLayer().Export(out_path)
    print(f"\nSaved cleaned USD to: {out_path}")
    print("Update USD_PATH in mc_quadruped_cfg.py (or your test scripts) to point at this file once you've verified it loads correctly.")


if __name__ == "__main__":
    main()
    simulation_app.close()
