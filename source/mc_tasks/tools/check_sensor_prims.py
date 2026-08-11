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


def main():
    usd_path = os.path.abspath(os.path.expanduser(args_cli.usd_path))
    stage = Usd.Stage.Open(usd_path)

    print("\nIsaacContactSensor / IsaacImuSensor prims found:")
    for prim in stage.Traverse():
        if prim.GetTypeName() in ("IsaacContactSensor", "IsaacImuSensor"):
            print(f"  path={prim.GetPath()}  type={prim.GetTypeName()}")
            for attr in prim.GetAttributes():
                if attr.HasAuthoredValue():
                    print(f"      {attr.GetName()} = {attr.Get()}")

    print("\n(no type) prims found:")
    for prim in stage.Traverse():
        if not prim.GetTypeName():
            print(f"  path={prim.GetPath()}  specifier={prim.GetSpecifier()}")


if __name__ == "__main__":
    main()
    simulation_app.close()
