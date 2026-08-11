"""Run Isaac Sim's built-in Asset Validator against the robot USD.

This is the authoritative diagnostic tool Isaac Sim ships for exactly this
kind of question ("is something wrong with this USD?") instead of guessing
prim-by-prim.

Run:
    python run_asset_validator.py --usd_path /path/to/quadruped_robot_01.usd
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run the Isaac Sim asset validator on a USD file.")
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

    try:
        from omni.asset_validator.core import ValidationEngine
    except ImportError:
        print("Could not import omni.asset_validator.core — the extension may not be enabled.")
        print("Try enabling it via the Extensions window in Isaac Sim, then rerun this script.")
        return

    stage = Usd.Stage.Open(usd_path)
    engine = ValidationEngine()
    engine.enable_all_rules(True)
    results = engine.validate(stage)

    issues = results.issues()
    print(f"\n{'=' * 70}\nASSET VALIDATOR REPORT: {usd_path}\n{'=' * 70}")
    print(f"Total issues found: {len(issues)}\n")

    for i, issue in enumerate(issues, 1):
        print(f"[{i}] {issue.message}")
        if hasattr(issue, "at") and issue.at is not None:
            print(f"    at: {issue.at}")
        print()

    if not issues:
        print("No issues found by the validator.")


if __name__ == "__main__":
    main()
    simulation_app.close()
