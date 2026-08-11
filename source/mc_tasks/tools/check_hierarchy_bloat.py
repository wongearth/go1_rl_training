"""Measure how many prims and how deep the hierarchy is — checks whether
this USD carries a bloated CAD feature-tree (chamfers, fillets, etc. as
separate scenegraph nodes) rather than clean flattened meshes.

Run:
    python check_hierarchy_bloat.py --usd_path /path/to/quadruped_robot_01.usd
"""

import argparse
import os
from collections import Counter

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

    total_prims = 0
    max_depth = 0
    type_counts = Counter()
    cad_feature_names = 0
    CAD_KEYWORDS = ("chamfer", "fillet", "asm_", "sketch", "extrude", "feature")

    for prim in stage.Traverse():
        total_prims += 1
        depth = len(str(prim.GetPath()).split("/")) - 1
        max_depth = max(max_depth, depth)
        type_counts[prim.GetTypeName() or "(no type)"] += 1
        name_lower = prim.GetName().lower()
        if any(kw in name_lower for kw in CAD_KEYWORDS):
            cad_feature_names += 1

    print(f"\n{'=' * 70}\nHIERARCHY STATS: {usd_path}\n{'=' * 70}")
    print(f"Total prims          : {total_prims}")
    print(f"Max hierarchy depth  : {max_depth}")
    print(f"Prims with CAD-feature-like names (chamfer/fillet/asm_/etc.): {cad_feature_names}")
    print(f"\nType breakdown:")
    for type_name, count in type_counts.most_common():
        print(f"  {type_name:20s} : {count}")


if __name__ == "__main__":
    main()
    simulation_app.close()
