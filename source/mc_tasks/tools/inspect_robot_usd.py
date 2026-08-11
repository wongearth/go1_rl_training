#!/usr/bin/env python3
"""
Extract physical parameters (link masses, inertias, joint drive properties)
from a USD robot file, for comparing two robots (e.g. mc_quadruped vs Go1).

Why this exists
---------------
Light link mass / small inertia + large reset velocities produce huge finite
accelerations (accel = dv/dt), which square into exploding penalties. This
script dumps the numbers that determine whether that will happen, so you can
compare your robot to a known-good one (Go1) side by side.

Usage
-----
Inspect one robot:
    python inspect_robot_usd.py /path/to/quadruped_robot_01.usd

Compare two robots:
    python inspect_robot_usd.py /path/to/mc_quadruped.usd --compare /path/to/go1.usd

Notes
-----
- Must run inside the Isaac Sim / Isaac Lab Python env (it needs `pxr`, the USD
  Python bindings). From an Isaac Lab shell:
      ./isaaclab.sh -p inspect_robot_usd.py <args>
  or with your conda env active:
      python inspect_robot_usd.py <args>
- Reads the USD statically (no simulation), so it's fast and needs no GPU.
"""

import argparse
import sys
from collections import OrderedDict

try:
    from pxr import Usd, UsdPhysics, UsdGeom, Gf
except ImportError:
    print(
        "ERROR: could not import `pxr` (USD Python bindings).\n"
        "Run this inside the Isaac Sim / Isaac Lab Python environment, e.g.:\n"
        "  ./isaaclab.sh -p inspect_robot_usd.py <usd_path>\n"
        "or with your conda env (.env_isaac2_3_0) active.",
        file=sys.stderr,
    )
    sys.exit(1)


def _get_attr(prim, name, default=None):
    """Safely read a USD attribute value."""
    attr = prim.GetAttribute(name)
    if attr and attr.HasAuthoredValue():
        return attr.Get()
    return default


def extract_rigid_bodies(stage):
    """Return {prim_path: {mass, inertia(diag), com, density}} for every rigid body."""
    bodies = OrderedDict()
    for prim in stage.Traverse():
        # a rigid body is a prim with the MassAPI or RigidBodyAPI applied
        has_mass_api = prim.HasAPI(UsdPhysics.MassAPI)
        has_rb_api = prim.HasAPI(UsdPhysics.RigidBodyAPI)
        if not (has_mass_api or has_rb_api):
            continue

        info = {}
        mass = _get_attr(prim, "physics:mass")
        info["mass"] = float(mass) if mass is not None else None

        density = _get_attr(prim, "physics:density")
        info["density"] = float(density) if density is not None else None

        # diagonal inertia (principal moments)
        diag = _get_attr(prim, "physics:diagonalInertia")
        if diag is not None:
            info["inertia_diag"] = tuple(float(x) for x in diag)
        else:
            info["inertia_diag"] = None

        com = _get_attr(prim, "physics:centerOfMass")
        if com is not None:
            info["com"] = tuple(float(x) for x in com)
        else:
            info["com"] = None

        bodies[str(prim.GetPath())] = info
    return bodies


def extract_joints(stage):
    """Return {prim_path: {type, drive stiffness/damping/maxforce, limits}} for joints."""
    joints = OrderedDict()
    for prim in stage.Traverse():
        type_name = prim.GetTypeName()
        if "Joint" not in str(type_name):
            continue
        # skip fixed root joints without drives if you like; here we keep all
        info = {"type": str(type_name)}

        # revolute/prismatic limits
        info["lower_limit"] = _get_attr(prim, "physics:lowerLimit")
        info["upper_limit"] = _get_attr(prim, "physics:upperLimit")

        # drive parameters live on the applied DriveAPI (angular/linear).
        # try both common attribute namespaces.
        for drive_ns in ("angular", "linear"):
            stiffness = _get_attr(prim, f"drive:{drive_ns}:physics:stiffness")
            damping = _get_attr(prim, f"drive:{drive_ns}:physics:damping")
            max_force = _get_attr(prim, f"drive:{drive_ns}:physics:maxForce")
            target_pos = _get_attr(prim, f"drive:{drive_ns}:physics:targetPosition")
            if any(v is not None for v in (stiffness, damping, max_force, target_pos)):
                info[f"{drive_ns}_stiffness"] = stiffness
                info[f"{drive_ns}_damping"] = damping
                info[f"{drive_ns}_maxForce"] = max_force
                info[f"{drive_ns}_targetPos"] = target_pos

        # joint armature (rotor inertia reflected to the joint) — matters a LOT
        # for acceleration stability on light links
        armature = _get_attr(prim, "physxJoint:armature")
        if armature is None:
            armature = _get_attr(prim, "physics:armature")
        info["armature"] = armature

        joints[str(prim.GetPath())] = info
    return joints


def summarize(usd_path):
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        print(f"ERROR: could not open USD stage: {usd_path}", file=sys.stderr)
        sys.exit(1)

    bodies = extract_rigid_bodies(stage)
    joints = extract_joints(stage)
    return bodies, joints


def print_bodies(bodies, title):
    print(f"\n{'='*70}\n{title} — RIGID BODIES ({len(bodies)})\n{'='*70}")
    print(f"{'link':<40}{'mass(kg)':>10}{'inertia_diag (kg·m²)':>28}")
    print("-" * 78)
    total_mass = 0.0
    for path, info in bodies.items():
        name = path.split("/")[-1]
        mass = info.get("mass")
        inertia = info.get("inertia_diag")
        mass_s = f"{mass:.5f}" if mass is not None else "  (none)"
        if mass is not None:
            total_mass += mass
        if inertia is not None:
            inertia_s = "(" + ", ".join(f"{x:.2e}" for x in inertia) + ")"
        else:
            inertia_s = "(none)"
        print(f"{name:<40}{mass_s:>10}{inertia_s:>28}")
    print("-" * 78)
    print(f"{'TOTAL MASS':<40}{total_mass:>10.5f}")


def print_joints(joints, title):
    print(f"\n{'='*70}\n{title} — JOINTS ({len(joints)})\n{'='*70}")
    print(f"{'joint':<32}{'type':<18}{'stiff':>8}{'damp':>8}{'maxF':>9}{'armature':>10}")
    print("-" * 85)
    for path, info in joints.items():
        name = path.split("/")[-1]
        jtype = info.get("type", "").replace("Physics", "").replace("Joint", "")
        stiff = info.get("angular_stiffness") or info.get("linear_stiffness")
        damp = info.get("angular_damping") or info.get("linear_damping")
        maxf = info.get("angular_maxForce") or info.get("linear_maxForce")
        arm = info.get("armature")
        def fmt(v):
            return f"{v:.3g}" if isinstance(v, (int, float)) else "-"
        print(f"{name:<32}{jtype:<18}{fmt(stiff):>8}{fmt(damp):>8}{fmt(maxf):>9}{fmt(arm):>10}")


def print_comparison(bodies_a, bodies_b, name_a, name_b):
    print(f"\n{'='*70}\nMASS / INERTIA COMPARISON: {name_a}  vs  {name_b}\n{'='*70}")

    def stats(bodies):
        masses = [b["mass"] for b in bodies.values() if b.get("mass")]
        inertias = []
        for b in bodies.values():
            if b.get("inertia_diag"):
                inertias.extend(b["inertia_diag"])
        return masses, inertias

    ma, ia = stats(bodies_a)
    mb, ib = stats(bodies_b)

    def line(label, va, vb):
        print(f"{label:<28}{va:>18}{vb:>18}")

    line("num rigid bodies", len(bodies_a), len(bodies_b))
    line("total mass (kg)", f"{sum(ma):.4f}" if ma else "n/a", f"{sum(mb):.4f}" if mb else "n/a")
    line("min link mass (kg)", f"{min(ma):.5f}" if ma else "n/a", f"{min(mb):.5f}" if mb else "n/a")
    line("max link mass (kg)", f"{max(ma):.5f}" if ma else "n/a", f"{max(mb):.5f}" if mb else "n/a")
    line("min inertia (kg·m²)", f"{min(ia):.2e}" if ia else "n/a", f"{min(ib):.2e}" if ib else "n/a")
    line("max inertia (kg·m²)", f"{max(ia):.2e}" if ia else "n/a", f"{max(ib):.2e}" if ib else "n/a")

    print(
        "\nWhy this matters: acceleration = force / mass, and the RL penalties square\n"
        "acceleration/velocity. The link with the SMALLEST mass/inertia is the one\n"
        "that produces the largest accelerations under the same force. If your\n"
        "min link mass is much smaller than Go1's, that link is the explosion source."
    )


def main():
    ap = argparse.ArgumentParser(description="Extract robot physical params from USD.")
    ap.add_argument("usd_path", help="Path to the robot .usd/.usda file")
    ap.add_argument("--compare", metavar="OTHER_USD", default=None,
                    help="Path to a second robot USD to compare against (e.g. Go1)")
    ap.add_argument("--no-joints", action="store_true", help="Skip joint drive dump")
    args = ap.parse_args()

    bodies_a, joints_a = summarize(args.usd_path)
    name_a = args.usd_path.split("/")[-1]

    print_bodies(bodies_a, name_a)
    if not args.no_joints:
        print_joints(joints_a, name_a)

    if args.compare:
        bodies_b, joints_b = summarize(args.compare)
        name_b = args.compare.split("/")[-1]
        print_bodies(bodies_b, name_b)
        if not args.no_joints:
            print_joints(joints_b, name_b)
        print_comparison(bodies_a, bodies_b, name_a, name_b)


if __name__ == "__main__":
    main()
