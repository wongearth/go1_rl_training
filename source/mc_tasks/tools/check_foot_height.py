"""Spawn the robot with REAL gravity and hold the init pose, no RL policy involved.

This isolates whether your actuator gains (stiffness/damping/effort_limit) and
initial spawn height can actually support the robot standing under gravity —
separate from anything about the RL policy or training loop.

Run:
    python spawn_gravity_test.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

# ------------------------------------------------------------------
# 1. CLI args + AppLauncher (must happen BEFORE any isaaclab.sim/assets import)
# ------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Spawn robot with zero gravity for initial-condition check.")
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
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import DCMotorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402


def design_scene() -> Articulation:
    # ground plane + light, purely for visual reference (gravity is zero,
    # so the robot will NOT fall onto the ground — it just floats/holds pose)
    sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func(
        "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    )

    usd_path = os.path.abspath(os.path.expanduser(args_cli.usd_path))
    if not os.path.isfile(usd_path):
        raise FileNotFoundError(f"USD not found at '{usd_path}'. Check --usd_path.")

    robot_cfg = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.35),  # Adjust height if needed
            joint_pos={
                "q0": 0.1, "q3": -0.1, "q6": 0.1, "q9": 0.1,        # hips
                "q1": -0.8, "q4": 0.8, "q7": -0.8, "q10": 0.8,        # upper legs
                "q2": -1.5, "q5": -1.5, "q8": -1.5, "q11": -1.5,    # lower legs
            },
        ),
        actuators={
            "base_legs": DCMotorCfg(
                joint_names_expr=[".*"],  # Use regex to cover all leg joints
                effort_limit=20.0,        # Update based on your robot's specs
                velocity_limit=20.0,
                stiffness=40.0,
                damping=1.0,
                saturation_effort=20.0,
            ),
        },
    )
    return Articulation(robot_cfg)


def main():
    # REAL gravity this time — the actual test.
    sim_cfg = sim_utils.SimulationCfg(dt=0.002, gravity=(0.0, 0.0, -9.81))
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 1.5], target=[0.0, 0.0, 0.35])

    robot = design_scene()

    sim.reset()

    # --- THIS is the part that was missing before ---
    # Setting init_state in the config only stores a *reference* default; it
    # does not by itself push the pose into the physics sim, and it does not
    # set the PD actuator's target. Both must be done explicitly:
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()

    # Measure actual foot (LowerLeg) height vs ground (ground plane is at z=0)
    # BEFORE any physics stepping, so we see the real starting gap in this pose.
    print("\n" + "=" * 70)
    print("FOOT HEIGHT CHECK (ground is at z=0)")
    print("=" * 70)
    for i, name in enumerate(robot.data.body_names):
        if "LowerLeg" in name:
            z = robot.data.body_pos_w[0, i, 2].item()
            print(f"  {name:12s}  world z = {z:7.4f} m   (gap to ground: {z:7.4f} m)")
    print("=" * 70 + "\n")

    print("\n" + "=" * 70)
    print(f"Joint names        : {robot.data.joint_names}")
    print(f"Default joint pos  : {robot.data.default_joint_pos}")
    print(f"Body names         : {robot.data.body_names}")
    print(f"Root initial pos   : {robot.data.default_root_state[:, :3]}")
    print("=" * 70)
    print("\nZero-gravity spawn running — orbit the camera in the viewport")
    print("to inspect the pose. Ctrl+C or close the window to exit.\n")

    step_count = 0
    while simulation_app.is_running():
        # Keep commanding the PD actuators to hold the target pose every step —
        # otherwise stiffness/damping will pull the joints back toward 0.
        robot.set_joint_position_target(joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

        step_count += 1
        if step_count % 50 == 0:  # print once per sim-second at dt=0.002
            root_h = robot.data.root_pos_w[0, 2].item()
            joint_err = (robot.data.joint_pos[0] - joint_pos[0]).abs().max().item()
            print(f"[t={step_count*0.01:5.1f}s] root height: {root_h:6.3f} m   max joint error: {joint_err:6.3f} rad")


if __name__ == "__main__":
    main()
    simulation_app.close()
