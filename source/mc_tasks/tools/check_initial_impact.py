"""Check whether there's a violent contact-force spike right at spawn.

This directly measures net contact force on the robot's feet every single
physics step for the first ~0.1s, which is the clearest way to tell a real
"free-fall onto ground" apart from "PhysX forcefully pushing an
interpenetrating pose back out" (which looks identical to a fall in a root
height plot, but shows up as a huge, near-instant force spike here).

Run:
    python check_initial_impact.py
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Check for a violent depenetration impulse at spawn.")
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

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import DCMotorCfg  # noqa: E402
from isaaclab.assets import Articulation, ArticulationCfg  # noqa: E402
from isaaclab.sensors import ContactSensor, ContactSensorCfg  # noqa: E402

SIM_DT = 0.002  # confirmed stable earlier; the old dt=0.01 was numerically too coarse


def design_scene():
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
            pos=(0.0, 0.0, 0.32),
                joint_pos={
                "q0": 0.1, "q3": -0.1, "q6": -0.1, "q9": 0.1,        # hips
                 "q1": 0.7, "q4": -0.7, "q7": 0.7, "q10": -0.7,        # upper legs
                 "q2": 1.4, "q5": 1.4, "q8": 1.4, "q11": 1.4,    # lower legs
                },

        ),
        actuators={
            "base_legs": DCMotorCfg(
                joint_names_expr=[".*"],
                effort_limit=20.0,
                velocity_limit=20.0,
                stiffness=40.0,
                damping=1.0,
                saturation_effort=20.0,
            ),
        },
    )
    robot = Articulation(robot_cfg)

    contact_cfg = ContactSensorCfg(
        prim_path="/World/Robot/QuadrupedRobot/.*",
        update_period=0.0,  # update every physics step, not throttled
        history_length=3,
        track_air_time=False,
    )
    contact_sensor = ContactSensor(contact_cfg)

    return robot, contact_sensor


def main():
    sim_cfg = sim_utils.SimulationCfg(dt=SIM_DT, gravity=(0.0, 0.0, -9.81))
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.0, 2.0, 1.5], target=[0.0, 0.0, 0.35])

    robot, contact_sensor = design_scene()

    sim.reset()

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    contact_sensor.reset()

    print("\n" + "=" * 80)
    print(f"{'step':>5} {'t (s)':>8} {'root z (m)':>12} {'max contact force (N)':>24}")
    print("=" * 80)

    step_count = 0
    while simulation_app.is_running():
        robot.set_joint_position_target(joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(SIM_DT)
        contact_sensor.update(SIM_DT)

        step_count += 1

        # Fine-grained: every step for the first 0.1s (50 steps at dt=0.002)
        # Coarse: every ~0.25s after that
        do_print = step_count <= 50 or step_count % 125 == 0
        if do_print:
            root_h = robot.data.root_pos_w[0, 2].item()
            forces = contact_sensor.data.net_forces_w  # [num_envs, num_bodies, 3]
            max_force = forces[0].norm(dim=-1).max().item() if forces is not None else float("nan")
            t = step_count * SIM_DT
            print(f"{step_count:5d} {t:8.3f} {root_h:12.4f} {max_force:24.2f}")


if __name__ == "__main__":
    main()
    simulation_app.close()
