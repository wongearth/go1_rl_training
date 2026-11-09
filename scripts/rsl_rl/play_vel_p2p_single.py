# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Velocity-based Point-to-Point playback for RSL-RL.

Uses a *velocity-tracking* policy to perform a point-to-point timed reach by
generating a cubic-spline (rest-to-rest) velocity profile in the world frame
from the robot's current pose to a random target pose within a target time,
then feeding that profile -- rotated into the current base frame -- to the
policy as (vx, vy, wz) commands each step.

This is the "velocity RL + open-loop path planning" baseline, meant to be
compared directly against the P2P-timed policy. Error reporting/plots match
play_p2p_time_single3_all.py so the two are directly comparable.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher
import numpy as np
# local imports
import cli_args  # isort: skip
import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- works headless, no display needed
import matplotlib.pyplot as plt
import math

from isaaclab.utils.math import euler_xyz_from_quat
import torch

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a velocity policy as a P2P timed reach with cubic-spline velocity profiles.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate (default 1 for single-robot viewing).")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--max_steps", type=int, default=None, help="Total steps to run before auto-exiting (default: video_length+50 if --video, else 100000).")
parser.add_argument(
    "--plot_mode", type=str, default="save", choices=["save", "live", "both"],
    help="How to output the per-episode error plot: 'save' writes PNG files (default, headless-safe), "
         "'live' shows an interactive updating window instead, 'both' does both."
)
parser.add_argument("--export_csv", type=str, default="",
                    help="If set, write per-step time-series (episode, time, err_x/y/yaw, height, roll, pitch) to this CSV.")
parser.add_argument("--ortho_size", type=float, default=0.0,
                    help="If > 0, use an orthographic top-down camera with this vertical view span in meters "
                         "(e.g. 3.0 tight on the robot, 12.0 wide). 0 = leave the perspective camera as-is.")
# --- P2P-via-velocity specific args ---
parser.add_argument("--target_time_range", type=float, nargs=2, default=[2.5, 4.0], help="Random per-episode time budget (s) for each cubic-spline reach; sampled uniformly.")
parser.add_argument("--target_x_range", type=float, nargs=2, default=[-2.0, 2.0], help="Target x range (m, world/base-init frame).")
parser.add_argument("--target_y_range", type=float, nargs=2, default=[-2.0, 2.0], help="Target y range (m).")
parser.add_argument("--target_yaw_range", type=float, nargs=2, default=[-math.pi, math.pi], help="Target yaw range (rad).")
parser.add_argument("--hold_time", type=float, default=1.0, help="Extra time (s) held at zero command after the reach, before resampling a new target.")
parser.add_argument("--max_lin_vel", type=float, default=1.0, help="Clamp on planned linear velocity command magnitude (m/s).")
parser.add_argument("--max_ang_vel", type=float, default=1.0, help="Clamp on planned yaw-rate command (rad/s).")
# --- batch / CSV comparison args ---
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to run before exiting (batch mode).")
parser.add_argument("--commands_csv", type=str, default="p2p_commands.csv", help="Shared command file; created (seeded) if missing, else reused.")
parser.add_argument("--commands_seed", type=int, default=42, help="Seed used ONLY when creating the shared command file.")
parser.add_argument("--results_csv", type=str, default="results_velocity.csv", help="Per-episode results output for this (velocity) run.")
parser.add_argument("--play_episode", type=int, default=None,
                    help="Play ONLY this row index from the commands CSV, looping it forever so you can watch it. "
                         "Omit to play all rows once (or use --cycle to loop through all rows forever).")
parser.add_argument("--cycle", action="store_true", default=False,
                    help="Cycle through all rows 0->end->0 forever (ignored if --play_episode is set).")
parser.add_argument("--once", action="store_true", default=False,
                    help="With --play_episode N: play row N exactly once, write its result, and exit "
                         "(instead of looping). Use in a shell loop for one-fresh-process-per-episode collection.")
parser.add_argument("--append", action="store_true", default=False,
                    help="Append to --results_csv instead of overwriting (write header only if file is new). "
                         "Lets many --once runs accumulate into one results file.")
parser.add_argument("--random_targets", action="store_true", default=False,
                    help="Sample targets randomly from the ranges (seeded by --seed) instead of reading "
                         "--commands_csv. Like the P2P play script's default behavior.")
parser.add_argument("--target_seed", type=int, default=42,
                    help="RNG seed for --random_targets (change for a different sequence).")
parser.add_argument("--ts_episode", type=int, default=None,
                    help="If set, ONLY save timeseries data for this specific episode index (prevents giant CSVs).")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
print(f"[DEBUG] hydra_args = {hydra_args}")
print(f"[DEBUG] full sys.argv = {sys.argv}")

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Restore clean args after AppLauncher (it can mutate sys.argv and corrupt Hydra).
sys.argv = [sys.argv[0]] + hydra_args

# =========================================================
# PASTE THE IMPORTS HERE (After simulation_app boots!)
# =========================================================
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils.math import quat_apply_yaw, quat_from_euler_xyz, yaw_quat, wrap_to_pi
# =========================================================


"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import go1_tasks
import mc_tasks

import csv
# shared command generator/loader (must sit next to this script)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_commands import ensure_commands


# =========================================================
# CUBIC SPLINE (rest-to-rest) VELOCITY PROFILE
# =========================================================
def cubic_rest_to_rest(p0, p1, T):
    """Return coefficients for a scalar clamped cubic from p0->p1 over [0,T]
    with zero endpoint velocity.

        p(t) = a0 + a1*t + a2*t^2 + a3*t^3
        p(0)=p0, p(T)=p1, p'(0)=0, p'(T)=0

    Returns (a1, a2, a3) for the *velocity* form p'(t) = a1 + 2*a2*t + 3*a3*t^2.
    (a0 not needed since we only ever evaluate velocity.)
    """
    d = p1 - p0
    a2 = 3.0 * d / (T ** 2)
    a3 = -2.0 * d / (T ** 3)
    a1 = 0.0
    return a1, a2, a3


def cubic_vel(a1, a2, a3, t):
    """Evaluate the cubic velocity p'(t) = a1 + 2*a2*t + 3*a3*t^2."""
    return a1 + 2.0 * a2 * t + 3.0 * a3 * t * t


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play a velocity policy as a P2P timed reach."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    # force a single robot for viewing (override the training-time default)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # world-anchored camera (does NOT follow the robot) -- fixed view of the
    # arena so you can watch the robot walk across the frame toward its target.
    env_cfg.viewer.origin_type = "world"
    # angled overhead view: high and back, looking at the origin where the robot
    # starts. Good for seeing the robot tip/fall (unlike pure top-down).
    env_cfg.viewer.eye = (0.0, 0.00, 10.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)


    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # ==========================================================
    # ORTHOGRAPHIC TOP-DOWN CAMERA (optional, --ortho_size > 0)
    # Orthographic = zero perspective distortion; the robot stays the same
    # apparent size regardless of distance, and ortho_size sets exactly how much
    # ground is visible (the view limit). Small = zoomed in, still perfectly flat.
    # ==========================================================
    if args_cli.ortho_size and args_cli.ortho_size > 0.0:
        try:
            import omni.usd
            from pxr import UsdGeom
            stage = omni.usd.get_context().get_stage()
            cam_path = "/OmniverseKit_Persp"
            cam_prim = stage.GetPrimAtPath(cam_path)
            if cam_prim and cam_prim.IsValid():
                cam = UsdGeom.Camera(cam_prim)
                cam.GetProjectionAttr().Set("orthographic")
                cam.GetHorizontalApertureAttr().Set(args_cli.ortho_size * 10.0)
                cam.GetVerticalApertureAttr().Set(args_cli.ortho_size * 10.0)
                print(f"[CAMERA] Orthographic top-down set, view span = {args_cli.ortho_size} m")
            else:
                print(f"[CAMERA][WARN] camera prim {cam_path} not found; staying perspective.")
        except Exception as e:
            print(f"[CAMERA][WARN] could not set orthographic camera: {e}")
    # ==========================================================

    # sanity: the velocity policy expects a base_velocity command term.
    try:
        term = env.unwrapped.command_manager.get_term("base_velocity")
        print(">>> heading_command =", term.cfg.heading_command)
        print(">>> resampling_time_range =", term.cfg.resampling_time_range)
    except Exception as e:
        print(f"[WARN] Could not fetch 'base_velocity' command term: {e}")
        print("[WARN] This script expects a velocity-tracking task. Check --task.")

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module (2.3 onwards vs 2.2 and below)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    device = env.unwrapped.device

    # direct handle to the robot for ground-truth state
    base_env = env.unwrapped
    robot = base_env.scene["robot"]

    # locate the base_velocity command term object so we can OVERRIDE its command
    # tensor each step with our spline-generated (vx, vy, wz).
    cmd_term = base_env.command_manager.get_term("base_velocity")

    def reset_to_origin():
        """Reset env 0 and place the robot at the floor's origin (x=y=0, yaw=0),
        keeping the default spawn height/orientation-uprightness and zero velocity.
        Returns fresh observations."""
        with torch.inference_mode():
            env.reset()
            # env-0 origin in world (accounts for the scene's per-env spacing)
            env_origin = base_env.scene.env_origins[0]
            root_state = robot.data.default_root_state[0:1].clone()
            # position: put base XY exactly at env-0 origin, keep default spawn height
            root_state[0, 0] = env_origin[0]
            root_state[0, 1] = env_origin[1]
            # orientation: identity yaw (default_root_state already upright)
            root_state[0, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
            # zero all velocities
            root_state[0, 7:13] = 0.0
            idx = torch.tensor([0], device=device)
            robot.write_root_pose_to_sim(root_state[:, 0:7], env_ids=idx)
            robot.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=idx)
            base_env.sim.forward()
            robot.update(dt=0.0)
            return env.get_observations()

    # reset environment -- start at floor origin
    obs = reset_to_origin()
    timestep = 0

    # ==========================================================
    # AUTO-EXIT
    # ==========================================================
    if args_cli.max_steps is not None:
        max_steps = args_cli.max_steps
    elif args_cli.video:
        max_steps = args_cli.video_length + 50
    else:
        max_steps = 100000
    print(f"[INFO]: Will auto-exit after {max_steps} steps.")
    print("[INFO]: Setup complete. Starting playback...")

    # ==========================================================
    # SETUP COORDINATE FRAME MARKER (target pose)
    # ==========================================================
    frame_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/TargetFrame")
    frame_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    target_frame_marker = VisualizationMarkers(frame_cfg)
    # ==========================================================

    # ==========================================================
    # DATA TRACKING VARIABLES FOR PLOTTING (match P2P script)
    # ==========================================================
    history_err_x = []
    history_err_y = []
    history_err_yaw = []
    history_height = []   # actual root height (cm)
    history_roll = []     # actual body roll (deg)
    history_pitch = []    # actual body pitch (deg)
    history_time = []
    history_vx = []
    history_vy = []
    history_vz = []
    history_wx = []
    history_wy = []
    history_wz = []

    current_episode_time = 0.0
    episode_count = 0

    # optional per-step CSV export
    csv_file = None
    csv_writer = None
    if args_cli.export_csv:
        csv_file = open(args_cli.export_csv, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "episode", "time_s", "target_time_s", 
            "err_x_cm", "err_y_cm", "err_yaw_deg", 
            "pos_x", "pos_y", "pos_z", 
            "roll_deg", "pitch_deg", "yaw_deg", 
            "vx", "vy", "vz", "wx", "wy", "wz"
        ])
        print(f"[CSV] exporting per-step data to {args_cli.export_csv}")

    # Auto-save the per-episode commands alongside the export CSV so the P2P
    # script can replay the exact same targets for a matched comparison.
    cmd_csv_file = None
    cmd_csv_writer = None
    if args_cli.export_csv:
        cmd_csv_path = os.path.splitext(args_cli.export_csv)[0] + "_commands.csv"
        cmd_csv_file = open(cmd_csv_path, "w", newline="")
        cmd_csv_writer = csv.writer(cmd_csv_file)
        cmd_csv_writer.writerow(["cmd_x", "cmd_y", "cmd_yaw", "target_time"])
        print(f"[CSV] saving per-episode commands to {cmd_csv_path}")

    # plot mode
    if args_cli.plot_mode in ("live", "both"):
        try:
            plt.switch_backend("TkAgg")
            plt.ion()
            print("[INFO] Live plot window enabled (--plot_mode live/both).")
        except Exception as e:
            print(f"[WARN] Could not enable live plotting (no display available?): {e}")
            print("[WARN] Falling back to --plot_mode save.")
            args_cli.plot_mode = "save"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # ==========================================================
    # P2P-VIA-VELOCITY STATE
    # ==========================================================
    hold_time = float(args_cli.hold_time)

    # ---- TARGET SOURCE ----
    # --random_targets: sample from the ranges with a seeded RNG (like P2P play).
    # otherwise: read from --commands_csv (shared, reproducible).
    target_rng = None
    commands = None
    if args_cli.random_targets:
        target_rng = np.random.default_rng(int(args_cli.target_seed))
        print(f"[TARGETS] random mode, seed={args_cli.target_seed} "
              f"(change --seed for a different sequence)")
    elif args_cli.cycle:
        from generate_commands import load_commands_csv
        if not os.path.exists(args_cli.commands_csv):
            raise RuntimeError(f"[cycle] {args_cli.commands_csv} does not exist. "
                               f"Point --commands_csv at an existing file (e.g. outlier_commands.csv).")
        commands = load_commands_csv(args_cli.commands_csv)
        print(f"[cycle] loaded {len(commands)} rows from {args_cli.commands_csv}; cycling forever.")
    else:
        n_required = (args_cli.play_episode + 1) if args_cli.play_episode is not None else args_cli.num_episodes
        commands = ensure_commands(
            args_cli.commands_csv, n_required, args_cli.commands_seed,
            x_range=tuple(args_cli.target_x_range),
            y_range=tuple(args_cli.target_y_range),
            yaw_range=tuple(args_cli.target_yaw_range),
            time_range=tuple(args_cli.target_time_range),
        )

    # open results CSV (append mode accumulates many --once runs into one file)
    write_header = True
    if args_cli.append and os.path.exists(args_cli.results_csv) and os.path.getsize(args_cli.results_csv) > 0:
        write_header = False
    results_file = open(args_cli.results_csv, "a" if args_cli.append else "w", newline="")
    results_writer = csv.writer(results_file)
    if write_header:
        results_writer.writerow(["cmd_x", "cmd_y", "cmd_yaw", "target_time",
                                 "err_x", "err_y", "err_yaw"])
    print(f"[INFO] {'Appending' if args_cli.append else 'Writing'} per-episode results to: {args_cli.results_csv}")

    def sample_target_and_plan():
        """Build the plan for episode `episode_count` from the SHARED command row.

        cmd_x, cmd_y, cmd_yaw are base-frame relative targets; the robot starts
        at the floor origin (yaw~0) each episode, so base frame == world frame at
        planning time. Returns a dict holding the plan + absolute world target.
        """
        # current world pose (env 0) -- robot is at origin after reset_to_origin()
        p_now = robot.data.root_pos_w[0, :2].detach().cpu().numpy()  # (x, y)
        quat_now = robot.data.root_quat_w[0]
        _, _, yaw_now_t = euler_xyz_from_quat(quat_now.unsqueeze(0))
        yaw_now = float(yaw_now_t[0].item())

        # ---- choose this episode's target ----
        if target_rng is not None:
            # random_targets mode: sample from the ranges
            dx_b = float(target_rng.uniform(*args_cli.target_x_range))
            dy_b = float(target_rng.uniform(*args_cli.target_y_range))
            dyaw = float(target_rng.uniform(*args_cli.target_yaw_range))
            target_time = float(target_rng.uniform(*args_cli.target_time_range))
            print(f"[TARGET] random ep{episode_count}: dx={dx_b:.2f} dy={dy_b:.2f} "
                  f"dyaw={math.degrees(dyaw):.0f}deg T={target_time:.2f}s")
        else:
            # CSV mode
            if args_cli.play_episode is not None:
                row_idx = args_cli.play_episode
            else:
                row_idx = episode_count % len(commands)
            dx_b, dy_b, dyaw, target_time = commands[row_idx]
            print(f"[CYCLE] playing row {row_idx} of {len(commands)}")

        # auto-save the command used (for paired comparison with the other script)
        if cmd_csv_writer is not None:
            cmd_csv_writer.writerow([f"{dx_b:.6f}", f"{dy_b:.6f}", f"{dyaw:.6f}", f"{target_time:.6f}"])
            cmd_csv_file.flush()

        # rotate base-frame offset into world
        cos_y, sin_y = math.cos(yaw_now), math.sin(yaw_now)
        tx = p_now[0] + cos_y * dx_b - sin_y * dy_b
        ty = p_now[1] + sin_y * dx_b + cos_y * dy_b
        tyaw = wrap_to_pi(torch.tensor(yaw_now + dyaw)).item()

        # cubic coefficients (world frame, rest-to-rest over target_time)
        ax1_, ax2_, ax3_ = cubic_rest_to_rest(p_now[0], tx, target_time)
        ay1_, ay2_, ay3_ = cubic_rest_to_rest(p_now[1], ty, target_time)
        # for yaw, go the short way: plan over the wrapped delta
        yaw_delta = wrap_to_pi(torch.tensor(tyaw - yaw_now)).item()
        aw1_, aw2_, aw3_ = cubic_rest_to_rest(0.0, yaw_delta, target_time)

        plan = {
            "target_time": target_time,
            "cmd": (dx_b, dy_b, dyaw),   # base-frame command as stored in the CSV
            "target_w": np.array([tx, ty, tyaw]),
            "start_w": np.array([p_now[0], p_now[1], yaw_now]),
            "coef_x": (ax1_, ax2_, ax3_),
            "coef_y": (ay1_, ay2_, ay3_),
            "coef_yaw": (aw1_, aw2_, aw3_),
        }
        print(f"[DEBUG] Ep {episode_count} target -> dx_b={dx_b:.2f} dy_b={dy_b:.2f} dyaw={math.degrees(dyaw):.1f}deg "
              f"| world target=({tx:.2f}, {ty:.2f}, {math.degrees(tyaw):.1f}deg) | T={target_time:.2f}s")
        return plan

    def draw_target_marker(plan):
        tx, ty, tyaw = plan["target_w"]
        pos = torch.tensor([[tx, ty, 0.45]], device=device, dtype=torch.float32)
        quat = quat_from_euler_xyz(
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
            torch.tensor([tyaw], device=device, dtype=torch.float32),
        )
        target_frame_marker.visualize(pos, quat)

    plan = sample_target_and_plan()
    draw_target_marker(plan)

    # per-episode budget + error-at-target-time capture
    target_time = plan["target_time"]
    episode_len = target_time + hold_time
    captured_at_target = False
    tgt_err_dist = None   # distance error (m) sampled exactly at target time
    tgt_err_yaw = None    # yaw error (rad) sampled exactly at target time
    tgt_err_x = None      # signed base-frame x error (m) at target time
    tgt_err_y = None      # signed base-frame y error (m) at target time
    # diagnostic: where did the robot start this episode (world xy)
    start_xy = robot.data.root_pos_w[0, :2].clone()
    ep_peak_cmd = 0.0     # peak commanded base-frame speed this episode

    # ==========================================================
    # Simulate environment
    # ==========================================================
    while simulation_app.is_running() and timestep < max_steps:
        with torch.inference_mode():
            # ---------- BUILD VELOCITY COMMAND FROM SPLINE ----------
            t_plan = min(current_episode_time, target_time)  # clamp to reach window
            in_reach = current_episode_time <= target_time

            if in_reach:
                vx_w = cubic_vel(*plan["coef_x"], t_plan)
                vy_w = cubic_vel(*plan["coef_y"], t_plan)
                wz = cubic_vel(*plan["coef_yaw"], t_plan)
            else:
                vx_w = vy_w = wz = 0.0  # hold phase

            # rotate world-frame planned linear velocity into CURRENT base frame,
            # because the policy consumes base-frame (vx, vy) commands.
            quat_now = robot.data.root_quat_w[0]
            _, _, yaw_now_t = euler_xyz_from_quat(quat_now.unsqueeze(0))
            yaw_now = float(yaw_now_t[0].item())
            cos_y, sin_y = math.cos(yaw_now), math.sin(yaw_now)
            vx_b = cos_y * vx_w + sin_y * vy_w
            vy_b = -sin_y * vx_w + cos_y * vy_w

            # clamp to policy's trained command envelope
            lin_norm = math.hypot(vx_b, vy_b)
            if lin_norm > args_cli.max_lin_vel and lin_norm > 1e-6:
                scale = args_cli.max_lin_vel / lin_norm
                vx_b *= scale
                vy_b *= scale
            wz = max(-args_cli.max_ang_vel, min(args_cli.max_ang_vel, wz))
            ep_peak_cmd = max(ep_peak_cmd, math.hypot(vx_b, vy_b))

            # OVERRIDE the command term's buffer (env 0). Shape: (num_envs, 3+).
            cmd_term.command[0, 0] = vx_b
            cmd_term.command[0, 1] = vy_b
            cmd_term.command[0, 2] = wz

            # ---------- CAPTURE ERROR *BEFORE* STEP ----------
            # target-relative error in the robot's CURRENT base frame (cm / deg),
            # computed from ground-truth world state (no base_pose command exists).
            p_now = robot.data.root_pos_w[0, :2]
            tgt = torch.tensor(plan["target_w"][:2], device=device, dtype=torch.float32)
            err_w = tgt - p_now  # world-frame position error
            err_bx = (cos_y * err_w[0] + sin_y * err_w[1]).item()
            err_by = (-sin_y * err_w[0] + cos_y * err_w[1]).item()
            yaw_err = wrap_to_pi(torch.tensor(plan["target_w"][2] - yaw_now)).item()

            final_dist = math.hypot(err_bx, err_by)
            final_yaw = yaw_err

            # ---------- CAPTURE ERROR EXACTLY AT TARGET TIME ----------
            # The first step whose timestamp reaches target_time is the reported
            # reach error, regardless of the extra hold phase that follows.
            if (not captured_at_target) and (current_episode_time >= target_time):
                tgt_err_dist = final_dist
                tgt_err_yaw = final_yaw
                tgt_err_x = err_bx      # signed base-frame components at target time
                tgt_err_y = err_by
                captured_at_target = True
                target_time_mark = current_episode_time  # actual t of the sample

            # actual body state (height, roll, pitch)
            current_height = robot.data.root_pos_w[0, 2].item()
            roll_t, pitch_t, _ = euler_xyz_from_quat(robot.data.root_quat_w)
            current_roll = roll_t[0].item()
            current_pitch = pitch_t[0].item()

            # ---------- STEP ----------
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        # ---------- RECORD FOR PLOT ----------
        history_err_x.append(err_bx * 100.0)
        history_err_y.append(err_by * 100.0)
        history_err_yaw.append(yaw_err * (180.0 / math.pi))
        history_height.append(current_height * 100.0)
        history_roll.append(current_roll * (180.0 / math.pi))
        history_pitch.append(current_pitch * (180.0 / math.pi))
        history_time.append(current_episode_time)
        history_vx.append(robot.data.root_lin_vel_b[0, 0].item()* 100.0)
        history_vy.append(robot.data.root_lin_vel_b[0, 1].item()* 100.0)
        history_vz.append(robot.data.root_lin_vel_b[0, 2].item()* 100.0)
        history_wx.append(robot.data.root_ang_vel_b[0, 0].item()* (180.0 / math.pi))
        history_wy.append(robot.data.root_ang_vel_b[0, 1].item()* (180.0 / math.pi))
        history_wz.append(robot.data.root_ang_vel_b[0, 2].item()* (180.0 / math.pi))

        # per-step CSV row
        # ---------- CONDITIONAL PER-STEP CSV EXPORT ----------
        if csv_writer is not None:
            # Only write if no specific episode was requested, OR if it matches the current episode
            if args_cli.ts_episode is None or episode_count == args_cli.ts_episode:
                # Extract global position
                pos_x = robot.data.root_pos_w[0, 0].item()
                pos_y = robot.data.root_pos_w[0, 1].item()
                pos_z = robot.data.root_pos_w[0, 2].item()
                
                # Extract local base velocities
                vx = robot.data.root_lin_vel_b[0, 0].item()
                vy = robot.data.root_lin_vel_b[0, 1].item()
                vz = robot.data.root_lin_vel_b[0, 2].item()
                wx = robot.data.root_ang_vel_b[0, 0].item()
                wy = robot.data.root_ang_vel_b[0, 1].item()
                wz = robot.data.root_ang_vel_b[0, 2].item()

                csv_writer.writerow([
                    # episode_count, f"{current_episode_time:.4f}", f"{target_time:.4f}",
                    args_cli.play_episode if args_cli.play_episode is not None else episode_count, f"{current_episode_time:.4f}", f"{target_time:.4f}",
                    f"{err_bx*100.0:.4f}", f"{err_by*100.0:.4f}", f"{yaw_err*(180.0/math.pi):.4f}",
                    f"{pos_x:.4f}", f"{pos_y:.4f}", f"{pos_z:.4f}",
                    f"{current_roll*(180.0/math.pi):.4f}", f"{current_pitch*(180.0/math.pi):.4f}", f"{yaw_now*(180.0/math.pi):.4f}",
                    f"{vx:.4f}", f"{vy:.4f}", f"{vz:.4f}", 
                    f"{wx:.4f}", f"{wy:.4f}", f"{wz:.4f}"
                ])

        current_episode_time += dt

        # ---------- EPISODE END: fixed budget elapsed, or env terminated ----------
        episode_over = (current_episode_time >= episode_len) or dones[0].item()

        if episode_over:
            # report the error sampled EXACTLY at target time (fall back to the
            # latest error if the env terminated before target time was reached).
            if captured_at_target:
                rep_dist, rep_yaw = tgt_err_dist, tgt_err_yaw
                rep_x, rep_y = tgt_err_x, tgt_err_y
                mark = target_time_mark
            else:
                rep_dist, rep_yaw = final_dist, final_yaw
                rep_x, rep_y = err_bx, err_by
                mark = current_episode_time
            print(f"[DEBUG] Episode Complete! Error @ target time -> Distance: {rep_dist*100:.2f} cm | Yaw: {abs(rep_yaw)/math.pi*180:.1f} deg")
            print(f"[DEBUG] Target Time was: {target_time:.2f}s (sampled at t={mark:.2f}s) | Episode lasted: {current_episode_time:.2f}s")
            # ---- STALL/DRIFT DIAGNOSTIC ----
            moved = torch.norm(robot.data.root_pos_w[0, :2] - start_xy).item()
            cmd_dist = math.hypot(plan["cmd"][0], plan["cmd"][1])
            print(f"[DIAG] robot moved {moved:.2f} m  (commanded reach {cmd_dist:.2f} m)  "
                  f"peak cmd speed {ep_peak_cmd:.2f} m/s")
            if moved < 0.1:
                print("[DIAG]  -> STALLED: robot barely moved (no feedback to recover).")
            elif rep_dist > 0.4:
                print("[DIAG]  -> MOVED but missed: drifted / wrong direction / overshoot.")

            # ---------- WRITE RESULT ROW (command + error at target time) ----------
            cmd_x, cmd_y, cmd_yaw = plan["cmd"]
            results_writer.writerow([
                f"{cmd_x:.6f}", f"{cmd_y:.6f}", f"{cmd_yaw:.6f}", f"{target_time:.6f}",
                f"{rep_x:.6f}", f"{rep_y:.6f}", f"{rep_yaw:.6f}",
            ])
            results_file.flush()

            if len(history_time) > 1:
                ax1.clear()
                ax2.clear()

                # Top subplot: position-related, all in cm
                ax1.plot(history_time, history_err_x, label="Error X (cm)", linewidth=2)
                ax1.plot(history_time, history_err_y, label="Error Y (cm)", linewidth=2)
                ax1.plot(history_time, history_height, label="Body Height (cm)", linewidth=2, color='green')
                ax1.axvline(x=target_time, color='red', linestyle=':', linewidth=2, label=f"Target Time ({target_time:.1f}s)")
                ax1.axhline(0, color='black', linestyle='-', linewidth=1)
                ax1.set_title("Position Error and Body Height over Time (Velocity-based P2P, cubic spline)", fontsize=13)
                ax1.set_ylabel("cm", fontsize=12)
                ax1.legend(loc="best")
                ax1.grid(True, alpha=0.3)

                # Bottom subplot: orientation-related, all in degrees
                ax2.plot(history_time, history_err_yaw, label="Error Yaw (deg)", linewidth=2, linestyle='--')
                ax2.plot(history_time, history_roll, label="Roll (deg)", linewidth=2, color='purple')
                ax2.plot(history_time, history_pitch, label="Pitch (deg)", linewidth=2, color='brown')
                ax2.axvline(x=target_time, color='red', linestyle=':', linewidth=2)
                ax2.axhline(0, color='black', linestyle='-', linewidth=1)
                ax2.set_title("Yaw Error, Roll, and Pitch over Time", fontsize=14)
                ax2.set_xlabel("Time Elapsed (Seconds)", fontsize=12)
                ax2.set_ylabel("degrees", fontsize=12)
                ax2.legend(loc="best")
                ax2.grid(True, alpha=0.3)

                fig.tight_layout()

                if args_cli.plot_mode in ("save", "both"):
                    plot_path = os.path.join(log_dir, f"vel_p2p_episode_plot_{episode_count:03d}.png")
                    fig.savefig(plot_path)
                    print(f"[DEBUG] Saved plot to: {plot_path}")

                if args_cli.plot_mode in ("live", "both"):
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.001)

            episode_count += 1

            # reset tracking
            history_err_x.clear()
            history_err_y.clear()
            history_err_yaw.clear()
            history_height.clear()
            history_roll.clear()
            history_pitch.clear()
            history_time.clear()
            current_episode_time = 0.0

            # exit logic:
            #  --play_episode N : loop that one row forever (never exit)
            #  --cycle          : loop through all rows forever (never exit)
            #  neither          : play all rows once, then exit after num_episodes
            if args_cli.play_episode is not None and args_cli.once:
                print(f"[INFO] --once: played episode {args_cli.play_episode} one time. Exiting.")
                break
            if args_cli.play_episode is not None:
                print(f"[INFO] Replaying episode {args_cli.play_episode} (loop #{episode_count}). "
                      f"Ctrl-C or close window to stop.")
            elif args_cli.cycle:
                if episode_count % len(commands) == 0:
                    print(f"[INFO] Completed a full cycle of {len(commands)} rows; looping again. "
                          f"Ctrl-C or close window to stop.")
            elif episode_count >= args_cli.num_episodes:
                print(f"[INFO] Completed {episode_count} episodes. Results in {args_cli.results_csv}.")
                break

            # start every episode from the floor origin
            obs = reset_to_origin()

            plan = sample_target_and_plan()
            draw_target_marker(plan)

            # refresh per-episode budget + error-capture state
            target_time = plan["target_time"]
            episode_len = target_time + hold_time
            captured_at_target = False
            tgt_err_dist = None
            tgt_err_yaw = None
            tgt_err_x = None
            tgt_err_y = None
            start_xy = robot.data.root_pos_w[0, :2].clone()
            ep_peak_cmd = 0.0
            if csv_file is not None:
                csv_file.flush()

        timestep += 1

    results_file.close()
    if csv_file is not None:
        csv_file.close()
        print(f"[CSV] wrote {args_cli.export_csv}")
    if cmd_csv_file is not None:
        cmd_csv_path = os.path.splitext(args_cli.export_csv)[0] + "_commands.csv"
        cmd_csv_file.close()
        print(f"[CSV] wrote {cmd_csv_path}")
    print(f"[INFO]: Reached {timestep} steps -- exiting.")

    # close the simulator
    env.close()


if __name__ == "__main__":
    print(f"[DEBUG] sys.argv right before main(): {sys.argv}")
    # run the main function
    main()
    # close sim app
    simulation_app.close()
