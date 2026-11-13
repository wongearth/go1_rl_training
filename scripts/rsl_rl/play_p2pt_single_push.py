# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher
import numpy as np
# local imports
import cli_args  # isort: skip
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — works headless, no display needed
import matplotlib.pyplot as plt
import math

from isaaclab.utils.math import euler_xyz_from_quat
import torch
from isaaclab.utils.math import quat_apply_yaw, quat_from_euler_xyz, quat_mul, yaw_quat, quat_apply_inverse
import csv

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
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
# --- noise injection on the pose_commands observation (imperfect estimator) ---
parser.add_argument("--noise_model", type=str, default="none",
                    choices=["none", "gaussian", "drift"],
                    help="Noise on the pose-error feedback the policy reads. 'none'=clean; "
                         "'gaussian'=per-step N(0,sigma) jitter; 'drift'=jitter + accumulating random-walk bias.")
parser.add_argument("--noise_k", type=float, default=1.0, help="Scale on the base sigmas (0=clean).")
parser.add_argument("--sigma_xy", type=float, default=0.02, help="Base std (m) for dx,dy at k=1 (default 2cm).")
parser.add_argument("--sigma_yaw", type=float, default=0.026, help="Base std (rad) for heading at k=1 (~1.5deg).")
parser.add_argument("--drift_xy_per_s", type=float, default=0.01, help="Drift RW std (m/sqrt-s) for dx,dy at k=1.")
parser.add_argument("--drift_yaw_per_s", type=float, default=0.009, help="Drift RW std (rad/sqrt-s) for heading at k=1.")
parser.add_argument("--noise_seed", type=int, default=0, help="Seed for the noise RNG.")
# --- proprioceptive sensor noise (extra Gaussian on lin_vel/ang_vel/gravity) ---
parser.add_argument("--sensor_noise", type=str, default="",
                    help="Comma list of channels to add extra noise to: lin_vel,ang_vel,gravity,all. Empty=none.")
parser.add_argument("--sensor_k", type=float, default=1.0, help="Scale on sensor sigmas (0 disables).")
parser.add_argument("--sigma_lin_vel", type=float, default=0.1, help="Extra std (m/s) on base_lin_vel at sensor_k=1.")
parser.add_argument("--sigma_ang_vel", type=float, default=0.2, help="Extra std (rad/s) on base_ang_vel at sensor_k=1.")
parser.add_argument("--sigma_gravity", type=float, default=0.05, help="Extra std on projected_gravity at sensor_k=1.")
# --- CSV export of the per-step plot data ---
parser.add_argument("--export_csv", type=str, default="",
                    help="If set, write per-step time-series (time, err_x/y/yaw, height, roll, pitch) to this CSV. "
                         "One file per run; episodes separated by an 'episode' column.")
# --- optional: read fixed targets from a shared commands CSV ---
# Default behavior is UNCHANGED (env samples random targets). Only when
# --commands_csv is given does it inject those fixed targets instead.
parser.add_argument("--commands_csv", type=str, default="",
                    help="If set, inject targets from this CSV (base-frame dx,dy,dyaw,target_time) instead of "
                         "the env's random sampling. Omit to keep the current random behavior.")
parser.add_argument("--play_episode", type=int, default=None,
                    help="With --commands_csv: play ONLY this row index, looping it.")
parser.add_argument("--ts_episode", type=int, default=None,
                    help="If set, ONLY save timeseries data for this specific episode index (prevents giant CSVs).")
parser.add_argument("--once", action="store_true", default=False,
                    help="Play the requested episode exactly once and exit.")

parser.add_argument("--results_csv", type=str, default="results_p2pt.csv", help="Per-episode results output.")
parser.add_argument("--append", action="store_true", default=False, help="Append to results_csv instead of overwriting.")

parser.add_argument("--boundary_push", action="store_true", default=False, help="Apply a severe lateral push 0.2s before target time.")

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
print(f"[DEBUG] hydra_args = {hydra_args}")
print(f"[DEBUG] full sys.argv = {sys.argv}")

if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# AppLauncher's --anim_recording_enabled path enables the omni.physx.pvd
# extension via Kit's own "--enable <ext>" convention, and leaves that
# mutated sys.argv behind — which then corrupts Hydra's parsing later.
# Restore the clean args here, after AppLauncher is done with them.
sys.argv = [sys.argv[0]] + hydra_args

# =========================================================
# PASTE THE IMPORTS HERE (After simulation_app boots!)
# =========================================================
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
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
# import go1_tasks.manager_based.locomotion.move_p2p
# import go1_tasks.manager_based.locomotion.velocity
# import go1_tasks.manager_based.locomotion.move_p2p_rel_time
import go1_tasks
import mc_tasks

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from generate_commands import load_commands_csv


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # ==========================================================
    # 🚨 UNLOCK RANDOM TARGET YAW 🚨
    # Force the generator to pick random angles (-180 to +180 deg)
    # ==========================================================
    import math
    if hasattr(env_cfg, "commands"):
        # Wrap in vars() because env_cfg.commands is a class object, not a dict!
        for cmd_name, cmd_cfg in vars(env_cfg.commands).items():
            if hasattr(cmd_cfg, "simple_heading"):
                cmd_cfg.simple_heading = False
            if hasattr(cmd_cfg, "ranges") and hasattr(cmd_cfg.ranges, "heading"):
                cmd_cfg.ranges.heading = (-math.pi, math.pi)
    # ==========================================================

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # env_cfg.viewer.origin_type = "asset_root"   # follow the robot
    env_cfg.viewer.origin_type = "world"   # fixed to world
    env_cfg.viewer.asset_name = "robot"
    # Look straight DOWN at the robot. With the orthographic camera set below,
    # the eye height doesn't affect zoom (ORTHO_SIZE controls that) -- height
    # only needs to be above the robot. Tiny y offset avoids the straight-down
    # gimbal ambiguity that can flip the view.
    env_cfg.viewer.eye = (0.0, 0.00, 20.0)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.0)

    # ---- EXTRA PROPRIOCEPTIVE SENSOR NOISE (before env build) ----
    if args_cli.sensor_noise.strip() and args_cli.sensor_k > 0.0:
        try:
            from isaaclab.utils.noise import GaussianNoiseCfg
            chans = {c.strip() for c in args_cli.sensor_noise.split(",") if c.strip()}
            if "all" in chans:
                chans = {"lin_vel", "ang_vel", "gravity"}
            pol = env_cfg.observations.policy
            k = args_cli.sensor_k
            applied = []
            if "lin_vel" in chans:
                pol.base_lin_vel.noise = GaussianNoiseCfg(mean=0.0, std=args_cli.sigma_lin_vel * k); applied.append("lin_vel")
            if "ang_vel" in chans:
                pol.base_ang_vel.noise = GaussianNoiseCfg(mean=0.0, std=args_cli.sigma_ang_vel * k); applied.append("ang_vel")
            if "gravity" in chans:
                pol.projected_gravity.noise = GaussianNoiseCfg(mean=0.0, std=args_cli.sigma_gravity * k); applied.append("gravity")
            # PLAY configs usually disable corruption; force it on so noise applies.
            pol.enable_corruption = True
            print(f"[SENSOR-NOISE] applied to {applied} (k={k}); enable_corruption forced True")
        except Exception as e:
            print(f"[SENSOR-NOISE][WARN] could not attach: {e}")
    else:
        print("[SENSOR-NOISE] none")


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

    print("CONTACT SENSOR BODY ORDER:", env.unwrapped.scene.sensors["contact_forces"].body_names)

    # ==========================================================
    # ORTHOGRAPHIC TOP-DOWN CAMERA
    # A perspective camera can't be both flat and zoomed-in. An ORTHOGRAPHIC
    # camera has zero perspective distortion and lets you set exactly how much
    # ground is visible (the "view limit") independent of distance -- so the
    # robot stays the same apparent size and looks flat like a map.
    #   ORTHO_SIZE = vertical extent of the view in METERS. Smaller = more zoomed
    #   in. ~2.0 frames a single Go1 tightly; ~4.0 shows more surrounding floor.
    # ==========================================================
    ORTHO_SIZE = 7.0  # meters of vertical world span shown (shrink to zoom in)
    try:
        import omni.usd
        from pxr import UsdGeom, Sdf, Gf

        stage = omni.usd.get_context().get_stage()
        # the default perspective viewport camera prim
        cam_path = "/OmniverseKit_Persp"
        cam_prim = stage.GetPrimAtPath(cam_path)
        if cam_prim and cam_prim.IsValid():
            cam = UsdGeom.Camera(cam_prim)
            cam.GetProjectionAttr().Set("orthographic")
            # In USD, ortho view width/height are set via aperture (in tenths of
            # scene units -> for meters, aperture = size * 10).
            cam.GetHorizontalApertureAttr().Set(ORTHO_SIZE * 10.0)
            cam.GetVerticalApertureAttr().Set(ORTHO_SIZE * 10.0)
            print(f"[CAMERA] Orthographic top-down set, view span = {ORTHO_SIZE} m")
        else:
            print(f"[CAMERA][WARN] could not find camera prim at {cam_path}; using perspective.")
    except Exception as e:
        print(f"[CAMERA][WARN] could not set orthographic camera: {e}")
    # ==========================================================

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

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
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

    # ==========================================================
    # POSE-ERROR NOISE MACHINERY + CLEAN-ERROR SCORING + CSV EXPORT
    # ==========================================================
    cmd_term = env.unwrapped.command_manager.get_term("base_pose")
    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device

    # ---- optional: fixed targets from a shared commands CSV (else env random) ----
    inject_commands = bool(args_cli.commands_csv)
    commands = None
    if inject_commands:
        commands = load_commands_csv(args_cli.commands_csv)
        print(f"[INJECT] loaded {len(commands)} commands from {args_cli.commands_csv} "
              f"(random sampling disabled)")

    def inject_target(ep_idx):
        """Override the env-sampled target with CSV row ep_idx. Composes the
        base-frame relative offset with the robot's CURRENT pose into the world
        pin, sets target_time, then _update_command() to sync the base error."""
        row = commands[ep_idx % len(commands)]
        cmd_x, cmd_y, cmd_yaw, target_time = row
        dev = cmd_term.device
        cur_pos = robot.data.root_pos_w[0:1]
        cur_quat = robot.data.root_quat_w[0:1]
        cur_yaw_q = yaw_quat(cur_quat)
        _, _, cur_yaw = euler_xyz_from_quat(cur_quat)
        offset = torch.tensor([[cmd_x, cmd_y, 0.0]], device=dev, dtype=cur_pos.dtype)
        gt_pos = cur_pos + quat_apply_yaw(cur_yaw_q, offset)
        gt_yaw = cur_yaw + torch.tensor([cmd_yaw], device=dev, dtype=cur_yaw.dtype)
        gt_yaw = torch.atan2(torch.sin(gt_yaw), torch.cos(gt_yaw))
        cmd_term._global_target_pose[0, 0:3] = gt_pos[0]
        cmd_term._global_target_pose[0, 3] = gt_yaw[0]
        cmd_term._command[0, 4] = target_time
        cmd_term._update_command()
        print(f"[INJECT] ep{ep_idx} row{ep_idx % len(commands)}: "
              f"dx={cmd_x:.2f} dy={cmd_y:.2f} dyaw={math.degrees(cmd_yaw):.0f}deg T={target_time:.2f}s")

    noise_rng = torch.Generator(device=device)
    noise_rng.manual_seed(int(args_cli.noise_seed))
    sig_xy = args_cli.sigma_xy * args_cli.noise_k
    sig_yaw = args_cli.sigma_yaw * args_cli.noise_k
    drift_xy_step = args_cli.drift_xy_per_s * args_cli.noise_k
    drift_yaw_step = args_cli.drift_yaw_per_s * args_cli.noise_k
    drift_bias = torch.zeros(3, device=device)  # [dx, dy, dyaw], reset each episode
    print(f"[NOISE] pose model={args_cli.noise_model} k={args_cli.noise_k}")

    def clean_error():
        """True base-frame error [dx,dy,dyaw] from the real target pin & pose,
        independent of any noise on the observation. Mirrors _update_command."""
        yaw_q = yaw_quat(robot.data.root_quat_w)
        pos_err_w = cmd_term._global_target_pose[:, :3] - robot.data.root_pos_w
        local_pos = quat_apply_inverse(yaw_q, pos_err_w)
        _, _, ryaw = euler_xyz_from_quat(yaw_q)
        yerr = cmd_term._global_target_pose[:, 3] - ryaw
        yerr = torch.atan2(torch.sin(yerr), torch.cos(yerr))
        return float(local_pos[0, 0]), float(local_pos[0, 1]), float(yerr[0])

    def apply_pose_noise(sim_dt):
        """Corrupt the pose_commands buffer the policy reads (indices 0,1,3)."""
        if args_cli.noise_model == "none" or args_cli.noise_k <= 0.0:
            return
        jx = torch.randn(1, generator=noise_rng, device=device).item() * sig_xy
        jy = torch.randn(1, generator=noise_rng, device=device).item() * sig_xy
        jyaw = torch.randn(1, generator=noise_rng, device=device).item() * sig_yaw
        if args_cli.noise_model == "drift":
            drift_bias[0] += torch.randn(1, generator=noise_rng, device=device).item() * drift_xy_step * math.sqrt(sim_dt)
            drift_bias[1] += torch.randn(1, generator=noise_rng, device=device).item() * drift_xy_step * math.sqrt(sim_dt)
            drift_bias[2] += torch.randn(1, generator=noise_rng, device=device).item() * drift_yaw_step * math.sqrt(sim_dt)
            bx, by, byaw = drift_bias[0].item(), drift_bias[1].item(), drift_bias[2].item()
        else:
            bx = by = byaw = 0.0
        cmd_term._command[0, 0] += (jx + bx)
        cmd_term._command[0, 1] += (jy + by)
        cmd_term._command[0, 3] += (jyaw + byaw)

    # CSV export of per-step time series
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

    # Also auto-save the per-episode commands (targets) alongside the export CSV
    cmd_csv_file = None
    cmd_csv_writer = None
    if args_cli.export_csv:
        cmd_csv_path = os.path.splitext(args_cli.export_csv)[0] + "_commands.csv"
        cmd_csv_file = open(cmd_csv_path, "w", newline="")
        cmd_csv_writer = csv.writer(cmd_csv_file)
        cmd_csv_writer.writerow(["cmd_x", "cmd_y", "cmd_yaw", "target_time"])
        print(f"[CSV] saving per-episode commands to {cmd_csv_path}")

    # ---------- SUMMARY CSV SETUP ----------
    write_header = True
    if args_cli.append and os.path.exists(args_cli.results_csv) and os.path.getsize(args_cli.results_csv) > 0:
        write_header = False
    results_file = open(args_cli.results_csv, "a" if args_cli.append else "w", newline="")
    results_writer = csv.writer(results_file)
    # if write_header:
    #     results_writer.writerow(["cmd_x", "cmd_y", "cmd_yaw", "target_time",
    #                              "err_x", "err_y", "err_yaw", "vel_x", "vel_y", "vel_yaw"])
    if write_header:
        results_writer.writerow(["cmd_x", "cmd_y", "cmd_yaw", "target_time",
                                 "err_x", "err_y", "err_yaw", "vel_x", "vel_y", "vel_yaw",
                                 "err_x_1s", "err_y_1s", "err_yaw_1s", "vel_x_1s", "vel_y_1s", "vel_yaw_1s"])



    # ---------- LATCH VARIABLES ----------
    captured_at_target = False
    tgt_err_x = tgt_err_y = tgt_err_yaw = 0.0
    tgt_vel_x = tgt_vel_y = tgt_vel_yaw = 0.0

    captured_at_1s = False
    err_x_1s = err_y_1s = err_yaw_1s = 0.0
    vel_x_1s = vel_y_1s = vel_yaw_1s = 0.0


    # reset environment
    obs = env.get_observations()
    timestep = 0

    # inject the starting episode's target if using a commands CSV
    if inject_commands:
        _start_ep = args_cli.play_episode if args_cli.play_episode is not None else 0
        with torch.inference_mode():
            inject_target(_start_ep)
            obs = env.get_observations()

    # ==========================================================
    # AUTO-EXIT: guarantees the loop terminates and reaches
    # env.close() cleanly, instead of relying on Ctrl-C (which
    # can cut off RecordVideo before it finalizes the clip).
    # ==========================================================
    if args_cli.max_steps is not None:
        max_steps = args_cli.max_steps
    elif args_cli.video:
        max_steps = args_cli.video_length + 50  # small buffer past the recorded clip
    else:
        max_steps = 100000
    print(f"[INFO]: Will auto-exit after {max_steps} steps.")

    print("[INFO]: Setup complete. Starting playback...")

    # ==========================================================
    # SETUP COORDINATE FRAME MARKER
    # ==========================================================
    frame_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/TargetFrame")
    # Scale it down so it is the same size as the robot, not massive
    frame_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    target_frame_marker = VisualizationMarkers(frame_cfg)

    # Body frame marker: a coordinate frame attached to the robot's base, updated
    # every step. Separate prim path so it renders independently of the target.
    body_frame_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/BodyFrame")
    body_frame_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
    body_frame_marker = VisualizationMarkers(body_frame_cfg)
    # ==========================================================

    # 1. ADD THIS FLAG HERE
    update_marker = True

    # ==========================================================
    # DATA TRACKING VARIABLES FOR PLOTTING
    # ==========================================================
    history_err_x = []
    history_err_y = []
    history_err_yaw = []
    history_height = []   # actual root height (cm) -- not a command error, there's no height target
    history_roll = []     # actual body roll (deg)
    history_pitch = []    # actual body pitch (deg)
    history_time = []

    dt = env.unwrapped.step_dt
    current_episode_time = 0.0
    episode_count = args_cli.play_episode if (bool(args_cli.commands_csv) and args_cli.play_episode is not None) else 0

    # ==========================================================
    # PLOT MODE: switch to an interactive backend only if requested.
    # Default stays "Agg" (headless-safe, file-only) unless the user
    # asks for "live" or "both". Falls back to save-only if no display
    # is available (e.g. running over SSH with no X forwarding).
    # ==========================================================
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


    applied_boundary_push = False
    # ==========================================================

    # Simulate environment
    while simulation_app.is_running() and timestep < max_steps:
        # run everything in inference mode
        with torch.inference_mode():
            # ---- NOISE the pose_commands buffer, refresh obs, so the policy acts
            # on NOISY feedback (the env recomputed the TRUE error last step). ----
            apply_pose_noise(dt)
            if args_cli.noise_model != "none" and args_cli.noise_k > 0.0:
                obs = env.get_observations()

            # 1. Compute actions from the policy (on possibly-noised obs)
            actions = policy(obs)

            # ==========================================================
            # 2. CAPTURE ERROR *BEFORE* STEP/RESET, from the CLEAN ground-truth
            # (true target pin vs true pose) -- NOT the noised buffer. This keeps
            # the plotted/recorded error honest even when the policy sees noise.
            # ==========================================================
            cx, cy, cyaw = clean_error()
            final_dist = math.hypot(cx, cy)
            final_yaw = cyaw
            target_time = float(cmd_term._command[0, 4].item())

            # TIMED BOUNDARY PERTURBATION TEST
            if args_cli.boundary_push and not applied_boundary_push and (current_episode_time >= target_time - 0.2):
                robot_asset = env.unwrapped.scene["robot"]
                
                # Clone current velocity and inject a +0.5 m/s lateral spike (equivalent to a massive sudden shove)
                root_vel = robot_asset.data.root_vel_w.clone()
                root_vel[:, 1] += 0.5  
                
                # Forcefully overwrite the physics engine state
                robot_asset.write_root_velocity_to_sim(root_vel)
                applied_boundary_push = True
                print(f"[DEBUG] Severe boundary push applied at t={current_episode_time:.2f}s!")


            # ---------- CAPTURE EXACTLY AT TARGET TIME ----------
            if (not captured_at_target) and (current_episode_time >= target_time):
                tgt_err_x = cx
                tgt_err_y = cy
                tgt_err_yaw = cyaw
                tgt_vel_x = robot_asset.data.root_lin_vel_b[0, 0].item()
                tgt_vel_y = robot_asset.data.root_lin_vel_b[0, 1].item()
                tgt_vel_yaw = robot_asset.data.root_ang_vel_b[0, 2].item()
                captured_at_target = True

            if (not captured_at_1s) and (current_episode_time >= target_time + 1.0):
                err_x_1s = cx
                err_y_1s = cy
                err_yaw_1s = cyaw
                vel_x_1s = robot_asset.data.root_lin_vel_b[0, 0].item()
                vel_y_1s = robot_asset.data.root_lin_vel_b[0, 1].item()
                vel_yaw_1s = robot_asset.data.root_ang_vel_b[0, 2].item()
                captured_at_1s = True



            # ==========================================================
            # CAPTURE ACTUAL BODY STATE (height, roll, pitch)
            # ==========================================================
            robot_asset = env.unwrapped.scene["robot"]
            current_height = robot_asset.data.root_pos_w[0, 2].item()
            current_roll, current_pitch, _ = euler_xyz_from_quat(robot_asset.data.root_quat_w)
            current_roll = current_roll[0].item()
            current_pitch = current_pitch[0].item()
            # ==========================================================

            # 3. Step the environment
            obs, _, dones, _ = env.step(actions)

            # ==========================================================
            # DRAW THE TARGET COORDINATE FRAME (Fixed Orientation)
            # ==========================================================
            if update_marker:
                command_term = env.unwrapped.command_manager.get_term("base_pose")
                cmd = command_term.command

                # 1. Directly extract the pure Yaw Command from index 3
                cmd_yaw_angle = cmd[:, 3]

                asset = env.unwrapped.scene["robot"]
                robot_pos = asset.data.root_pos_w.clone()
                robot_quat = asset.data.root_quat_w.clone()
                robot_yaw_q = yaw_quat(robot_quat)

                # 2. Calculate Absolute Target Position
                local_target_pos = torch.zeros_like(robot_pos)
                local_target_pos[:, 0] = cmd[:, 0]
                local_target_pos[:, 1] = cmd[:, 1]

                target_pos_w = robot_pos + quat_apply_yaw(robot_yaw_q, local_target_pos)
                target_pos_w[:, 2] = 0.45  # Lock to floor

                # 3. Calculate Absolute Target Orientation
                _, _, robot_yaw_angle = euler_xyz_from_quat(robot_yaw_q)
                absolute_target_yaw = robot_yaw_angle + cmd_yaw_angle

                # This perfectly aligns the marker's local X-axis (Red Arrow) with the Yaw command!
                target_quat_w = quat_from_euler_xyz(
                    torch.zeros_like(absolute_target_yaw),
                    torch.zeros_like(absolute_target_yaw),
                    absolute_target_yaw
                )

                print(f"[DEBUG MARKER INIT]")
                print(f"  -> Target Yaw: {absolute_target_yaw[0].item()/np.pi*180:.4f} deg")
                print(f"  -> target_quat_w: {[round(val, 4) for val in target_quat_w[0].tolist()]}\n")

                # save this episode's command (base-frame relative target) so it can
                # be replayed by the velocity script for a matched comparison.
                if cmd_csv_writer is not None:
                    _dx = cmd[0, 0].item()
                    _dy = cmd[0, 1].item()
                    _dyaw = cmd[0, 3].item()
                    _tt = cmd[0, 4].item()
                    cmd_csv_writer.writerow([f"{_dx:.6f}", f"{_dy:.6f}", f"{_dyaw:.6f}", f"{_tt:.6f}"])
                    cmd_csv_file.flush()

                target_frame_marker.visualize(target_pos_w, target_quat_w)
                update_marker = False
            # ==========================================================

            # ==========================================================
            # DRAW THE BODY COORDINATE FRAME (follows the robot, every step)
            # Red = body +X (forward), Green = +Y (left), Blue = +Z (up).
            # ==========================================================
            body_asset = env.unwrapped.scene["robot"]
            body_pos_w = body_asset.data.root_pos_w.clone()
            body_quat_w = body_asset.data.root_quat_w.clone()
            body_frame_marker.visualize(body_pos_w, body_quat_w)
            # ==========================================================

            # ==========================================================
            # RECORD DATA FOR THE PLOT (unaffected by the bug above —
            # this was already being appended every step, before reset)
            # ==========================================================
            ex_cm = cx * 100.0
            ey_cm = cy * 100.0
            eyaw_deg = cyaw * (180.0 / math.pi)
            h_cm = current_height * 100.0
            roll_deg = current_roll * (180.0 / math.pi)
            pitch_deg = current_pitch * (180.0 / math.pi)
            history_err_x.append(ex_cm)
            history_err_y.append(ey_cm)
            history_err_yaw.append(eyaw_deg)
            history_height.append(h_cm)
            history_roll.append(roll_deg)
            history_pitch.append(pitch_deg)

            history_time.append(current_episode_time)

            # ---------- CONDITIONAL PER-STEP CSV EXPORT ----------
            if csv_writer is not None:
                # Determine the correct episode label (handles --once edge case)
                ep_label = args_cli.play_episode if args_cli.play_episode is not None else episode_count
                
                if args_cli.ts_episode is None or ep_label == args_cli.ts_episode:
                    
                    # Extract global position
                    pos_x = robot_asset.data.root_pos_w[0, 0].item()
                    pos_y = robot_asset.data.root_pos_w[0, 1].item()
                    pos_z = robot_asset.data.root_pos_w[0, 2].item()
                    
                    # Extract local base velocities
                    vx = robot_asset.data.root_lin_vel_b[0, 0].item()
                    vy = robot_asset.data.root_lin_vel_b[0, 1].item()
                    vz = robot_asset.data.root_lin_vel_b[0, 2].item()
                    wx = robot_asset.data.root_ang_vel_b[0, 0].item()
                    wy = robot_asset.data.root_ang_vel_b[0, 1].item()
                    wz = robot_asset.data.root_ang_vel_b[0, 2].item()

                    # Extract absolute yaw for plotting
                    _, _, yaw_now_t = euler_xyz_from_quat(robot_asset.data.root_quat_w)
                    yaw_deg = yaw_now_t[0].item() * (180.0 / math.pi)

                    csv_writer.writerow([
                        ep_label, f"{current_episode_time:.4f}", f"{target_time:.4f}",
                        f"{ex_cm:.4f}", f"{ey_cm:.4f}", f"{eyaw_deg:.4f}",
                        f"{pos_x:.4f}", f"{pos_y:.4f}", f"{pos_z:.4f}",
                        f"{roll_deg:.4f}", f"{pitch_deg:.4f}", f"{yaw_deg:.4f}",
                        f"{vx:.4f}", f"{vy:.4f}", f"{vz:.4f}", f"{wx:.4f}", f"{wy:.4f}", f"{wz:.4f}"
                    ])             

                # csv_writer.writerow([
                #     episode_count, f"{current_episode_time:.4f}",
                #     f"{ex_cm:.4f}", f"{ey_cm:.4f}", f"{eyaw_deg:.4f}",
                #     f"{h_cm:.4f}", f"{roll_deg:.4f}", f"{pitch_deg:.4f}", f"{target_time:.4f}", 
                #     args_cli.noise_model, f"{args_cli.noise_k:.3f}",
                #     args_cli.sensor_noise if args_cli.sensor_noise.strip() else "none",
                #     f"{args_cli.sensor_k:.3f}",
                # ])

            current_episode_time += dt
            # ==========================================================

            # ==========================================================
            # PRINT THE ERROR AND SAVE PLOT IF THE EPISODE JUST ENDED
            # Uses final_dist / final_yaw / target_time captured ABOVE,
            # BEFORE env.step() reset them to the next episode's target.
            # ==========================================================
            if dones[0].item():
                print(f"[DEBUG] Episode Complete! Final Error -> Distance: {final_dist*100:.2f} cm | Yaw: {abs(final_yaw)/math.pi*180:.1f} deg")
                print(f"[DEBUG] Target Time was: {target_time:.2f}s | Episode lasted: {current_episode_time:.2f}s")
                # ---------- WRITE SUMMARY ROW ----------
                cmd_x, cmd_y, cmd_yaw, cmd_T = commands[episode_count]
                results_writer.writerow([
                    f"{cmd_x:.6f}", f"{cmd_y:.6f}", f"{cmd_yaw:.6f}", f"{cmd_T:.6f}",
                    f"{tgt_err_x:.6f}", f"{tgt_err_y:.6f}", f"{tgt_err_yaw:.6f}",
                    f"{tgt_vel_x:.6f}", f"{tgt_vel_y:.6f}", f"{tgt_vel_yaw:.6f}",
                    f"{err_x_1s:.6f}", f"{err_y_1s:.6f}", f"{err_yaw_1s:.6f}",
                    f"{vel_x_1s:.6f}", f"{vel_y_1s:.6f}", f"{vel_yaw_1s:.6f}"
                ])
                results_file.flush()
                captured_at_target = False  # Reset for the next run
                captured_at_1s = False

                applied_boundary_push = False
                

                if len(history_time) > 1:
                    ax1.clear()
                    ax2.clear()

                    # Top subplot: position-related, all in cm
                    ax1.plot(history_time, history_err_x, label="Error X (cm)", linewidth=2)
                    ax1.plot(history_time, history_err_y, label="Error Y (cm)", linewidth=2)
                    ax1.plot(history_time, history_height, label="Body Height (cm)", linewidth=2, color='green')
                    ax1.axvline(x=target_time, color='red', linestyle=':', linewidth=2, label=f"Target Time ({target_time:.1f}s)")
                    ax1.axhline(0, color='black', linestyle='-', linewidth=1)
                    ax1.set_title("Position Error and Body Height over Time", fontsize=14)
                    ax1.set_ylabel("cm", fontsize=12)
                    ax1.legend(loc="best")
                    ax1.grid(True, alpha=0.3)

                    # Bottom subplot: orientation-related, all in degrees.
                    # NOTE: only yaw has an actual command target -- roll and
                    # pitch have no target in this task (there's nothing to be
                    # "wrong" relative to), they're just the robot's real
                    # orientation, useful for spotting rocking/instability.
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

                    # Save to disk if requested (works headless)
                    if args_cli.plot_mode in ("save", "both"):
                        plot_path = os.path.join(log_dir, f"episode_plot_{episode_count:03d}.png")
                        fig.savefig(plot_path)
                        print(f"[DEBUG] Saved plot to: {plot_path}")

                    # Update the live window if requested (non-blocking)
                    if args_cli.plot_mode in ("live", "both"):
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                        plt.pause(0.001)

                episode_count += 1

                # Reset the tracking variables
                history_err_x.clear()
                history_err_y.clear()
                history_err_yaw.clear()
                history_height.clear()
                history_roll.clear()
                history_pitch.clear()
                history_time.clear()
                current_episode_time = 0.0

                # reset accumulating drift bias for the next episode
                drift_bias.zero_()
                if csv_file is not None:
                    csv_file.flush()

                # ---------- ADD THIS EXIT LOGIC ----------
                if args_cli.play_episode is not None and args_cli.once:
                    print(f"[INFO] --once: played episode {args_cli.play_episode} one time. Exiting.")
                    break
                # -----------------------------------------

                # CSV mode: override the freshly-resampled target with the next row
                if inject_commands:
                    next_ep = args_cli.play_episode if args_cli.play_episode is not None else episode_count
                    inject_target(next_ep)
                    obs = env.get_observations()

                update_marker = True

        timestep += 1

    print(f"[INFO]: Reached {timestep} steps — exiting.")

    if csv_file is not None:
        csv_file.close()
        print(f"[CSV] wrote {args_cli.export_csv}")
    if cmd_csv_file is not None:
        cmd_csv_path = os.path.splitext(args_cli.export_csv)[0] + "_commands.csv"
        cmd_csv_file.close()
        print(f"[CSV] wrote {cmd_csv_path}")

    results_file.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    print(f"[DEBUG] sys.argv right before main(): {sys.argv}")
    # run the main function
    main()
    # close sim app
    simulation_app.close()
