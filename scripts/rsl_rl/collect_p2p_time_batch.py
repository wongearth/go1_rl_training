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
from isaaclab.utils.math import quat_apply_yaw, quat_from_euler_xyz, quat_mul, yaw_quat

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
# --- batch / CSV comparison args ---
parser.add_argument("--num_episodes", type=int, default=100, help="Number of episodes to run before exiting (batch mode).")
parser.add_argument("--commands_csv", type=str, default="p2p_commands.csv", help="Shared command file; created (seeded) if missing, else reused.")
parser.add_argument("--commands_seed", type=int, default=42, help="Seed used ONLY when creating the shared command file.")
parser.add_argument("--results_csv", type=str, default="results_p2p.csv", help="Per-episode results output for this (P2P) run.")
parser.add_argument("--inject_commands", action="store_true", default=False,
                    help="Override the env's internal target each episode with the shared CSV command (needed for a paired comparison).")
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

import csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_commands import ensure_commands


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

    env_cfg.viewer.origin_type = "asset_root"
    env_cfg.viewer.asset_name = "robot"
    # top-down view: camera directly above the robot, looking straight down
    env_cfg.viewer.eye = (0.0, 0.0, 4.0)
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

    print("CONTACT SENSOR BODY ORDER:", env.unwrapped.scene.sensors["contact_forces"].body_names)

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

    # reset environment
    obs = env.get_observations()
    timestep = 0

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
    episode_count = 0

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

    # ==========================================================
    # BATCH / PAIRED-COMPARISON SETUP
    # ==========================================================
    commands = ensure_commands(
        args_cli.commands_csv, args_cli.num_episodes, args_cli.commands_seed,
    )
    results_file = open(args_cli.results_csv, "w", newline="")
    results_writer = csv.writer(results_file)
    results_writer.writerow(["cmd_x", "cmd_y", "cmd_yaw", "target_time",
                             "err_x", "err_y", "err_yaw"])
    print(f"[INFO] Writing per-episode results to: {args_cli.results_csv}")

    cmd_term = env.unwrapped.command_manager.get_term("base_pose")

    # command term's configured ranges -- used to warn on out-of-distribution injects
    _r = cmd_term.cfg.ranges

    def inject_target(ep_idx):
        """Override the env-resampled target for THIS episode with CSV row ep_idx.

        The shared CSV stores base-frame relative (cmd_x=dx, cmd_y=dy, cmd_yaw=dyaw)
        plus target_time. This replicates LocalPoseTimeCommand._resample_command:
        it composes the relative offset with the robot's CURRENT world pose to form
        the world-absolute pin (_global_target_pose), writes the static target_time
        into _command[:,4], then calls _update_command() to sync the base-frame
        error the policy observes. dz is forced to 0 (the term's pos_z range is
        (0,0)).
        """
        if not args_cli.inject_commands:
            return  # injection disabled -> env uses its own internal targets
        cmd_x, cmd_y, cmd_yaw, target_time = commands[ep_idx]
        env0 = 0
        dev = cmd_term.device

        asset = env.unwrapped.scene["robot"]
        current_pos = asset.data.root_pos_w[env0:env0 + 1]          # (1,3)
        current_quat = asset.data.root_quat_w[env0:env0 + 1]        # (1,4)
        current_yaw_q = yaw_quat(current_quat)
        _, _, current_yaw_rad = euler_xyz_from_quat(current_quat)

        # compose relative offset -> world pin (same math as _resample_command)
        local_pos_offset = torch.tensor([[cmd_x, cmd_y, 0.0]], device=dev, dtype=current_pos.dtype)
        global_target_pos = current_pos + quat_apply_yaw(current_yaw_q, local_pos_offset)
        gt_yaw = current_yaw_rad + torch.tensor([cmd_yaw], device=dev, dtype=current_yaw_rad.dtype)
        global_target_yaw = torch.atan2(torch.sin(gt_yaw), torch.cos(gt_yaw))

        # write the hidden world pin + static target time, then re-sync base error
        cmd_term._global_target_pose[env0, 0:3] = global_target_pos[0]
        cmd_term._global_target_pose[env0, 3] = global_target_yaw[0]
        cmd_term._command[env0, 4] = target_time
        cmd_term._update_command()

        # out-of-distribution warning: injected target outside the term's train ranges
        oob = []
        if not (_r.pos_x[0] <= cmd_x <= _r.pos_x[1]): oob.append(f"dx={cmd_x:.2f} not in {_r.pos_x}")
        if not (_r.pos_y[0] <= cmd_y <= _r.pos_y[1]): oob.append(f"dy={cmd_y:.2f} not in {_r.pos_y}")
        if not (_r.time[0] <= target_time <= _r.time[1]): oob.append(f"T={target_time:.2f} not in {_r.time}")

        if ep_idx == 0:
            # verify the injection took: _command[:,:2] should now equal the
            # freshly-composed local offset (robot hasn't moved yet).
            synced = cmd_term._command[env0, 0:2].tolist()
            print(f"[INJECT] ep0 target rel=({cmd_x:.3f},{cmd_y:.3f},yaw={cmd_yaw:.3f}) T={target_time:.3f}")
            print(f"[INJECT] world pin=({global_target_pos[0,0]:.3f},{global_target_pos[0,1]:.3f},"
                  f"yaw={global_target_yaw[0]:.3f})")
            print(f"[INJECT] _command[:2] after sync = ({synced[0]:.3f},{synced[1]:.3f})  "
                  f"(should match rel dx,dy at reset)")
        if oob:
            print(f"[INJECT][OOD] ep{ep_idx}: " + "; ".join(oob)
                  + "  -> commanding outside trained range (expected if CSV ranges exceed the task's).")

    # ==========================================================

    # inject the episode-0 target before the first step
    inject_target(0)
    # refresh observations so the FIRST policy step sees the injected command
    # (the obs captured at setup were built from the pre-injection target).
    with torch.inference_mode():
        obs = env.get_observations()

    # at-target-time capture (P2P episode ends AT target time, so the last
    # pre-done error IS the at-target error)
    last_err_x = last_err_y = last_err_yaw = 0.0

    # Simulate environment
    while simulation_app.is_running() and timestep < max_steps:
        # run everything in inference mode
        with torch.inference_mode():
            # 1. Compute actions from the policy
            actions = policy(obs)

            # ==========================================================
            # 2. CAPTURE ERROR *BEFORE* STEP/RESET — this is the actual
            # final error of the episode that's about to end. Capturing
            # it AFTER env.step() would read the NEXT (freshly reset)
            # episode's random target instead, which is why the old
            # printed numbers looked huge/random despite the plotted
            # trajectory (built up during the episode, before any
            # reset) looking correct.
            # ==========================================================
            current_command = env.unwrapped.command_manager.get_command("base_pose")

            final_dist = torch.norm(current_command[0, :2]).item()
            # signed base-frame error components (what the CSV records)
            last_err_x = current_command[0, 0].item()
            last_err_y = current_command[0, 1].item()
            # FIXED: was current_command[0, 2] (that's dz, always 0 -- your
            # command's z range is fixed at (0.0, 0.0)). Yaw error is index 3,
            # matching what history_err_yaw already correctly used below.
            final_yaw = current_command[0, 3].item()
            last_err_yaw = final_yaw
            target_time = current_command[0, 4].item()

            # ==========================================================
            # CAPTURE ACTUAL BODY STATE (height, roll, pitch) -- these
            # aren't command errors like X/Y/yaw (there's no roll/pitch/height
            # target in this task), they're the robot's real orientation and
            # height, useful for seeing things like body sag or excessive roll.
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

                target_frame_marker.visualize(target_pos_w, target_quat_w)
                update_marker = False
            # ==========================================================

            # ==========================================================
            # RECORD DATA FOR THE PLOT (unaffected by the bug above —
            # this was already being appended every step, before reset)
            # ==========================================================
            history_err_x.append(current_command[0, 0].item() * 100.0)
            history_err_y.append(current_command[0, 1].item() * 100.0)
            history_err_yaw.append(current_command[0, 3].item() * (180.0 / math.pi))
            history_height.append(current_height * 100.0)
            history_roll.append(current_roll * (180.0 / math.pi))
            history_pitch.append(current_pitch * (180.0 / math.pi))

            history_time.append(current_episode_time)
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

                # ---- WRITE RESULT ROW: CSV command + at-target error ----
                cmd_x, cmd_y, cmd_yaw, cmd_T = commands[episode_count]
                results_writer.writerow([
                    f"{cmd_x:.6f}", f"{cmd_y:.6f}", f"{cmd_yaw:.6f}", f"{cmd_T:.6f}",
                    f"{last_err_x:.6f}", f"{last_err_y:.6f}", f"{last_err_yaw:.6f}",
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

                update_marker = True

                # done with all requested episodes -> exit
                if episode_count >= args_cli.num_episodes:
                    print(f"[INFO] Completed {episode_count} episodes. Results in {args_cli.results_csv}.")
                    break

                # inject the NEXT episode's shared target (env already resampled
                # during the env.step() reset that produced this `done`).
                inject_target(episode_count)
                # refresh obs so the next policy step sees the injected command
                obs = env.get_observations()

        timestep += 1

    results_file.close()
    print(f"[INFO]: Reached {timestep} steps — exiting.")

    # close the simulator
    env.close()


if __name__ == "__main__":
    print(f"[DEBUG] sys.argv right before main(): {sys.argv}")
    # run the main function
    main()
    # close sim app
    simulation_app.close()
