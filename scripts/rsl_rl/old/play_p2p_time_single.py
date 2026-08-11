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
    # env_cfg.viewer.origin_type = "world"
    env_cfg.viewer.asset_name = "robot"
    env_cfg.viewer.eye = (1.0, 0.0, -0.05)      # close, and near LEG height (not overhead)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.05)   # look roughly at leg-height, not the body center


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
    history_time = []

    dt = env.unwrapped.step_dt
    current_episode_time = 0.0
    episode_count = 0

    fig, ax = plt.subplots(figsize=(10, 6))

    # ==========================================================

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
            final_yaw = current_command[0, 2].item()
            target_time = current_command[0, 4].item()

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

                if len(history_time) > 1:
                    ax.clear()

                    ax.plot(history_time, history_err_x, label="Error X (cm)", linewidth=2)
                    ax.plot(history_time, history_err_y, label="Error Y (cm)", linewidth=2)
                    ax.plot(history_time, history_err_yaw, label="Error Yaw (deg)", linewidth=2, linestyle='--')

                    ax.axvline(x=target_time, color='red', linestyle=':', linewidth=2, label=f"Target Time ({target_time:.1f}s)")
                    ax.axhline(0, color='black', linestyle='-', linewidth=1)

                    ax.set_title("Relative Position and Yaw Error over Time", fontsize=14)
                    ax.set_xlabel("Time Elapsed (Seconds)", fontsize=12)
                    ax.set_ylabel("Error (cm / degrees)", fontsize=12)
                    ax.legend(loc="best")
                    ax.grid(True, alpha=0.3)

                    # Save to disk instead of showing/pausing an interactive
                    # window — works headless, and doesn't block/hang.
                    plot_path = os.path.join(log_dir, f"episode_plot_{episode_count:03d}.png")
                    fig.savefig(plot_path)
                    print(f"[DEBUG] Saved plot to: {plot_path}")

                episode_count += 1

                # Reset the tracking variables
                history_err_x.clear()
                history_err_y.clear()
                history_err_yaw.clear()
                history_time.clear()
                current_episode_time = 0.0

                update_marker = True

        timestep += 1

    print(f"[INFO]: Reached {timestep} steps — exiting.")

    # close the simulator
    env.close()


if __name__ == "__main__":
    print(f"[DEBUG] sys.argv right before main(): {sys.argv}")
    # run the main function
    main()
    # close sim app
    simulation_app.close()
