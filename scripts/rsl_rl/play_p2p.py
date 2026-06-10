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
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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
import go1_tasks.manager_based.locomotion.move_p2p
import go1_tasks.manager_based.locomotion.velocity



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


    print("[INFO]: Setup complete. Starting playback...")
        
    # ==========================================================
    # SETUP COORDINATE FRAME MARKER
    # ==========================================================
    frame_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/TargetFrame")
    # Scale it down so it is the same size as the robot, not massive
    frame_cfg.markers["frame"].scale = (0.2, 0.2, 0.2) 
    target_frame_marker = VisualizationMarkers(frame_cfg)
    # ==========================================================


    # Simulate environment
    target_frame_marker = VisualizationMarkers(frame_cfg)
    
    # 1. ADD THIS FLAG HERE
    update_marker = True

    # Simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # 1. Compute actions from the policy
            actions = policy(obs)

            # ==========================================================
            # 2. CAPTURE ERROR BEFORE RESET
            # Grab the command (which contains the remaining local error)
            # ==========================================================
            current_command = env.unwrapped.command_manager.get_command("base_pose")

            final_dist = torch.norm(current_command[0, :2]).item()
            final_yaw = current_command[0, 2].item()

            # 3. Step the environment
            obs, _, dones, _ = env.step(actions)
            
            # ==========================================================
            # DRAW THE TARGET COORDINATE FRAME (ONCE PER EPISODE)
            # ==========================================================
            # if update_marker:
            #     command_term = env.unwrapped.command_manager.get_term("base_pose")
            #     asset = env.unwrapped.scene["robot"]
                
            #     robot_pos = asset.data.root_pos_w
            #     robot_quat = asset.data.root_quat_w
            #     robot_yaw_q = yaw_quat(robot_quat)
                
            #     cmd = command_term.command 
                
            #     local_target_pos = torch.zeros_like(robot_pos)
            #     local_target_pos[:, 0] = cmd[:, 0]
            #     local_target_pos[:, 1] = cmd[:, 1]
                
            #     target_pos_w = robot_pos + quat_apply_yaw(robot_quat, local_target_pos)
            #     target_pos_w[:, 2] = 0.65  # Lock to floor
                
            #     cmd_yaw = cmd[:, 2]
            #     cmd_quat = quat_from_euler_xyz(torch.zeros_like(cmd_yaw), torch.zeros_like(cmd_yaw), cmd_yaw)
                
            #     target_quat_w = quat_mul(robot_yaw_q, cmd_quat)
                
            #     target_frame_marker.visualize(target_pos_w, target_quat_w)
                
            #     # TURN OFF THE SWITCH: The frame will now freeze in place
            #     update_marker = False
            # if update_marker:
            #     command_term = env.unwrapped.command_manager.get_term("base_pose")
            #     asset = env.unwrapped.scene["robot"]
                
            #     robot_pos = asset.data.root_pos_w
            #     robot_quat = asset.data.root_quat_w
            #     robot_yaw_q = yaw_quat(robot_quat)
                
            #     cmd = command_term.command 
                
            #     local_target_pos = torch.zeros_like(robot_pos)
            #     local_target_pos[:, 0] = cmd[:, 0]
            #     local_target_pos[:, 1] = cmd[:, 1]
                
            #     target_pos_w = robot_pos + quat_apply_yaw(robot_quat, local_target_pos)
            #     target_pos_w[:, 2] = 0.65  # Lock to floor
                
            #     cmd_yaw = cmd[:, 2]
            #     cmd_quat = quat_from_euler_xyz(torch.zeros_like(cmd_yaw), torch.zeros_like(cmd_yaw), cmd_yaw)
            #     target_quat_w = quat_mul(robot_yaw_q, cmd_quat)
                
            #     # ==========================================================
            #     # 🚨 DEBUG PRINT 🚨
            #     # ==========================================================
            #     cmd_type = type(command_term.cfg).__name__
            #     print(f"\n[DIAGNOSTIC] Command Generator is: {cmd_type}")
            #     if hasattr(command_term.cfg, "ranges") and hasattr(command_term.cfg.ranges, "heading"):
            #         print(f"[DIAGNOSTIC] Active Heading Range in Engine: {command_term.cfg.ranges.heading}")
            #     else:
            #         print("[DIAGNOSTIC] WARNING: This command generator DOES NOT SUPPORT HEADING!")
            #     # ==========================================================


            #     print("\n[DEBUG MARKER INIT]")
            #     print(f"  -> cmd_yaw:       {cmd_yaw[0].item():.4f} rad")
            #     # Using .tolist() to cleanly print the 4 quaternion components (usually w, x, y, z)
            #     print(f"  -> cmd_quat:      {[round(val, 4) for val in cmd_quat[0].tolist()]}")
            #     print(f"  -> target_quat_w: {[round(val, 4) for val in target_quat_w[0].tolist()]}\n")
            #     # ==========================================================

            #     target_frame_marker.visualize(target_pos_w, target_quat_w)
                
            #     # TURN OFF THE SWITCH: The frame will now freeze in place
            #     update_marker = False

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
            # 4. PRINT THE ERROR IF THE EPISODE JUST ENDED
            # ==========================================================
            if dones[0].item():
                print(f"[DEBUG] Episode Complete! Final Error -> Distance: {final_dist:.4f} m | Yaw: {final_yaw/np.pi*180:.1f} deg")

                # TURN THE SWITCH BACK ON FOR THE NEXT EPISODE
                update_marker = True

            # (Optional) LIVE TRACKING: 
            # If you want to see the distance shrink in real-time without spamming 
            # your terminal, uncomment the line below:
            # print(f"Live Distance: {final_dist:.3f} m      ", end="\r")
            # ==========================================================



    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
