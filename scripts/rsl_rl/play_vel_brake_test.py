# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

from isaaclab.utils.math import euler_xyz_from_quat
import torch

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

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import math
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

import mc_tasks


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0
    
    # --- HARD BRAKE TEST VARIABLES ---
    yaw_at_brake = None
    braked = False
    print("\n[INFO] Starting Hard Brake Test: Sprinting at 1.0 m/s...")

    while simulation_app.is_running():
        start_time = time.time()
        current_time = timestep * dt

        with torch.inference_mode():
            
            # 1. HACK THE COMMAND MANAGER: Force the Sprint and Brake
            try:
                cmd_term = env.unwrapped.command_manager.get_term("base_velocity")
                if current_time < 1.5:
                    cmd_term.command[:, 0] = 1.0  # Sprint forward
                    cmd_term.command[:, 1] = 0.0
                    cmd_term.command[:, 2] = 0.0  # Zero commanded yaw
                else:
                    cmd_term.command[:, 0] = 0.0  # Hard Brake
                    cmd_term.command[:, 1] = 0.0
                    cmd_term.command[:, 2] = 0.0
            except KeyError:
                pass # Fails safely if the command term is named something else

            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        # 2. MEASURE THE PHYSICAL DRIFT
        asset = env.unwrapped.scene["robot"]
        _, _, yaw_t = euler_xyz_from_quat(asset.data.root_quat_w)
        current_yaw = yaw_t[0].item()

        # 3. DEBUG PRINT LOGIC
        if current_time >= 1.5 and not braked:
            yaw_at_brake = current_yaw
            braked = True
            print(f"\n[DEBUG] t={current_time:.2f}s | Hard Brake Commanded!")
            print(f"[DEBUG] -> Heading right before brake: {yaw_at_brake * 180.0 / math.pi:.2f} deg")

        if current_time >= 3.0:
            final_yaw = current_yaw
            yaw_error = abs(final_yaw - yaw_at_brake) * 180.0 / math.pi
            print(f"\n[DEBUG] t={current_time:.2f}s | Robot fully settled.")
            print(f"[DEBUG] -> Final Heading: {final_yaw * 180.0 / math.pi:.2f} deg")
            print(f"[DEBUG] -> PERMANENT YAW ERROR FROM SKIDDING: {yaw_error:.2f} deg\n")
            print("[INFO] Test complete. Exiting.")
            break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)
            
        timestep += 1

    env.close()

if __name__ == "__main__":
    main()
    simulation_app.close()