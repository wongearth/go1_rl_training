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
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless-safe; writes files without a display
import matplotlib.pyplot as plt

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
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

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

    term = env.unwrapped.command_manager.get_term("base_velocity")
    print(">>> heading_command =", term.cfg.heading_command)
    print(">>> resampling_time_range =", term.cfg.resampling_time_range)

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

    # --- direct handles to the sim + robot for ground-truth state ---
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    cmd_mgr = base_env.command_manager

    # plot every 5 seconds of sim time
    window_steps = int(round(5.0 / dt))
    plot_dir = os.path.join(log_dir, "play_plots")
    os.makedirs(plot_dir, exist_ok=True)
    print(f"[INFO] Saving 5s error plots to: {plot_dir}  ({window_steps} steps/window)")

    # rolling buffers (env 0 only)
    buf = {k: [] for k in ["t", "cmd_vx", "cmd_vy", "cmd_wz",
                           "vx", "vy", "wz", "height"]}
    window_idx = 0

    def flush_window(idx):
        """Save a plot + print RMS errors for the finished 5s window, then clear buffers."""
        if len(buf["t"]) == 0:
            return
        t = np.array(buf["t"])
        cmd_vx, cmd_vy, cmd_wz = map(np.array, (buf["cmd_vx"], buf["cmd_vy"], buf["cmd_wz"]))
        vx, vy, wz = map(np.array, (buf["vx"], buf["vy"], buf["wz"]))
        height = np.array(buf["height"])

        # RMS tracking errors over the window
        rms = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
        e_vx, e_vy, e_wz = rms(vx, cmd_vx), rms(vy, cmd_vy), rms(wz, cmd_wz)
        print(f"[window {idx}]  RMS err  vx={e_vx:.3f}  vy={e_vy:.3f}  "
              f"wz={e_wz:.3f}  |  height mean={height.mean():.3f} "
              f"min={height.min():.3f} max={height.max():.3f}")

        fig, ax = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
        ax[0].plot(t, cmd_vx, "--", label="cmd vx"); ax[0].plot(t, vx, label="vx")
        ax[0].set_ylabel("vx (m/s)"); ax[0].legend(loc="upper right")
        ax[0].set_title(f"Window {idx}  |  RMS: vx={e_vx:.3f} vy={e_vy:.3f} wz={e_wz:.3f}")
        ax[1].plot(t, cmd_vy, "--", label="cmd vy"); ax[1].plot(t, vy, label="vy")
        ax[1].set_ylabel("vy (m/s)"); ax[1].legend(loc="upper right")
        ax[2].plot(t, cmd_wz, "--", label="cmd wz"); ax[2].plot(t, wz, label="wz")
        ax[2].set_ylabel("wz (rad/s)"); ax[2].legend(loc="upper right")
        ax[3].plot(t, height, label="base height"); ax[3].set_ylabel("height (m)")
        ax[3].set_xlabel("time (s)"); ax[3].legend(loc="upper right")
        fig.tight_layout()
        out = os.path.join(plot_dir, f"window_{idx:03d}.png")
        fig.savefig(out, dpi=110); plt.close(fig)
        print(f"           saved {out}")

        for v in buf.values():
            v.clear()

    # reset environment
    obs = env.get_observations()
    step_in_window = 0
    sim_time = 0.0

    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        # --- record env 0 ground-truth state ---
        cmd = cmd_mgr.get_command("base_velocity")[0]          # (3,): vx, vy, wz
        lin_b = robot.data.root_lin_vel_b[0]                   # base-frame linear vel
        ang_b = robot.data.root_ang_vel_b[0]                   # base-frame angular vel
        h = robot.data.root_pos_w[0, 2]                        # world-z height
        if step_in_window % 50 == 0:
            print("root_z:", float(robot.data.root_pos_w[0, 2]),
                  "default_root_z:", float(robot.data.default_root_state[0, 2]))

        buf["t"].append(sim_time)
        buf["cmd_vx"].append(float(cmd[0])); buf["cmd_vy"].append(float(cmd[1])); buf["cmd_wz"].append(float(cmd[2]))
        buf["vx"].append(float(lin_b[0]));   buf["vy"].append(float(lin_b[1]));   buf["wz"].append(float(ang_b[2]))
        buf["height"].append(float(h))

        sim_time += dt
        step_in_window += 1

        # --- every 5 s: plot, then restart robot at origin ---
        if step_in_window >= window_steps:
            flush_window(window_idx)
            window_idx += 1
            step_in_window = 0
            sim_time = 0.0
            # reset env 0 back to origin / default pose
            with torch.inference_mode():
                obs, _ = env.reset()

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
