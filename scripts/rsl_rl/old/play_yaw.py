# scripts/reinforcement_learning/rsl_rl/play_yaw_test.py

import os
import argparse
import math
import torch
from isaaclab.app import AppLauncher

# --- CONFIG ---
CHECKPOINT_PATH = "logs/rsl_rl/unitree_go1_flat/model_go1_vel.pt"
TASK_NAME = "Isaac-Velocity-Flat-Unitree-Go1-v0"
TARGET_YAW_VEL_DEG = -90.0  # Target rotation speed (deg/s)
DURATION_SEC = 10.0        # How long to spin
# --------------

parser = argparse.ArgumentParser(description="Yaw Rotation Test")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = False 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.agents.rsl_rl_ppo_cfg import UnitreeGo1RoughPPORunnerCfg
from isaaclab.terrains import TerrainImporterCfg

class SimpleRslRlWrapper:
    def __init__(self, env):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.num_obs = env.observation_manager.group_obs_dim["policy"][0]
        self.num_actions = env.action_manager.total_action_dim
        if "critic" in env.observation_manager.group_obs_dim:
            self.num_privileged_obs = env.observation_manager.group_obs_dim["critic"][0]
        else:
            self.num_privileged_obs = None

    def get_observations(self):
        return self.env.observation_manager.compute()

    def reset(self):
        obs, _ = self.env.reset()
        return obs, None

    def step(self, actions):
        obs, rew, terminated, truncated, extras = self.env.step(actions)
        return obs, rew, terminated, extras

    def __getattr__(self, name):
        return getattr(self.env, name)

def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        simulation_app.close()
        return

    # 1. Setup Environment (Flat Floor)
    env_cfg = parse_env_cfg(TASK_NAME)
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=None,
        visual_material=None,
        debug_vis=False,
    )
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.5) 
    env_cfg.scene.num_envs = 1
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env_wrapped = SimpleRslRlWrapper(env)
    device = env.device
    dt = env.step_dt

    # 2. Load Policy
    agent_cfg_obj = UnitreeGo1RoughPPORunnerCfg()
    agent_cfg = agent_cfg_obj.to_dict()
    agent_cfg["device"] = device
    agent_cfg["seed"] = 42
    agent_cfg["policy"]["actor_hidden_dims"] = [128, 128, 128]
    agent_cfg["policy"]["critic_hidden_dims"] = [128, 128, 128]
    
    runner = OnPolicyRunner(env_wrapped, agent_cfg, log_dir=None, device=device)
    runner.load(CHECKPOINT_PATH)
    policy = runner.get_inference_policy(device=device)

    # Command Term Name
    cmd_name = "base_velocity"
    if cmd_name not in env.command_manager._terms:
        if "base_velocity_rough" in env.command_manager._terms:
             cmd_name = "base_velocity_rough"
        else:
             cmd_name = list(env.command_manager._terms.keys())[0]

    obs, _ = env.reset()

    # Calculate Target in Radians
    target_omega_rad = TARGET_YAW_VEL_DEG * (math.pi / 180.0)
    
    print("\n" + "="*60)
    print(f" >>> YAW TEST: SPINNING AT {TARGET_YAW_VEL_DEG} deg/s ({target_omega_rad:.2f} rad/s) <<<")
    print("="*60 + "\n")
    print(f"{'Time (s)':<10} | {'Yaw Angle (deg)':<15} | {'Target Vel':<12} | {'Actual Vel':<12} | {'Error'}")
    print("-" * 70)
    
    current_time = 0.0
    
    with torch.inference_mode():
        # Reset to (0,0)
        env.scene["robot"].write_root_pose_to_sim(torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]], device=device))
        
        while simulation_app.is_running() and current_time < DURATION_SEC:
            # 1. Inject Yaw Command
            term = env.command_manager.get_term(cmd_name)
            term.time_left[:] = 10000.0 
            term.command[:, 0] = 0.0  # Vx
            term.command[:, 1] = 0.0  # Vy
            term.command[:, 2] = target_omega_rad # Omega Z

            # 2. Step Env
            actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)
            
            # 3. Measure Data
            # Get Yaw Angle (Position)
            q = env.scene["robot"].data.root_quat_w[0] 
            w, x, y, z = q[0].item(), q[1].item(), q[2].item(), q[3].item()
            yaw_rad = math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
            yaw_deg = math.degrees(yaw_rad)
            
            # Get Yaw Velocity (Speed)
            ang_vel = env.scene["robot"].data.root_ang_vel_w[0]
            actual_omega_rad = ang_vel[2].item()
            actual_omega_deg = math.degrees(actual_omega_rad)
            
            # 4. Print Report (Every 0.2s to be readable)
            if current_time % 0.2 < dt:
                error = abs(TARGET_YAW_VEL_DEG - actual_omega_deg)
                print(f"{current_time:<10.2f} | {yaw_deg:<15.2f} | {TARGET_YAW_VEL_DEG:<12.1f} | {actual_omega_deg:<12.1f} | {error:.1f}")

            current_time += dt

    print("-" * 70)
    print("TEST COMPLETE")
    simulation_app.close()

if __name__ == "__main__":
    main()