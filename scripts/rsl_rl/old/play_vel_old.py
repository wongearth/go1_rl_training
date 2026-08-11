# scripts/reinforcement_learning/rsl_rl/play_forward.py

import os
import argparse
import torch
from isaaclab.app import AppLauncher

# --- USER CONFIG ---
CHECKPOINT_PATH = "logs/rsl_rl/unitree_go1_flat/model_go1_vel.pt"
TASK_NAME = "Isaac-Velocity-Flat-Unitree-Go1-v0"
# -------------------

# 1. Launch App
parser = argparse.ArgumentParser(description="Forward Walk")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = False 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 2. Imports
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

# Configs
from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.rough_env_cfg import UnitreeGo1RoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.agents.rsl_rl_ppo_cfg import UnitreeGo1RoughPPORunnerCfg

# --- THE WRAPPER (Standard RSL-RL Interface) ---
class SimpleRslRlWrapper:
    def __init__(self, env):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        # Load dims from manager
        self.num_obs = env.observation_manager.group_obs_dim["policy"][0]
        self.num_actions = env.action_manager.total_action_dim
        if "critic" in env.observation_manager.group_obs_dim:
            self.num_privileged_obs = env.observation_manager.group_obs_dim["critic"][0]
        else:
            self.num_privileged_obs = None

    def get_observations(self):
        # Return FULL DICT (Crucial for RSL-RL 2.x)
        return self.env.observation_manager.compute()

    def reset(self):
        obs, _ = self.env.reset()
        return obs, None

    def step(self, actions):
        obs, rew, terminated, truncated, extras = self.env.step(actions)
        return obs, rew, terminated, extras

    def __getattr__(self, name):
        return getattr(self.env, name)
# -----------------------------------------------

def main():
    # 3. Setup Environment
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        simulation_app.close()
        return

    print(f"[INFO] Loading Environment: {TASK_NAME}")
    env_cfg = parse_env_cfg(TASK_NAME)
    
    # Fix Visuals
    rough_cfg = UnitreeGo1RoughEnvCfg()
    env_cfg.scene.terrain = rough_cfg.scene.terrain
    env_cfg.scene.terrain.terrain_generator.curriculum = False
    env_cfg.scene.terrain.terrain_generator.difficulty = 0.0
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.40) 
    
    env_cfg.scene.num_envs = 1
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    # Wrap it
    env_wrapped = SimpleRslRlWrapper(env)
    device = env.device

    # 4. Load Policy
    print("[INFO] Loading Policy Configuration...")
    agent_cfg_obj = UnitreeGo1RoughPPORunnerCfg()
    agent_cfg = agent_cfg_obj.to_dict()
    agent_cfg["device"] = device
    agent_cfg["seed"] = 42
    agent_cfg["policy"]["actor_hidden_dims"] = [128, 128, 128]
    agent_cfg["policy"]["critic_hidden_dims"] = [128, 128, 128]
    
    runner = OnPolicyRunner(env_wrapped, agent_cfg, log_dir=None, device=device)
    runner.load(CHECKPOINT_PATH)
    policy = runner.get_inference_policy(device=device)

    # Detect Command Name
    cmd_manager = env.command_manager
    cmd_name = "base_velocity"
    if cmd_name not in cmd_manager._terms:
        if "base_velocity_rough" in cmd_manager._terms:
             cmd_name = "base_velocity_rough"
        else:
             cmd_name = list(cmd_manager._terms.keys())[0]
    
    print(f"[INFO] Controlling command term: '{cmd_name}'")

    # Initial Reset
    obs, _ = env.reset() # Wrapper returns full dict

    # 6. Run Loop
# ... inside main() ...

    print(">>> ROBOT SHOULD MOVE FORWARD AT 0.5 m/s")
    print(">>> CHECK THE TERMINAL FOR COORDINATES!")
    
    with torch.inference_mode():
        while simulation_app.is_running():
            # --- HARDCODED COMMAND ---
            vel_x = 0.5
            vel_y = 0.0
            ang_z = 0.0
            
            # Inject Command
            term = env.command_manager.get_term(cmd_name)
            term.time_left[:] = 10000.0 
            term.command[:, 0] = vel_x
            term.command[:, 1] = vel_y
            term.command[:, 2] = ang_z

            # Run Robot
            actions = policy(obs)
            
            # Step Env
            obs, _, _, _, _ = env.step(actions) 
            
            # --- DIAGNOSTIC PRINT ---
            # Get robot position from the simulation
            pos = env.scene["robot"].data.root_pos_w[0]
            
            # Print X, Y, Z coordinates
            print(f"Pos: [X={pos[0]:.2f}, Y={pos[1]:.2f}, Z={pos[2]:.2f}]", end="\r")
            
            # Check for "Death"
            if pos[2] < 0.1:
                print(f"\n[ALERT] Robot fell through the floor! Z={pos[2]:.2f}")
                break
            if torch.isnan(pos[0]):
                print(f"\n[ALERT] Robot exploded! Coordinates are NaN.")
                break
            # ------------------------

    simulation_app.close()

if __name__ == "__main__":
    main()