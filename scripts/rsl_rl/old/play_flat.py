# scripts/reinforcement_learning/rsl_rl/play_flat.py

import os
import argparse
import torch
from isaaclab.app import AppLauncher

# --- USER CONFIG ---
CHECKPOINT_PATH = "logs/rsl_rl/unitree_go1_flat/model_go1_vel.pt"
TASK_NAME = "Isaac-Velocity-Flat-Unitree-Go1-v0"
# -------------------

parser = argparse.ArgumentParser(description="Flat Floor Test")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = False 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner

# Import Terrain & Robot Configs
from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.rough_env_cfg import UnitreeGo1RoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.agents.rsl_rl_ppo_cfg import UnitreeGo1RoughPPORunnerCfg

# Import the Plane Config class
from isaaclab.terrains import TerrainImporterCfg
import go1_tasks.manager_based.locomotion.move_p2p
import go1_tasks.manager_based.locomotion.velocity



# --- WRAPPER ---
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
# ----------------

def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        simulation_app.close()
        return

    print(f"[INFO] Loading Environment: {TASK_NAME}")
    env_cfg = parse_env_cfg(TASK_NAME)
    
    # --- FIX 1: THE FLOOR ---
    # Instead of a generator, we create a simple TerrainImporterCfg
    # This creates a basic physics plane (infinite floor).
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=None,
        visual_material=None,
        debug_vis=False,
    )
    
    # --- FIX 2: THE SPAWN ---
    # Force robot to spawn exactly at (0,0) raised by 0.5m
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.5) 
    env_cfg.scene.num_envs = 1
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env_wrapped = SimpleRslRlWrapper(env)
    device = env.device

    print("[INFO] Loading Policy...")
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

    obs, _ = env.reset()

    print("\n" + "="*50)
    print(" >>> ROBOT IS AT (0,0) ON AN INFINITE FLOOR <<<")
    print("="*50 + "\n")
    
    with torch.inference_mode():
        # Double-check reset position
        env.scene["robot"].write_root_pose_to_sim(torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]], device=device))
        
        while simulation_app.is_running():
            # Walk Forward 0.5 m/s
            term = env.command_manager.get_term(cmd_name)
            term.time_left[:] = 10000.0 
            term.command[:, 0] = 1.0 
            term.command[:, 1] = 0.0
            term.command[:, 2] = 0.0

            actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)
            
            # Print coords
            pos = env.scene["robot"].data.root_pos_w[0]
            print(f"X: {pos[0]:.2f} | Y: {pos[1]:.2f} | Z: {pos[2]:.2f}", end="\r")

    simulation_app.close()

if __name__ == "__main__":
    main()