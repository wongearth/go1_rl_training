# scripts/reinforcement_learning/rsl_rl/play_goto.py

import os
import argparse
import math
import torch
from isaaclab.app import AppLauncher

# --- CONFIG ---
CHECKPOINT_PATH = "logs/rsl_rl/unitree_go1_flat/model_go1_vel.pt"
TASK_NAME = "Isaac-Velocity-Flat-Unitree-Go1-v0"
TARGET_POS = [3.0, 1.5] # Target X, Y
# --------------

parser = argparse.ArgumentParser(description="Go To Goal")
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

    # 1. Setup Env
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

    # Command Term
    cmd_name = "base_velocity"
    if cmd_name not in env.command_manager._terms:
        if "base_velocity_rough" in env.command_manager._terms:
             cmd_name = "base_velocity_rough"
        else:
             cmd_name = list(env.command_manager._terms.keys())[0]

    obs, _ = env.reset()

    print("\n" + "="*50)
    print(f" >>> GO TO GOAL: {TARGET_POS} <<<")
    print("="*50 + "\n")
    
    with torch.inference_mode():
        env.scene["robot"].write_root_pose_to_sim(torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]], device=device))
        
        while simulation_app.is_running():
            # 1. Get Robot State
            root_pos = env.scene["robot"].data.root_pos_w[0]
            current_x = root_pos[0].item()
            current_y = root_pos[1].item()
            
            # Quaternion to Yaw
            q = env.scene["robot"].data.root_quat_w[0] 
            w, x, y, z = q[0], q[1], q[2], q[3]
            yaw = torch.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z)).item()

            # 2. Calculate Error Vector
            err_x = TARGET_POS[0] - current_x
            err_y = TARGET_POS[1] - current_y
            distance = math.sqrt(err_x**2 + err_y**2)

            # 3. Controller Logic
            if distance < 0.2:
                # STOP if close enough
                v_x_cmd = 0.0
                v_y_cmd = 0.0
                w_z_cmd = 0.0
                print(f"GOAL REACHED! Dist: {distance:.2f}m     ", end="\r")
            else:
                # Calculate Heading to Target
                target_heading = math.atan2(err_y, err_x)
                
                # Heading Error (Steering)
                heading_err = target_heading - yaw
                # Normalize angle to [-pi, pi]
                heading_err = (heading_err + math.pi) % (2 * math.pi) - math.pi
                
                # Proportional Control
                v_x_cmd = 0.6  # Constant forward speed
                w_z_cmd = 2.0 * heading_err # Steer towards target
                v_y_cmd = 0.0 # No strafing needed
                
                # Safety: If facing wrong way, turn in place first
                if abs(heading_err) > 1.0: # > 57 degrees off
                    v_x_cmd = 0.0 

            # 4. Inject Command
            term = env.command_manager.get_term(cmd_name)
            term.time_left[:] = 10000.0 
            term.command[:, 0] = v_x_cmd
            term.command[:, 1] = v_y_cmd
            term.command[:, 2] = w_z_cmd

            # 5. Step
            actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)
            
            # Print
            print(f"Dist:{distance:.2f}m | YawErr:{heading_err:.2f} | Cmd:[{v_x_cmd:.1f}, {w_z_cmd:.1f}]", end="\r")

    simulation_app.close()

if __name__ == "__main__":
    main()