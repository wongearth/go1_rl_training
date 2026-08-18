# scripts/reinforcement_learning/rsl_rl/play_yaw_spot.py

import os
import argparse
import math
import torch
from isaaclab.app import AppLauncher

# --- CONFIG ---
CHECKPOINT_PATH = "logs/rsl_rl/spot_flat/model_spot.pt" 
TASK_NAME = "Isaac-Velocity-Flat-Spot-v0"
TARGET_YAW_VEL_DEG = -90.0  
DURATION_SEC = 10.0        
# --------------

parser = argparse.ArgumentParser(description="Yaw Rotation Test")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = False 
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- IMPORTS ---
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.terrains import TerrainImporterCfg

def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: File not found at {CHECKPOINT_PATH}")
        simulation_app.close()
        return

    # 1. Setup Environment
    print(f"[INFO] Loading Task: {TASK_NAME}")
    env_cfg = parse_env_cfg(TASK_NAME)
    
    # Force Plane Terrain
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=None,
        visual_material=None,
        debug_vis=False,
    )
    
    # --- CRITICAL FIX: DISABLE CURRICULUM ---
    # The curriculum expects complex terrain properties (generator.size).
    # Since we switched to a simple plane, we must turn off the terrain curriculum
    # to prevent the "NoneType object has no attribute size" crash.
    env_cfg.curriculum.terrain_levels = None
    # ----------------------------------------
    
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.5) 
    env_cfg.scene.num_envs = 1
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device
    dt = env.step_dt

    # 2. Load JIT Policy
    print(f"[INFO] Loading JIT Policy from: {CHECKPOINT_PATH}")
    try:
        policy = torch.jit.load(CHECKPOINT_PATH, map_location=device)
        policy.eval()
        print("[INFO] Policy loaded successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to load JIT policy: {e}")
        simulation_app.close()
        return

    # 3. Command Setup
    cmd_name = "base_velocity"
    if cmd_name not in env.command_manager._terms:
        if "base_velocity_rough" in env.command_manager._terms:
             cmd_name = "base_velocity_rough"
        else:
             cmd_name = list(env.command_manager._terms.keys())[0]
    
    print(f"[INFO] Using Command Term: {cmd_name}")

    obs, _ = env.reset()

    # Calculate Target
    target_omega_rad = TARGET_YAW_VEL_DEG * (math.pi / 180.0)
    
    print("\n" + "="*60)
    print(f" >>> YAW TEST: SPINNING AT {TARGET_YAW_VEL_DEG} deg/s ({target_omega_rad:.2f} rad/s) <<<")
    print("="*60 + "\n")
    print(f"{'Time (s)':<10} | {'Yaw Angle (deg)':<15} | {'Target Vel':<12} | {'Actual Vel':<12} | {'Error'}")
    print("-" * 70)
    
    current_time = 0.0
    
    with torch.inference_mode():
        # Reset Root State
        root_state = env.scene["robot"].data.default_root_state.clone()
        root_state[:, :3] = torch.tensor([0.0, 0.0, 0.5], device=device)
        env.scene["robot"].write_root_pose_to_sim(root_state[:, :7])
        env.scene["robot"].write_root_velocity_to_sim(root_state[:, 7:])
        
        while simulation_app.is_running() and current_time < DURATION_SEC:
            # 1. Inject Command
            term = env.command_manager.get_term(cmd_name)
            term.time_left[:] = 10000.0 
            term.command[:, 0] = 0.0  # Vx
            term.command[:, 1] = 0.0  # Vy
            term.command[:, 2] = target_omega_rad # Omega Z

            # 2. Step Policy
            if isinstance(obs, dict):
                input_obs = obs["policy"] 
            else:
                input_obs = obs

            actions = policy(input_obs)
            
            # 3. Step Env
            obs, _, _, _, _ = env.step(actions)
            
            # 4. Measure Data
            q = env.scene["robot"].data.root_quat_w[0] 
            w, x, y, z = q[0].item(), q[1].item(), q[2].item(), q[3].item()
            yaw_rad = math.atan2(2.0*(w*z + x*y), 1.0 - 2.0*(y*y + z*z))
            yaw_deg = math.degrees(yaw_rad)
            
            ang_vel = env.scene["robot"].data.root_ang_vel_w[0]
            actual_omega_rad = ang_vel[2].item()
            actual_omega_deg = math.degrees(actual_omega_rad)
            
            # 5. Print
            if current_time % 0.2 < dt:
                error = abs(TARGET_YAW_VEL_DEG - actual_omega_deg)
                print(f"{current_time:<10.2f} | {yaw_deg:<15.2f} | {TARGET_YAW_VEL_DEG:<12.1f} | {actual_omega_deg:<12.1f} | {error:.1f}")

            current_time += dt

    print("-" * 70)
    print("TEST COMPLETE")
    simulation_app.close()

if __name__ == "__main__":
    main()