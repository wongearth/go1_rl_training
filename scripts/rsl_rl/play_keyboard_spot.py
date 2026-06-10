# scripts/reinforcement_learning/rsl_rl/play_keyboard_spot.py

import os
import argparse
import math
import torch
from isaaclab.app import AppLauncher

# --- CONFIG ---
CHECKPOINT_PATH = "logs/rsl_rl/spot_flat/model_spot.pt"
TASK_NAME = "Isaac-Velocity-Flat-Spot-v0"
# --------------

parser = argparse.ArgumentParser(description="Spot Keyboard Control")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = False  # MUST be False to capture keyboard input
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ==============================================================================
# IMPORTS MUST BE HERE (AFTER simulation_app starts)
# ==============================================================================
import carb 
import omni.appwindow
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.terrains import TerrainImporterCfg

class KeyboardController:
    """Simple keyboard controller for velocity commands."""
    
    def __init__(self):
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
        
        # Spot Parameters (Stable speeds)
        self.linear_speed_increment = 0.05
        self.angular_speed_increment = 0.1
        self.max_linear_speed = 1.0
        self.max_angular_speed = 1.5
        
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        
    def update(self):
        """Update velocities based on keyboard input."""
        # Check keys
        W = self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.W)
        S = self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.S)
        A = self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.A)
        D = self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.D)
        Q = self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.Q)
        E = self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.E)
        SPACE = self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.SPACE)
        
        # Linear X (Surge)
        if W:
            self.vx = min(self.vx + self.linear_speed_increment, self.max_linear_speed)
        elif S:
            self.vx = max(self.vx - self.linear_speed_increment, -self.max_linear_speed)
        else:
            self.vx *= 0.9 # Decay
            if abs(self.vx) < 0.01: self.vx = 0.0
        
        # Linear Y (Sway)
        if A:
            self.vy = min(self.vy + self.linear_speed_increment, self.max_linear_speed)
        elif D:
            self.vy = max(self.vy - self.linear_speed_increment, -self.max_linear_speed)
        else:
            self.vy *= 0.9 # Decay
            if abs(self.vy) < 0.01: self.vy = 0.0
        
        # Angular Z (Yaw)
        if Q:
            self.omega = min(self.omega + self.angular_speed_increment, self.max_angular_speed)
        elif E:
            self.omega = max(self.omega - self.angular_speed_increment, -self.max_angular_speed)
        else:
            self.omega *= 0.9 # Decay
            if abs(self.omega) < 0.01: self.omega = 0.0
        
        # E-Stop
        if SPACE:
            self.vx = 0.0
            self.vy = 0.0
            self.omega = 0.0
        
        return self.vx, self.vy, self.omega

def get_robot_pose(env):
    root_pos = env.scene["robot"].data.root_pos_w[0]
    q = env.scene["robot"].data.root_quat_w[0]
    w, qx, qy, qz = q[0].item(), q[1].item(), q[2].item(), q[3].item()
    yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
    return [root_pos[0].item(), root_pos[1].item(), yaw]

def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        simulation_app.close()
        return

    # 1. Setup Environment
    env_cfg = parse_env_cfg(TASK_NAME)
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=None,
        visual_material=None,
        debug_vis=False,
    )
    
    # CRITICAL FIXES FOR SPOT
    env_cfg.curriculum.terrain_levels = None # Prevent crash on flat ground
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.75) # Higher spawn for Spot
    env_cfg.scene.num_envs = 1
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device

    # 2. Load JIT Policy (Direct Load)
    print(f"[INFO] Loading JIT Policy: {CHECKPOINT_PATH}")
    try:
        policy = torch.jit.load(CHECKPOINT_PATH, map_location=device)
        policy.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load JIT model: {e}")
        simulation_app.close()
        return

    # Find command term
    cmd_name = "base_velocity"
    if cmd_name not in env.command_manager._terms:
        if "base_velocity_rough" in env.command_manager._terms:
             cmd_name = "base_velocity_rough"
        else:
             cmd_name = list(env.command_manager._terms.keys())[0]

    obs, _ = env.reset()
    controller = KeyboardController()

    print("\n" + "="*70)
    print(" "*20 + "SPOT KEYBOARD CONTROL")
    print("="*70)
    print("  W/S   : Forward / Backward")
    print("  A/D   : Left / Right (Strafe)")
    print("  Q/E   : Turn Left / Right")
    print("  SPACE : STOP")
    print("="*70 + "\n")
    
    with torch.inference_mode():
        # Reset robot
        env.scene["robot"].write_root_pose_to_sim(
            torch.tensor([[0.0, 0.0, 0.75, 0.0, 0.0, 0.0, 1.0]], device=device)
        )
        obs, _ = env.reset()
        
        frame_count = 0
        
        while simulation_app.is_running():
            # 1. Update Command from Keyboard
            vx_cmd, vy_cmd, omega_cmd = controller.update()
            
            # 2. Apply Command to Env
            term = env.command_manager.get_term(cmd_name)
            term.command[:, 0] = vx_cmd
            term.command[:, 1] = vy_cmd
            term.command[:, 2] = omega_cmd
            
            # 3. Step Policy
            # Handle JIT input (dictionary vs tensor)
            input_obs = obs["policy"] if isinstance(obs, dict) else obs
            actions = policy(input_obs)
            
            # 4. Step Physics
            obs, _, _, _, _ = env.step(actions)
            
            # 5. Logging
            frame_count += 1
            if frame_count % 10 == 0:
                pose = get_robot_pose(env)
                
                # Get actual velocity
                lin_vel = env.scene["robot"].data.root_lin_vel_w[0]
                ang_vel = env.scene["robot"].data.root_ang_vel_w[0]
                
                # Convert to body frame for comparison
                yaw = pose[2]
                v_x = lin_vel[0] * math.cos(yaw) + lin_vel[1] * math.sin(yaw)
                v_y = -lin_vel[0] * math.sin(yaw) + lin_vel[1] * math.cos(yaw)
                v_w = ang_vel[2]

                print(f"Cmd: [{vx_cmd:+.2f}, {vy_cmd:+.2f}, {omega_cmd:+.2f}] | "
                      f"Act: [{v_x:+.2f}, {v_y:+.2f}, {v_w:+.2f}] | "
                      f"Yaw: {math.degrees(yaw):.1f}°   ", end="\r")
    
    print("\nShutting down...")
    simulation_app.close()

if __name__ == "__main__":
    main()