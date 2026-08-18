import os
import argparse
import math
import torch
from isaaclab.app import AppLauncher

# --- CONFIG ---
CHECKPOINT_PATH = "logs/rsl_rl/unitree_go1_flat/model_go1_vel.pt"
TASK_NAME = "Isaac-Velocity-Flat-Unitree-Go1-v0"
# --------------

parser = argparse.ArgumentParser(description="Keyboard Velocity Control")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = False  # Must have GUI for keyboard input
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.agents.rsl_rl_ppo_cfg import UnitreeGo1RoughPPORunnerCfg
from isaaclab.terrains import TerrainImporterCfg
import carb


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


class KeyboardController:
    """Simple keyboard controller for velocity commands."""
    
    def __init__(self):
        self.vx = 0.0
        self.vy = 0.0
        self.omega = 0.0
        
        # Speed increments
        self.linear_speed_increment = 0.1
        self.angular_speed_increment = 0.2
        self.max_linear_speed = 1.5
        self.max_angular_speed = 2.0
        
        # Get keyboard interface
        self._appwindow = None
        self._input = None
        self._setup_keyboard()
        
    def _setup_keyboard(self):
        """Setup keyboard input interface."""
        import omni.appwindow
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        
    def update(self):
        """Update velocities based on keyboard input."""
        # Get keyboard state
        W = self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.W)
        S = self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.S)
        A = self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.A)
        D = self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.D)
        Q = self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.Q)
        E = self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.E)
        SPACE = self._input.get_keyboard_value(self._keyboard, carb.input.KeyboardInput.SPACE)
        
        # Forward/Backward (W/S)
        if W:
            self.vx = min(self.vx + self.linear_speed_increment, self.max_linear_speed)
        elif S:
            self.vx = max(self.vx - self.linear_speed_increment, -self.max_linear_speed)
        else:
            # Decay to zero
            if abs(self.vx) < 0.05:
                self.vx = 0.0
            else:
                self.vx *= 0.9
        
        # Left/Right strafe (A/D)
        if A:
            self.vy = min(self.vy + self.linear_speed_increment, self.max_linear_speed)
        elif D:
            self.vy = max(self.vy - self.linear_speed_increment, -self.max_linear_speed)
        else:
            # Decay to zero
            if abs(self.vy) < 0.05:
                self.vy = 0.0
            else:
                self.vy *= 0.9
        
        # Turn (Q/E)
        if Q:
            self.omega = min(self.omega + self.angular_speed_increment, self.max_angular_speed)
        elif E:
            self.omega = max(self.omega - self.angular_speed_increment, -self.max_angular_speed)
        else:
            # Decay to zero
            if abs(self.omega) < 0.05:
                self.omega = 0.0
            else:
                self.omega *= 0.9
        
        # SPACE = Emergency stop
        if SPACE:
            self.vx = 0.0
            self.vy = 0.0
            self.omega = 0.0
        
        return self.vx, self.vy, self.omega


def get_robot_pose(env):
    """Extract current robot pose [x, y, yaw]."""
    root_pos = env.scene["robot"].data.root_pos_w[0]
    x = root_pos[0].item()
    y = root_pos[1].item()
    
    q = env.scene["robot"].data.root_quat_w[0]
    w, qx, qy, qz = q[0].item(), q[1].item(), q[2].item(), q[3].item()
    yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
    
    return [x, y, yaw]


def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: Checkpoint not found at {CHECKPOINT_PATH}")
        simulation_app.close()
        return

    # Setup Environment
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

    # Load Policy
    agent_cfg_obj = UnitreeGo1RoughPPORunnerCfg()
    agent_cfg = agent_cfg_obj.to_dict()
    agent_cfg["device"] = device
    agent_cfg["seed"] = 42
    agent_cfg["policy"]["actor_hidden_dims"] = [128, 128, 128]
    agent_cfg["policy"]["critic_hidden_dims"] = [128, 128, 128]
    
    runner = OnPolicyRunner(env_wrapped, agent_cfg, log_dir=None, device=device)
    runner.load(CHECKPOINT_PATH)
    policy = runner.get_inference_policy(device=device)

    # Find command term
    cmd_name = "base_velocity"
    if cmd_name not in env.command_manager._terms:
        if "base_velocity_rough" in env.command_manager._terms:
             cmd_name = "base_velocity_rough"
        else:
             cmd_name = list(env.command_manager._terms.keys())[0]

    obs, _ = env.reset()
    
    # Initialize keyboard controller
    controller = KeyboardController()

    print("\n" + "="*70)
    print(" "*15 + "KEYBOARD VELOCITY CONTROL TEST")
    print("="*70)
    print("\nControls:")
    print("  W/S     - Forward/Backward")
    print("  A/D     - Strafe Left/Right")
    print("  Q/E     - Turn Left/Right")
    print("  SPACE   - Emergency Stop")
    print("  ESC     - Quit")
    print("\nThe robot will follow your velocity commands.")
    print("Watch if it turns correctly when you press Q/E!")
    print("="*70 + "\n")
    
    with torch.inference_mode():
        # Reset robot
        env.scene["robot"].write_root_pose_to_sim(
            torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]], device=device)
        )
        obs, _ = env.reset()
        
        frame_count = 0
        
        while simulation_app.is_running():
            # Get keyboard input
            vx_cmd, vy_cmd, omega_cmd = controller.update()
            
            # Get current pose for display
            pose = get_robot_pose(env)
            
            # Get actual velocities
            actual_lin_vel = env.scene["robot"].data.root_lin_vel_w[0]
            actual_ang_vel = env.scene["robot"].data.root_ang_vel_w[0]
            
            # Transform actual velocities to robot frame
            cos_yaw = math.cos(pose[2])
            sin_yaw = math.sin(pose[2])
            actual_vx = actual_lin_vel[0].item() * cos_yaw + actual_lin_vel[1].item() * sin_yaw
            actual_vy = -actual_lin_vel[0].item() * sin_yaw + actual_lin_vel[1].item() * cos_yaw
            actual_omega = actual_ang_vel[2].item()
            
            # Set velocity commands
            term = env.command_manager.get_term(cmd_name)
            term.command[:, 0] = vx_cmd
            term.command[:, 1] = vy_cmd
            term.command[:, 2] = omega_cmd
            
            # Step simulation
            actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)
            
            # Display status (every 10 frames)
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Pos: [{pose[0]:+.2f}, {pose[1]:+.2f}] Yaw: {math.degrees(pose[2]):+6.1f}° | "
                      f"Cmd: vx={vx_cmd:+.2f} vy={vy_cmd:+.2f} ω={omega_cmd:+.2f} | "
                      f"Actual: vx={actual_vx:+.2f} vy={actual_vy:+.2f} ω={actual_omega:+.2f}    ",
                      end="\r")
    
    print("\n\nShutting down...")
    simulation_app.close()


if __name__ == "__main__":
    main()