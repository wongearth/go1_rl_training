import os
import argparse
import math
import torch
import numpy as np
from isaaclab.app import AppLauncher

# --- CONFIG ---
CHECKPOINT_PATH = "logs/rsl_rl/unitree_go1_flat/model_go1_vel.pt"
TASK_NAME = "Isaac-Velocity-Flat-Unitree-Go1-v0"
TRAJECTORY_DURATION = 8.0  # seconds (increased for better convergence)
MAX_LINEAR_VEL = 0.8  # m/s
MAX_ANGULAR_VEL = 1.5  # rad/s
POSITION_TOLERANCE = 0.25  # meters (relaxed tolerance)
YAW_TOLERANCE = 0.3  # radians (~17 degrees, relaxed)
MAX_RANDOM_DISTANCE = 2.5  # max combined x,y distance for new goals
TIMEOUT_DURATION = 15.0  # seconds - give up and move to next goal if taking too long
# --------------

parser = argparse.ArgumentParser(description="Go To Goal with Cubic Spline")
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
from pxr import UsdGeom, Gf, UsdShade
import omni.usd


class CubicSplineTrajectory:
    """Generates smooth cubic spline trajectory for position and orientation."""
    
    def __init__(self, start_pose, relative_goal, duration, dt):
        """
        Args:
            start_pose: [x, y, yaw] starting pose
            relative_goal: [dx, dy, dyaw] relative to start
            duration: trajectory duration in seconds
            dt: timestep
        """
        self.duration = duration
        self.dt = dt
        self.t = 0.0
        
        # Convert relative to absolute goal in start frame
        start_x, start_y, start_yaw = start_pose
        dx, dy, dyaw = relative_goal
        
        # Rotate relative goal by start yaw to get world frame goal
        cos_yaw = math.cos(start_yaw)
        sin_yaw = math.sin(start_yaw)
        goal_x = start_x + dx * cos_yaw - dy * sin_yaw
        goal_y = start_y + dx * sin_yaw + dy * cos_yaw
        goal_yaw = start_yaw + dyaw
        
        # Normalize goal yaw
        goal_yaw = (goal_yaw + math.pi) % (2 * math.pi) - math.pi
        
        self.start = np.array([start_x, start_y, start_yaw])
        self.goal = np.array([goal_x, goal_y, goal_yaw])
        
        # Store for velocity computation
        self.prev_pos = self.start.copy()
        self.prev_t = 0.0
        
    def get_reference_at_time(self, t):
        """Returns [x, y, yaw, vx, vy, omega] at time t."""
        # Clamp time
        t = max(0.0, min(t, self.duration))
        
        # Cubic spline: s(t) = a0 + a1*t + a2*t^2 + a3*t^3
        # Boundary conditions: s(0)=0, s'(0)=0, s(T)=1, s'(T)=0
        # Solution: s(t) = 3*(t/T)^2 - 2*(t/T)^3
        T = self.duration
        tau = t / T  # normalized time [0, 1]
        
        # Position interpolation
        s = 3 * tau**2 - 2 * tau**3  # smooth 0 to 1
        s_dot = (6 * tau - 6 * tau**2) / T  # velocity profile
        
        # Current pose
        pos = self.start + s * (self.goal - self.start)
        x, y, yaw = pos
        
        # Velocity in world frame
        vel = s_dot * (self.goal - self.start)
        vx, vy, omega = vel
        
        return x, y, yaw, vx, vy, omega
    
    def is_complete(self, t):
        """Check if trajectory is complete."""
        return t >= self.duration


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


def get_robot_pose(env):
    """Extract current robot pose [x, y, yaw]."""
    root_pos = env.scene["robot"].data.root_pos_w[0]
    x = root_pos[0].item()
    y = root_pos[1].item()
    
    # Quaternion to yaw
    q = env.scene["robot"].data.root_quat_w[0]
    w, qx, qy, qz = q[0].item(), q[1].item(), q[2].item(), q[3].item()
    yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
    
    return [x, y, yaw]


def world_to_robot_frame(vel_world, robot_yaw):
    """Transform velocity from world frame to robot frame."""
    vx_world, vy_world, omega = vel_world
    
    cos_yaw = math.cos(robot_yaw)
    sin_yaw = math.sin(robot_yaw)
    
    vx_robot = vx_world * cos_yaw + vy_world * sin_yaw
    vy_robot = -vx_world * sin_yaw + vy_world * cos_yaw
    
    return vx_robot, vy_robot, omega


def create_goal_marker(stage, position, name="goal_marker"):
    """Create a visual marker at the goal position."""
    prim_path = f"/World/{name}"
    
    # Remove existing marker if it exists
    existing_prim = stage.GetPrimAtPath(prim_path)
    if existing_prim.IsValid():
        stage.RemovePrim(prim_path)
    
    # Create a cone marker (pointing down)
    cone_prim = UsdGeom.Cone.Define(stage, prim_path)
    cone_prim.CreateHeightAttr(0.3)
    cone_prim.CreateRadiusAttr(0.15)
    
    # Position and rotate the cone
    cone_prim.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], 0.3))
    # Rotate 90 degrees around X to point down
    cone_prim.AddRotateXYZOp().Set(Gf.Vec3f(90, 0, 0))
    
    # Set color (bright green)
    cone_gprim = UsdGeom.Gprim(cone_prim)
    cone_gprim.CreateDisplayColorAttr([(0.0, 1.0, 0.0)])
    
    return prim_path


def generate_random_goal(max_distance):
    """Generate a random relative goal within max_distance."""
    # Random angle
    angle = np.random.uniform(0, 2 * math.pi)
    # Random distance (uniform in area, not radius)
    distance = np.random.uniform(1.0, max_distance)
    
    # Convert to x, y
    dx = distance * math.cos(angle)
    dy = distance * math.sin(angle)
    
    # Random yaw change
    dyaw = np.random.uniform(-math.pi, math.pi)
    
    return [dx, dy, dyaw]


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

    # Find command term
    cmd_name = "base_velocity"
    if cmd_name not in env.command_manager._terms:
        if "base_velocity_rough" in env.command_manager._terms:
             cmd_name = "base_velocity_rough"
        else:
             cmd_name = list(env.command_manager._terms.keys())[0]

    obs, _ = env.reset()

    print("\n" + "="*60)
    print(f" >>> AUTONOMOUS NAVIGATION WITH RANDOM GOALS <<<")
    print(f" >>> Duration: {TRAJECTORY_DURATION}s | dt: {dt}s")
    print(f" >>> Tolerances: {POSITION_TOLERANCE}m, {math.degrees(YAW_TOLERANCE):.1f}°")
    print("="*60 + "\n")
    
    # Get USD stage for creating markers
    stage = omni.usd.get_context().get_stage()
    
    # 3. Initialize trajectory
    with torch.inference_mode():
        # Reset robot position
        env.scene["robot"].write_root_pose_to_sim(
            torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]], device=device)
        )
        env.step(torch.zeros((1, env.action_manager.total_action_dim), device=device))
        
        # State variables
        goal_count = 0
        trajectory = None
        trajectory_time = 0.0
        total_time = 0.0  # Track total time on current goal
        at_goal = False
        wait_at_goal_time = 0.0
        WAIT_DURATION = 1.5  # Wait 1.5 seconds at goal before next command
        
        print("Generating first goal...\n")
        
        while simulation_app.is_running():
            # Get current robot state
            current_pose = get_robot_pose(env)
            current_x, current_y, current_yaw = current_pose
            
            # Calculate current error to goal (need this before state machine)
            if trajectory is not None:
                goal_err_x = trajectory.goal[0] - current_x
                goal_err_y = trajectory.goal[1] - current_y
                dist_to_goal = math.sqrt(goal_err_x**2 + goal_err_y**2)
                yaw_err = trajectory.goal[2] - current_yaw
                yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
            else:
                dist_to_goal = 0.0
                yaw_err = 0.0
            
            # Check if at goal (before state machine)
            goal_reached = (dist_to_goal < POSITION_TOLERANCE and abs(yaw_err) < YAW_TOLERANCE)
            
            if goal_reached and not at_goal:
                print(f"\n{'*'*60}")
                print(f"🎯 GOAL #{goal_count} REACHED!")
                print(f"Final error: {dist_to_goal:.3f}m, {math.degrees(yaw_err):.1f}°")
                print(f"{'*'*60}\n")
                at_goal = True
                wait_at_goal_time = 0.0
            
            # Check for timeout
            if trajectory is not None and not at_goal and total_time > TIMEOUT_DURATION:
                print(f"\n{'!'*60}")
                print(f"⏱️  TIMEOUT on Goal #{goal_count} after {total_time:.1f}s")
                print(f"Final error: {dist_to_goal:.3f}m, {math.degrees(yaw_err):.1f}°")
                print(f"Moving to next goal anyway...")
                print(f"{'!'*60}\n")
                # Force immediate transition to next goal
                trajectory = None
                at_goal = False
            
            # State machine: Generate new goal if needed
            if trajectory is None or (at_goal and wait_at_goal_time >= WAIT_DURATION):
                # Generate new random goal
                goal_count += 1
                relative_goal = generate_random_goal(MAX_RANDOM_DISTANCE)
                
                start_pose = get_robot_pose(env)
                trajectory = CubicSplineTrajectory(
                    start_pose, relative_goal, TRAJECTORY_DURATION, dt
                )
                trajectory_time = 0.0
                total_time = 0.0
                at_goal = False
                wait_at_goal_time = 0.0
                
                # Create visual marker at goal
                create_goal_marker(stage, trajectory.goal[:2], f"goal_marker_{goal_count}")
                
                print(f"\n{'='*60}")
                print(f"GOAL #{goal_count}")
                print(f"Start: [{start_pose[0]:.2f}, {start_pose[1]:.2f}, {math.degrees(start_pose[2]):.1f}°]")
                print(f"Relative: [{relative_goal[0]:.2f}, {relative_goal[1]:.2f}, {math.degrees(relative_goal[2]):.1f}°]")
                print(f"Target: [{trajectory.goal[0]:.2f}, {trajectory.goal[1]:.2f}, {math.degrees(trajectory.goal[2]):.1f}°]")
                print(f"{'='*60}\n")
            
            # Command generation (skip if we just created a new trajectory)
            if at_goal:
                # Stay still at goal
                vx_cmd = vy_cmd = omega_cmd = 0.0
            else:
                # Get reference from trajectory
                ref_x, ref_y, ref_yaw, vx_world, vy_world, omega = \
                    trajectory.get_reference_at_time(trajectory_time)
                
                # Transform velocities to robot frame
                vx_cmd, vy_cmd, omega_cmd = world_to_robot_frame(
                    [vx_world, vy_world, omega], current_yaw
                )
                
                # Clamp velocities
                vx_cmd = max(-MAX_LINEAR_VEL, min(MAX_LINEAR_VEL, vx_cmd))
                vy_cmd = max(-MAX_LINEAR_VEL, min(MAX_LINEAR_VEL, vy_cmd))
                omega_cmd = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, omega_cmd))
                
                # Add feedback controller for stability (increased gains)
                # Position error in world frame
                err_x = trajectory.goal[0] - current_x
                err_y = trajectory.goal[1] - current_y
                
                # Transform error to robot frame
                cos_yaw = math.cos(current_yaw)
                sin_yaw = math.sin(current_yaw)
                err_x_robot = err_x * cos_yaw + err_y * sin_yaw
                err_y_robot = -err_x * sin_yaw + err_y * cos_yaw
                
                # Proportional feedback - use trajectory feedforward + error correction
                Kp_pos = 1.2  # Increased from 0.5
                Kp_yaw = 2.0  # Increased from 1.0
                
                # After trajectory ends, rely more on feedback
                if trajectory.is_complete(trajectory_time):
                    # Pure feedback control to converge to goal
                    vx_cmd = Kp_pos * err_x_robot
                    vy_cmd = Kp_pos * err_y_robot
                    omega_cmd = Kp_yaw * yaw_err
                else:
                    # During trajectory: feedforward + feedback
                    vx_cmd += Kp_pos * err_x_robot
                    vy_cmd += Kp_pos * err_y_robot
                    omega_cmd += Kp_yaw * yaw_err
                
                # Clamp again after feedback
                vx_cmd = max(-MAX_LINEAR_VEL, min(MAX_LINEAR_VEL, vx_cmd))
                vy_cmd = max(-MAX_LINEAR_VEL, min(MAX_LINEAR_VEL, vy_cmd))
                omega_cmd = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, omega_cmd))
            
            # Inject velocity command
            term = env.command_manager.get_term(cmd_name)
            term.command[:, 0] = vx_cmd
            term.command[:, 1] = vy_cmd
            term.command[:, 2] = omega_cmd
            
            # Step simulation
            actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)
            
            # Update time counters
            if not at_goal:
                trajectory_time += dt
                total_time += dt
            else:
                # Increment wait timer when at goal
                wait_at_goal_time += dt
            
            # Progress display
            progress = min(100, int(100 * trajectory_time / TRAJECTORY_DURATION))
            if at_goal:
                status = f"AT GOAL (waiting {wait_at_goal_time:.1f}s)"
            else:
                status = f"MOVING [{progress:3d}%] t={total_time:.1f}s"
            
            print(f"Goal #{goal_count} | {status} | "
                  f"Err: {dist_to_goal:.3f}m, {math.degrees(yaw_err):+6.1f}° | "
                  f"Cmd: [{vx_cmd:+.2f}, {vy_cmd:+.2f}, {omega_cmd:+.2f}]    ",
                  end="\r")
    
    simulation_app.close()


if __name__ == "__main__":
    main()