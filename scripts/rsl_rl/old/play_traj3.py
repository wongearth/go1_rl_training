import os
import argparse
import math
import torch
import numpy as np
from isaaclab.app import AppLauncher

# --- CONFIG ---
CHECKPOINT_PATH = "logs/rsl_rl/unitree_go1_flat/model_go1_vel.pt"
TASK_NAME = "Isaac-Velocity-Flat-Unitree-Go1-v0"
TRAJECTORY_DURATION = 6.0  # seconds
MAX_LINEAR_VEL = 0.8  # m/s
MAX_ANGULAR_VEL = 1.5  # rad/s
MAX_LINEAR_ACCEL = 0.4  # m/s^2
MAX_ANGULAR_ACCEL = 1.0  # rad/s^2
MAX_RANDOM_DISTANCE = 2.5  # max combined x,y distance for new goals
WAIT_AT_GOAL = 2.0  # seconds to wait at goal before next
# --------------

parser = argparse.ArgumentParser(description="Open-Loop Trajectory Following")
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


class ConstantAccelerationTrajectory:
    """Trapezoidal velocity profile with constant acceleration."""
    
    def __init__(self, start_pose, relative_goal, max_vel, max_accel, dt):
        """
        Args:
            start_pose: [x, y, yaw] starting pose
            relative_goal: [dx, dy, dyaw] relative to start
            max_vel: maximum velocity [vx_max, vy_max, omega_max]
            max_accel: maximum acceleration [ax_max, ay_max, alpha_max]
            dt: timestep
        """
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
        self.delta = self.goal - self.start
        
        # Compute trapezoidal profile for each axis
        self.profiles = []
        max_vels = max_vel
        max_accels = max_accel
        
        for i, (distance, v_max, a_max) in enumerate(zip(self.delta, max_vels, max_accels)):
            profile = self._compute_trapezoidal_profile(abs(distance), v_max, a_max)
            profile['sign'] = np.sign(distance) if distance != 0 else 1.0
            self.profiles.append(profile)
        
        # Duration is the longest of all axes
        self.duration = max(p['duration'] for p in self.profiles)
        
    def _compute_trapezoidal_profile(self, distance, v_max, a_max):
        """Compute trapezoidal velocity profile parameters."""
        # Time to reach max velocity
        t_accel = v_max / a_max
        
        # Distance covered during accel and decel
        d_accel = 0.5 * a_max * t_accel**2
        d_decel = d_accel
        
        # Check if we reach max velocity
        if (d_accel + d_decel) <= distance:
            # Trapezoidal profile
            d_const = distance - d_accel - d_decel
            t_const = d_const / v_max
            t_decel = t_accel
            duration = t_accel + t_const + t_decel
        else:
            # Triangular profile (don't reach max velocity)
            t_accel = math.sqrt(distance / a_max)
            t_const = 0.0
            t_decel = t_accel
            duration = t_accel + t_decel
            v_max = a_max * t_accel
            d_accel = 0.5 * a_max * t_accel**2
            d_const = 0.0
            d_decel = d_accel
        
        return {
            't_accel': t_accel,
            't_const': t_const,
            't_decel': t_decel,
            'duration': duration,
            'v_max': v_max,
            'a_max': a_max,
            'd_accel': d_accel,
            'd_const': d_const,
            'd_decel': d_decel,
        }
    
    def get_state_at_time(self, t):
        """Returns [x, y, yaw, vx, vy, omega] at time t."""
        t = max(0.0, min(t, self.duration))
        
        pos = np.zeros(3)
        vel = np.zeros(3)
        
        for i, profile in enumerate(self.profiles):
            sign = profile['sign']
            t_a = profile['t_accel']
            t_c = profile['t_const']
            t_d = profile['t_decel']
            v_max = profile['v_max']
            a_max = profile['a_max']
            
            if t <= t_a:
                # Acceleration phase
                p = 0.5 * a_max * t**2
                v = a_max * t
            elif t <= (t_a + t_c):
                # Constant velocity phase
                t_in_phase = t - t_a
                p = profile['d_accel'] + v_max * t_in_phase
                v = v_max
            elif t <= (t_a + t_c + t_d):
                # Deceleration phase
                t_in_phase = t - (t_a + t_c)
                p = profile['d_accel'] + profile['d_const'] + v_max * t_in_phase - 0.5 * a_max * t_in_phase**2
                v = v_max - a_max * t_in_phase
            else:
                # Finished
                p = abs(self.delta[i])
                v = 0.0
            
            pos[i] = self.start[i] + sign * p
            vel[i] = sign * v
        
        return pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]
    
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
    print(f" >>> OPEN-LOOP TRAJECTORY FOLLOWING <<<")
    print(f" >>> Constant Acceleration Profile")
    print(f" >>> Max Vel: {MAX_LINEAR_VEL}m/s, Max Accel: {MAX_LINEAR_ACCEL}m/s²")
    print("="*60 + "\n")
    
    # Get USD stage for creating markers
    stage = omni.usd.get_context().get_stage()
    
    # 3. Main loop
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
        wait_time = 0.0
        state = "WAITING"  # States: WAITING, EXECUTING, EVALUATING
        
        # Statistics
        errors = []
        
        print("Generating first goal...\n")
        
        while simulation_app.is_running():
            current_pose = get_robot_pose(env)
            current_x, current_y, current_yaw = current_pose
            
            # Default: no movement
            vx_cmd = vy_cmd = omega_cmd = 0.0
            
            # State machine
            if state == "WAITING":
                wait_time += dt
                
                if wait_time >= WAIT_AT_GOAL or trajectory is None:
                    # Generate new goal
                    goal_count += 1
                    relative_goal = generate_random_goal(MAX_RANDOM_DISTANCE)
                    
                    start_pose = get_robot_pose(env)
                    trajectory = ConstantAccelerationTrajectory(
                        start_pose, 
                        relative_goal,
                        max_vel=[MAX_LINEAR_VEL, MAX_LINEAR_VEL, MAX_ANGULAR_VEL],
                        max_accel=[MAX_LINEAR_ACCEL, MAX_LINEAR_ACCEL, MAX_ANGULAR_ACCEL],
                        dt=dt
                    )
                    trajectory_time = 0.0
                    wait_time = 0.0
                    state = "EXECUTING"
                    
                    # Create visual marker
                    create_goal_marker(stage, trajectory.goal[:2], f"goal_marker_{goal_count}")
                    
                    print(f"\n{'='*60}")
                    print(f"GOAL #{goal_count}")
                    print(f"Start: [{start_pose[0]:.2f}, {start_pose[1]:.2f}, {math.degrees(start_pose[2]):.1f}°]")
                    print(f"Relative: [{relative_goal[0]:.2f}, {relative_goal[1]:.2f}, {math.degrees(relative_goal[2]):.1f}°]")
                    print(f"Target: [{trajectory.goal[0]:.2f}, {trajectory.goal[1]:.2f}, {math.degrees(trajectory.goal[2]):.1f}°]")
                    print(f"Duration: {trajectory.duration:.2f}s")
                    print(f"{'='*60}\n")
                # If still waiting, commands remain at default [0,0,0]
            
            elif state == "EXECUTING":
                # Pure open-loop: just follow the trajectory profile
                ref_x, ref_y, ref_yaw, vx_world, vy_world, omega = \
                    trajectory.get_state_at_time(trajectory_time)
                
                # Transform velocities to robot frame
                vx_cmd, vy_cmd, omega_cmd = world_to_robot_frame(
                    [vx_world, vy_world, omega], current_yaw
                )
                
                # Update time
                trajectory_time += dt
                
                # Check if trajectory complete
                if trajectory.is_complete(trajectory_time):
                    state = "EVALUATING"
                    
            elif state == "EVALUATING":
                # Stop and evaluate error
                vx_cmd = vy_cmd = omega_cmd = 0.0
                
                # Calculate final error
                goal_err_x = trajectory.goal[0] - current_x
                goal_err_y = trajectory.goal[1] - current_y
                dist_error = math.sqrt(goal_err_x**2 + goal_err_y**2)
                yaw_error = trajectory.goal[2] - current_yaw
                yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
                
                # Store statistics
                errors.append({
                    'goal_num': goal_count,
                    'distance': dist_error,
                    'yaw': abs(yaw_error),
                    'duration': trajectory.duration
                })
                
                print(f"\n{'*'*60}")
                print(f"📊 TRAJECTORY #{goal_count} COMPLETE")
                print(f"Planned Duration: {trajectory.duration:.2f}s")
                print(f"Position Error: {dist_error:.3f}m")
                print(f"Yaw Error: {math.degrees(yaw_error):.1f}° ({yaw_error:.3f} rad)")
                print(f"{'*'*60}\n")
                
                # Print running statistics
                if len(errors) > 0:
                    avg_dist = np.mean([e['distance'] for e in errors])
                    avg_yaw = np.mean([e['yaw'] for e in errors])
                    print(f"Running Stats (n={len(errors)}):")
                    print(f"  Avg Position Error: {avg_dist:.3f}m")
                    print(f"  Avg Yaw Error: {math.degrees(avg_yaw):.1f}°")
                    print()
                
                # Transition to waiting
                state = "WAITING"
                wait_time = 0.0
            
            # Apply velocity commands
            term = env.command_manager.get_term(cmd_name)
            term.command[:, 0] = vx_cmd
            term.command[:, 1] = vy_cmd
            term.command[:, 2] = omega_cmd
            
            # Step simulation
            actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)
            
            # Display
            if state == "EXECUTING":
                progress = min(100, int(100 * trajectory_time / trajectory.duration))
                print(f"Goal #{goal_count} | EXECUTING [{progress:3d}%] t={trajectory_time:.2f}s | "
                      f"Cmd: [{vx_cmd:+.2f}, {vy_cmd:+.2f}, {omega_cmd:+.2f}]    ",
                      end="\r")
            elif state == "WAITING":
                print(f"Goal #{goal_count} | WAITING {wait_time:.1f}/{WAIT_AT_GOAL:.1f}s    ",
                      end="\r")
    
    simulation_app.close()


if __name__ == "__main__":
    main()