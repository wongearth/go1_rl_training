"""
Best Practice Trajectory Controller for Quadruped Robots
Combines: Realistic velocities, yaw feedback, proper error reporting
"""
import os
import argparse
import math
import torch
import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Best Practice Trajectory Control")
parser.add_argument("--robot", type=str, default="go1", choices=["go1", "spot"], help="Robot to test")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# --- ROBOT-SPECIFIC CONFIG ---
if args_cli.robot == "go1":
    CHECKPOINT_PATH = "logs/rsl_rl/unitree_go1_flat/model_*.pt"
    TASK_NAME = "Isaac-Velocity-Flat-Unitree-Go1-v0"
    from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.agents.rsl_rl_ppo_cfg import UnitreeGo1RoughPPORunnerCfg as AgentCfg
else:  # spot
    CHECKPOINT_PATH = "logs/rsl_rl/spot_flat/model_*.pt"
    TASK_NAME = "Isaac-Velocity-Flat-Spot-v0"
    try:
        from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.agents.rsl_rl_ppo_cfg import SpotFlatPPORunnerCfg as AgentCfg
    except ImportError:
        from isaaclab_tasks.manager_based.locomotion.velocity.config.spot.agents.rsl_rl_ppo_cfg import SpotRoughPPORunnerCfg as AgentCfg

# Realistic quadruped limits (match training range)
MAX_LINEAR_VEL = 0.8      # m/s (trained range: -1.0 to 1.0)
MAX_ANGULAR_VEL = 0.8     # rad/s (trained range: -1.0 to 1.0)
MAX_LINEAR_ACCEL = 0.6    # m/s²
MAX_ANGULAR_ACCEL = 0.8   # rad/s²

# Control mode
USE_YAW_FEEDBACK = True   # Closed-loop yaw, open-loop position
KP_YAW = 1.2             # Yaw feedback gain

# Test parameters
MAX_RANDOM_DISTANCE = 2.0  # meters
NUM_GOALS = 10
WAIT_AT_GOAL = 2.0
# ---------------------------------------

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from rsl_rl.runners import OnPolicyRunner
from isaaclab.terrains import TerrainImporterCfg
import glob


class TrapezoidalTrajectory:
    """Trapezoidal velocity profile - realistic for robots."""
    
    def __init__(self, start_pose, relative_goal, max_vel, max_accel, dt):
        self.dt = dt
        start_x, start_y, start_yaw = start_pose
        dx, dy, dyaw = relative_goal
        
        # Transform to world frame
        cos_yaw = math.cos(start_yaw)
        sin_yaw = math.sin(start_yaw)
        goal_x = start_x + dx * cos_yaw - dy * sin_yaw
        goal_y = start_y + dx * sin_yaw + dy * cos_yaw
        goal_yaw = start_yaw + dyaw
        
        # Wrap yaw properly
        goal_yaw = (goal_yaw + math.pi) % (2 * math.pi) - math.pi
        yaw_diff = goal_yaw - start_yaw
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
        goal_yaw = start_yaw + yaw_diff
        
        self.start = np.array([start_x, start_y, start_yaw])
        self.goal = np.array([goal_x, goal_y, goal_yaw])
        self.delta = self.goal - self.start
        self.delta[2] = yaw_diff
        
        # Compute profiles
        self.profiles = []
        for i, (distance, v_max, a_max) in enumerate(zip(self.delta, max_vel, max_accel)):
            profile = self._compute_profile(abs(distance), v_max, a_max)
            profile['sign'] = np.sign(distance) if distance != 0 else 1.0
            self.profiles.append(profile)
        
        self.duration = max(p['duration'] for p in self.profiles)
        
    def _compute_profile(self, distance, v_max, a_max):
        t_accel = v_max / a_max
        d_accel = 0.5 * a_max * t_accel**2
        
        if 2 * d_accel <= distance:
            d_const = distance - 2 * d_accel
            t_const = d_const / v_max
            duration = 2 * t_accel + t_const
        else:
            t_accel = math.sqrt(distance / a_max)
            t_const = 0.0
            duration = 2 * t_accel
            v_max = a_max * t_accel
            d_accel = 0.5 * a_max * t_accel**2
            d_const = 0.0
        
        return {'t_accel': t_accel, 't_const': t_const, 'duration': duration,
                'v_max': v_max, 'a_max': a_max, 'd_accel': d_accel, 'd_const': d_const}
    
    def get_state_at_time(self, t):
        t = max(0.0, min(t, self.duration))
        pos = np.zeros(3)
        vel = np.zeros(3)
        
        for i, profile in enumerate(self.profiles):
            sign = profile['sign']
            t_a = profile['t_accel']
            t_c = profile['t_const']
            v_max = profile['v_max']
            a_max = profile['a_max']
            
            if t <= t_a:
                p = 0.5 * a_max * t**2
                v = a_max * t
            elif t <= (t_a + t_c):
                p = profile['d_accel'] + v_max * (t - t_a)
                v = v_max
            elif t <= self.duration:
                t_d = t - t_a - t_c
                p = profile['d_accel'] + profile['d_const'] + v_max * t_d - 0.5 * a_max * t_d**2
                v = v_max - a_max * t_d
            else:
                p = abs(self.delta[i])
                v = 0.0
            
            pos[i] = self.start[i] + sign * p
            vel[i] = sign * v
            
            if i == 2:  # Normalize yaw
                pos[i] = (pos[i] + math.pi) % (2 * math.pi) - math.pi
        
        return pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]
    
    def is_complete(self, t):
        return t >= self.duration


class SimpleRslRlWrapper:
    def __init__(self, env):
        self.env = env
        self.device = env.device
        self.num_envs = env.num_envs
        self.num_obs = env.observation_manager.group_obs_dim["policy"][0]
        self.num_actions = env.action_manager.total_action_dim
        self.num_privileged_obs = env.observation_manager.group_obs_dim.get("critic", [None])[0]

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
    root_pos = env.scene["robot"].data.root_pos_w[0]
    x, y = root_pos[0].item(), root_pos[1].item()
    q = env.scene["robot"].data.root_quat_w[0]
    w, qx, qy, qz = q[0].item(), q[1].item(), q[2].item(), q[3].item()
    yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
    return [x, y, yaw]


def world_to_robot_frame(vel_world, robot_yaw):
    vx_w, vy_w, omega = vel_world
    cos_yaw, sin_yaw = math.cos(robot_yaw), math.sin(robot_yaw)
    vx_r = vx_w * cos_yaw + vy_w * sin_yaw
    vy_r = -vx_w * sin_yaw + vy_w * cos_yaw
    return vx_r, vy_r, omega


def main():
    checkpoint_files = glob.glob(CHECKPOINT_PATH)
    if not checkpoint_files:
        print(f"Error: No checkpoint at {CHECKPOINT_PATH}")
        print(f"\nLooking for {args_cli.robot.upper()} checkpoints...")
        print(f"Make sure you have trained or downloaded a model for {args_cli.robot}")
        simulation_app.close()
        return
    checkpoint_path = sorted(checkpoint_files)[-1]
    print(f"\n🤖 Testing {args_cli.robot.upper()} robot")
    print(f"📁 Checkpoint: {checkpoint_path}\n")

    # Setup env
    env_cfg = parse_env_cfg(TASK_NAME)
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane",
        collision_group=-1, physics_material=None, visual_material=None, debug_vis=False)
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.5)
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum.terrain_levels = None
    if hasattr(env_cfg.scene, 'height_scanner'):
        env_cfg.scene.height_scanner = None
    if hasattr(env_cfg.observations.policy, 'height_scan'):
        env_cfg.observations.policy.height_scan = None

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env_wrapped = SimpleRslRlWrapper(env)
    device, dt = env.device, env.step_dt

    # Load policy
    agent_cfg_obj = AgentCfg()
    agent_cfg = agent_cfg_obj.to_dict()
    agent_cfg["device"] = device
    agent_cfg["seed"] = 42
    
    runner = OnPolicyRunner(env_wrapped, agent_cfg, log_dir=None, device=device)
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=device)

    cmd_name = list(env.command_manager._terms.keys())[0]
    obs, _ = env.reset()

    print("\n" + "="*70)
    print(f"  TRAJECTORY CONTROLLER - {args_cli.robot.upper()}")
    print("="*70)
    print(f"  Max Linear Vel:  {MAX_LINEAR_VEL} m/s")
    print(f"  Max Angular Vel: {MAX_ANGULAR_VEL} rad/s ({math.degrees(MAX_ANGULAR_VEL):.0f}°/s)")
    print(f"  Yaw Control:     {'CLOSED-LOOP (Feedback)' if USE_YAW_FEEDBACK else 'OPEN-LOOP'}")
    print(f"  Goals to test:   {NUM_GOALS}")
    print("="*70 + "\n")

    errors = []
    
    with torch.inference_mode():
        env.scene["robot"].write_root_pose_to_sim(
            torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]], device=device))
        env.step(torch.zeros((1, env.action_manager.total_action_dim), device=device))
        
        goal_count = 0
        trajectory = None
        trajectory_time = 0.0
        wait_time = 0.0
        state = "WAITING"
        
        while simulation_app.is_running() and goal_count < NUM_GOALS:
            current_pose = get_robot_pose(env)
            current_x, current_y, current_yaw = current_pose
            vx_cmd = vy_cmd = omega_cmd = 0.0
            
            if state == "WAITING":
                wait_time += dt
                if wait_time >= WAIT_AT_GOAL or trajectory is None:
                    goal_count += 1
                    angle = np.random.uniform(0, 2 * math.pi)
                    distance = np.random.uniform(1.0, MAX_RANDOM_DISTANCE)
                    relative_goal = [distance * math.cos(angle), distance * math.sin(angle),
                                   np.random.uniform(-math.pi/2, math.pi/2)]
                    
                    start_pose = get_robot_pose(env)
                    trajectory = TrapezoidalTrajectory(
                        start_pose, relative_goal,
                        max_vel=[MAX_LINEAR_VEL, MAX_LINEAR_VEL, MAX_ANGULAR_VEL],
                        max_accel=[MAX_LINEAR_ACCEL, MAX_LINEAR_ACCEL, MAX_ANGULAR_ACCEL],
                        dt=dt)
                    trajectory_time = 0.0
                    wait_time = 0.0
                    state = "EXECUTING"
                    
                    print(f"Goal #{goal_count}: Target [{trajectory.goal[0]:.2f}, {trajectory.goal[1]:.2f}, {math.degrees(trajectory.goal[2]):.0f}°]")
            
            elif state == "EXECUTING":
                ref_x, ref_y, ref_yaw, vx_w, vy_w, omega_ff = trajectory.get_state_at_time(trajectory_time)
                
                if USE_YAW_FEEDBACK:
                    # Add yaw feedback
                    yaw_error = trajectory.goal[2] - current_yaw
                    yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
                    omega_fb = KP_YAW * yaw_error
                    omega_w = omega_ff + omega_fb
                else:
                    omega_w = omega_ff
                
                vx_cmd, vy_cmd, omega_cmd = world_to_robot_frame([vx_w, vy_w, omega_w], current_yaw)
                vx_cmd = max(-MAX_LINEAR_VEL, min(MAX_LINEAR_VEL, vx_cmd))
                vy_cmd = max(-MAX_LINEAR_VEL, min(MAX_LINEAR_VEL, vy_cmd))
                omega_cmd = max(-MAX_ANGULAR_VEL, min(MAX_ANGULAR_VEL, omega_cmd))
                
                trajectory_time += dt
                if trajectory.is_complete(trajectory_time):
                    state = "EVALUATING"
            
            elif state == "EVALUATING":
                err_x = trajectory.goal[0] - current_x
                err_y = trajectory.goal[1] - current_y
                dist_error = math.sqrt(err_x**2 + err_y**2)
                yaw_error = trajectory.goal[2] - current_yaw
                yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
                
                errors.append({'dist': dist_error, 'yaw': abs(yaw_error)})
                
                print(f"  ✓ Position Error: {dist_error:.3f}m, Yaw Error: {math.degrees(yaw_error):.1f}°")
                
                state = "WAITING"
                wait_time = 0.0
            
            # Set commands
            term = env.command_manager.get_term(cmd_name)
            term.command[:, 0] = vx_cmd
            term.command[:, 1] = vy_cmd
            term.command[:, 2] = omega_cmd
            
            actions = policy(obs)
            obs, _, _, _, _ = env.step(actions)
        
        # Final summary
        if errors:
            dist_errors = [e['dist'] for e in errors]
            yaw_errors_deg = [math.degrees(e['yaw']) for e in errors]
            
            print("\n" + "="*70)
            print(f"  FINAL RESULTS - {args_cli.robot.upper()}")
            print("="*70)
            print(f"  Position Error: {np.mean(dist_errors):.3f} ± {np.std(dist_errors):.3f} m")
            print(f"  Yaw Error:      {np.mean(yaw_errors_deg):.1f} ± {np.std(yaw_errors_deg):.1f}°")
            print(f"\n  🎯 Compare Go1 vs Spot turning performance!")
            print("="*70 + "\n")
    
    simulation_app.close()


if __name__ == "__main__":
    main()