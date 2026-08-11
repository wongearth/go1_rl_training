import os
import argparse
import math
import torch
import numpy as np
from isaaclab.app import AppLauncher

# --- CONFIG ---
CHECKPOINT_PATH = "logs/rsl_rl/unitree_go1_flat/model_go1_vel.pt"
TASK_NAME = "Isaac-Velocity-Flat-Unitree-Go1-v0"
MAX_LINEAR_VEL = 1.5  # m/s (reduced from 1.5 to reduce load)
MAX_ANGULAR_VEL = 2.  # rad/s (reduced from 2.5)
MAX_LINEAR_ACCEL = 0.8  # m/s^2 (reduced from 1.0)
MAX_ANGULAR_ACCEL = 1.5  # rad/s^2 (reduced from 2.0)
MAX_RANDOM_DISTANCE = 2.5  # max combined x,y distance for new goals
WAIT_AT_GOAL = 2.0  # seconds to wait at goal before next (increased to give GPU rest)
ENABLE_RENDERING_THROTTLE = True  # Reduce rendering frequency
NUM_GOALS = 20  # Number of goals to test before stopping
SAVE_RESULTS = True  # Save results to CSV and generate plots
OUTPUT_DIR = "results/open_loop_tracking"  # Where to save results
# --------------

parser = argparse.ArgumentParser(description="Open-Loop Trajectory Following")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Force headless mode to avoid GUI freezing issues
if not hasattr(args_cli, 'headless') or args_cli.headless is None:
    args_cli.headless = True
    
args_cli.enable_cameras = False  # Disable camera rendering to reduce load
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
        
        # Start pose in world frame
        start_x, start_y, start_yaw = start_pose
        dx, dy, dyaw = relative_goal
        
        # FIXED: Rotate relative goal by start yaw to get world frame goal
        # The relative goal [dx, dy] is in the robot's frame, not world frame
        cos_yaw = math.cos(start_yaw)
        sin_yaw = math.sin(start_yaw)
        goal_x = start_x + dx * cos_yaw - dy * sin_yaw
        goal_y = start_y + dx * sin_yaw + dy * cos_yaw
        
        # FIXED: Goal yaw is absolute, not relative to start yaw
        # The dyaw is already the desired change in orientation
        goal_yaw = start_yaw + dyaw
        
        # Normalize goal yaw to [-pi, pi]
        goal_yaw = (goal_yaw + math.pi) % (2 * math.pi) - math.pi
        
        # IMPORTANT: For yaw trajectories, we need to take the shortest path
        # If the difference is > 180 degrees, go the other way
        yaw_diff = goal_yaw - start_yaw
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
        goal_yaw = start_yaw + yaw_diff
        
        self.start = np.array([start_x, start_y, start_yaw])
        self.goal = np.array([goal_x, goal_y, goal_yaw])
        self.delta = self.goal - self.start
        
        # IMPORTANT: For yaw, use the wrapped difference
        self.delta[2] = yaw_diff
        
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
            
            # FIXED: For yaw (index 2), normalize the result
            if i == 2:
                # Yaw position
                pos[i] = self.start[i] + sign * p
                # Normalize to [-pi, pi]
                pos[i] = (pos[i] + math.pi) % (2 * math.pi) - math.pi
                vel[i] = sign * v
            else:
                # X, Y positions
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


def save_results_to_csv(errors, output_dir):
    """Save trajectory results to CSV file."""
    import csv
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(output_dir, f"tracking_results_{timestamp}.csv")
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Goal', 'Distance_Error_m', 'Yaw_Error_deg', 'Yaw_Error_rad', 
                        'Duration_s', 'Total_Time_s'])
        for e in errors:
            writer.writerow([
                e['goal_num'],
                f"{e['distance']:.4f}",
                f"{math.degrees(e['yaw']):.2f}",
                f"{e['yaw']:.4f}",
                f"{e['duration']:.2f}",
                f"{e['total_time']:.1f}"
            ])
    
    print(f"\n✅ Results saved to: {csv_path}")
    return csv_path


def generate_plots(errors, output_dir):
    """Generate plots of tracking performance."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend for headless
        import matplotlib.pyplot as plt
        from datetime import datetime
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract data
        goals = [e['goal_num'] for e in errors]
        dist_errors = [e['distance'] for e in errors]
        yaw_errors_deg = [math.degrees(e['yaw']) for e in errors]
        durations = [e['duration'] for e in errors]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Open-Loop Trajectory Tracking Performance', fontsize=16, fontweight='bold')
        
        # 1. Position Error over time
        ax = axes[0, 0]
        ax.plot(goals, dist_errors, 'b-o', linewidth=2, markersize=4)
        ax.axhline(np.mean(dist_errors), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(dist_errors):.3f}m')
        ax.set_xlabel('Goal Number', fontweight='bold')
        ax.set_ylabel('Position Error (m)', fontweight='bold')
        ax.set_title('Position Tracking Error', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 2. Yaw Error over time
        ax = axes[0, 1]
        ax.plot(goals, yaw_errors_deg, 'g-o', linewidth=2, markersize=4)
        ax.axhline(np.mean(yaw_errors_deg), color='r', linestyle='--',
                   label=f'Mean: {np.mean(yaw_errors_deg):.1f}°')
        ax.set_xlabel('Goal Number', fontweight='bold')
        ax.set_ylabel('Yaw Error (degrees)', fontweight='bold')
        ax.set_title('Orientation Tracking Error', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 3. Error Distribution Histogram
        ax = axes[1, 0]
        ax.hist(dist_errors, bins=15, color='blue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(dist_errors), color='r', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(dist_errors):.3f}m')
        ax.axvline(np.median(dist_errors), color='orange', linestyle='--', linewidth=2,
                   label=f'Median: {np.median(dist_errors):.3f}m')
        ax.set_xlabel('Position Error (m)', fontweight='bold')
        ax.set_ylabel('Frequency', fontweight='bold')
        ax.set_title('Position Error Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Summary Statistics Table
        ax = axes[1, 1]
        ax.axis('off')
        
        stats_data = [
            ['Metric', 'Mean', 'Std Dev', 'Min', 'Max'],
            ['Position (m)', 
             f'{np.mean(dist_errors):.3f}',
             f'{np.std(dist_errors):.3f}',
             f'{np.min(dist_errors):.3f}',
             f'{np.max(dist_errors):.3f}'],
            ['Yaw (deg)',
             f'{np.mean(yaw_errors_deg):.1f}',
             f'{np.std(yaw_errors_deg):.1f}',
             f'{np.min(yaw_errors_deg):.1f}',
             f'{np.max(yaw_errors_deg):.1f}'],
            ['Duration (s)',
             f'{np.mean(durations):.2f}',
             f'{np.std(durations):.2f}',
             f'{np.min(durations):.2f}',
             f'{np.max(durations):.2f}'],
        ]
        
        table = ax.table(cellText=stats_data, cellLoc='center', loc='center',
                        colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header row
        for i in range(5):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Summary Statistics', fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(output_dir, f"tracking_performance_{timestamp}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Plots saved to: {plot_path}")
        return plot_path
        
    except ImportError:
        print("\n⚠️  Matplotlib not available. Install with: pip install matplotlib")
        return None
    except Exception as e:
        print(f"\n⚠️  Could not generate plots: {e}")
        return None


def print_final_summary(errors, total_elapsed_time):
    """Print comprehensive final summary."""
    dist_errors = [e['distance'] for e in errors]
    yaw_errors = [e['yaw'] for e in errors]
    yaw_errors_deg = [math.degrees(y) for y in yaw_errors]
    durations = [e['duration'] for e in errors]
    
    print("\n" + "="*70)
    print(" "*20 + "FINAL PERFORMANCE SUMMARY")
    print("="*70)
    print(f"\n📊 Total Goals Completed: {len(errors)}")
    print(f"⏱️  Total Elapsed Time: {total_elapsed_time:.1f}s")
    print(f"⏱️  Total Moving Time: {sum(durations):.1f}s")
    print(f"⏱️  Total Waiting Time: {total_elapsed_time - sum(durations):.1f}s")
    
    print("\n" + "-"*70)
    print("POSITION TRACKING ERROR (meters)")
    print("-"*70)
    print(f"  Mean:     {np.mean(dist_errors):.4f} m")
    print(f"  Median:   {np.median(dist_errors):.4f} m")
    print(f"  Std Dev:  {np.std(dist_errors):.4f} m")
    print(f"  Min:      {np.min(dist_errors):.4f} m")
    print(f"  Max:      {np.max(dist_errors):.4f} m")
    
    print("\n" + "-"*70)
    print("ORIENTATION TRACKING ERROR (degrees)")
    print("-"*70)
    print(f"  Mean:     {np.mean(yaw_errors_deg):.2f}°")
    print(f"  Median:   {np.median(yaw_errors_deg):.2f}°")
    print(f"  Std Dev:  {np.std(yaw_errors_deg):.2f}°")
    print(f"  Min:      {np.min(yaw_errors_deg):.2f}°")
    print(f"  Max:      {np.max(yaw_errors_deg):.2f}°")
    
    print("\n" + "-"*70)
    print("TRAJECTORY DURATION (seconds)")
    print("-"*70)
    print(f"  Mean:     {np.mean(durations):.2f} s")
    print(f"  Median:   {np.median(durations):.2f} s")
    print(f"  Std Dev:  {np.std(durations):.2f} s")
    print(f"  Min:      {np.min(durations):.2f} s")
    print(f"  Max:      {np.max(durations):.2f} s")
    
    print("\n" + "="*70)
    print(" "*15 + "✅ EVALUATION COMPLETE")
    print("="*70 + "\n")


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
    print(f" >>> Mode: {'HEADLESS (No GUI)' if args_cli.headless else 'GUI'}")
    print("="*60 + "\n")
    
    # Get USD stage for creating markers (only if not headless)
    stage = None
    if not args_cli.headless:
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
        total_elapsed_time = 0.0  # Track total time since start
        wait_time = 0.0
        state = "WAITING"  # States: WAITING, EXECUTING, EVALUATING
        frame_count = 0  # For rendering throttle
        
        # Statistics
        errors = []
        velocity_tracking = []  # Track commanded vs actual velocities
        
        print("Generating first goal...\n")
        
        while simulation_app.is_running() and goal_count < NUM_GOALS:
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
                    
                    # Create visual marker (only if not headless)
                    if stage is not None:
                        create_goal_marker(stage, trajectory.goal[:2], f"goal_marker_{goal_count}")
                    
                    print(f"\n{'='*60}")
                    print(f"GOAL #{goal_count}")
                    print(f"Start: [{start_pose[0]:.2f}, {start_pose[1]:.2f}, {math.degrees(start_pose[2]):.1f}°]")
                    print(f"Relative: [{relative_goal[0]:.2f}, {relative_goal[1]:.2f}, {math.degrees(relative_goal[2]):.1f}°]")
                    print(f"Target: [{trajectory.goal[0]:.2f}, {trajectory.goal[1]:.2f}, {math.degrees(trajectory.goal[2]):.1f}°]")
                    print(f"Yaw Delta: {math.degrees(trajectory.delta[2]):.1f}°")
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
                
                # Get actual velocities for tracking analysis
                actual_lin_vel = env.scene["robot"].data.root_lin_vel_w[0]
                actual_ang_vel = env.scene["robot"].data.root_ang_vel_w[0]
                
                # Transform actual velocities to robot frame for comparison
                actual_vx = actual_lin_vel[0].item() * math.cos(current_yaw) + actual_lin_vel[1].item() * math.sin(current_yaw)
                actual_vy = -actual_lin_vel[0].item() * math.sin(current_yaw) + actual_lin_vel[1].item() * math.cos(current_yaw)
                actual_omega = actual_ang_vel[2].item()
                
                # Store velocity tracking data
                velocity_tracking.append({
                    'time': trajectory_time,
                    'vx_cmd': vx_cmd,
                    'vy_cmd': vy_cmd,
                    'omega_cmd': omega_cmd,
                    'vx_actual': actual_vx,
                    'vy_actual': actual_vy,
                    'omega_actual': actual_omega,
                    'vx_error': abs(vx_cmd - actual_vx),
                    'vy_error': abs(vy_cmd - actual_vy),
                    'omega_error': abs(omega_cmd - actual_omega)
                })
                
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
                    'duration': trajectory.duration,
                    'total_time': total_elapsed_time
                })
                
                # Calculate velocity tracking stats for this trajectory
                if len(velocity_tracking) > 0:
                    recent_tracking = velocity_tracking[-int(trajectory.duration/dt):]  # Last trajectory
                    avg_vx_error = np.mean([v['vx_error'] for v in recent_tracking])
                    avg_omega_error = np.mean([v['omega_error'] for v in recent_tracking])
                else:
                    avg_vx_error = 0
                    avg_omega_error = 0
                
                print(f"\n{'*'*60}")
                print(f"📊 TRAJECTORY #{goal_count} COMPLETE")
                print(f"Planned Duration: {trajectory.duration:.2f}s")
                print(f"Position Error: {dist_error:.3f}m")
                print(f"Yaw Error: {math.degrees(yaw_error):.1f}° ({yaw_error:.3f} rad)")
                print(f"Velocity Tracking - Vx Error: {avg_vx_error:.3f} m/s, Omega Error: {avg_omega_error:.3f} rad/s")
                print(f"Total Elapsed Time: {total_elapsed_time:.1f}s")
                print(f"{'*'*60}\n")
                
                # Print running statistics
                if len(errors) > 0:
                    avg_dist = np.mean([e['distance'] for e in errors])
                    avg_yaw = np.mean([e['yaw'] for e in errors])
                    avg_duration = np.mean([e['duration'] for e in errors])
                    total_move_time = sum([e['duration'] for e in errors])
                    total_wait_time = total_elapsed_time - total_move_time
                    
                    # Overall velocity tracking stats
                    if len(velocity_tracking) > 0:
                        overall_vx_error = np.mean([v['vx_error'] for v in velocity_tracking])
                        overall_omega_error = np.mean([v['omega_error'] for v in velocity_tracking])
                    else:
                        overall_vx_error = 0
                        overall_omega_error = 0
                    
                    print(f"Running Stats (n={len(errors)}):")
                    print(f"  Avg Position Error: {avg_dist:.3f}m")
                    print(f"  Avg Yaw Error: {math.degrees(avg_yaw):.1f}°")
                    print(f"  Avg Vx Tracking Error: {overall_vx_error:.3f} m/s")
                    print(f"  Avg Omega Tracking Error: {overall_omega_error:.3f} rad/s ({math.degrees(overall_omega_error):.1f}°/s)")
                    print(f"  Avg Trajectory Duration: {avg_duration:.2f}s")
                    print(f"  Total Time: {total_elapsed_time:.1f}s (Moving: {total_move_time:.1f}s, Waiting: {total_wait_time:.1f}s)")
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
            
            # Update total elapsed time
            total_elapsed_time += dt
            frame_count += 1
            
            # Display (throttle to every 10 frames to reduce terminal overhead)
            if frame_count % 10 == 0:
                if state == "EXECUTING":
                    progress = min(100, int(100 * trajectory_time / trajectory.duration))
                    print(f"Goal #{goal_count} | EXECUTING [{progress:3d}%] t={trajectory_time:.2f}s | Total: {total_elapsed_time:.1f}s | "
                          f"Cmd: [{vx_cmd:+.2f}, {vy_cmd:+.2f}, {omega_cmd:+.2f}]    ",
                          end="\r")
                elif state == "WAITING":
                    print(f"Goal #{goal_count} | WAITING {wait_time:.1f}/{WAIT_AT_GOAL:.1f}s | Total: {total_elapsed_time:.1f}s    ",
                          end="\r")
        
        # After completing all goals, show final summary
        print("\n\n" + "="*70)
        print(" "*20 + "ALL GOALS COMPLETED!")
        print("="*70)
        
        print_final_summary(errors, total_elapsed_time)
        
        # Save results
        if SAVE_RESULTS and len(errors) > 0:
            csv_path = save_results_to_csv(errors, OUTPUT_DIR)
            plot_path = generate_plots(errors, OUTPUT_DIR)
            
            print("\n📁 Output Files:")
            print(f"   CSV:  {csv_path}")
            if plot_path:
                print(f"   Plot: {plot_path}")
        
        print("\nPress Ctrl+C to exit...")
        
        # Keep simulation running briefly to show final state
        for _ in range(50):
            if not simulation_app.is_running():
                break
            env.step(torch.zeros((1, env.action_manager.total_action_dim), device=device))
    
    simulation_app.close()


if __name__ == "__main__":
    main()