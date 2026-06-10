# scripts/reinforcement_learning/rsl_rl/play_traj_spot.py

import os
import argparse
import math
import torch
import numpy as np
from isaaclab.app import AppLauncher

# --- CONFIG ---
# Path to your working Spot JIT model
CHECKPOINT_PATH = "logs/rsl_rl/spot_flat/model_spot.pt"
TASK_NAME = "Isaac-Velocity-Flat-Spot-v0"

# Trajectory Parameters (Tuned for Spot)
MAX_LINEAR_VEL = 1.0  # m/s 
MAX_ANGULAR_VEL = 1.5  # rad/s 
MAX_LINEAR_ACCEL = 0.8  # m/s^2 
MAX_ANGULAR_ACCEL = 1.0  # rad/s^2 

MAX_RANDOM_DISTANCE = 3.0  # Spot can travel further
WAIT_AT_GOAL = 2.0  
NUM_GOALS = 20  
SAVE_RESULTS = True  
OUTPUT_DIR = "results/spot_tracking"
# --------------

parser = argparse.ArgumentParser(description="Spot Open-Loop Trajectory")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()

# Force headless if not specified, but allow GUI if requested
if not hasattr(args_cli, 'headless') or args_cli.headless is None:
    args_cli.headless = False # Default to GUI so you can see it
    
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.terrains import TerrainImporterCfg
from pxr import UsdGeom, Gf
import omni.usd

# --- TRAJECTORY GENERATOR (Unchanged) ---
class ConstantAccelerationTrajectory:
    """Trapezoidal velocity profile with constant acceleration."""
    def __init__(self, start_pose, relative_goal, max_vel, max_accel, dt):
        self.dt = dt
        self.t = 0.0
        
        start_x, start_y, start_yaw = start_pose
        dx, dy, dyaw = relative_goal
        
        # Transform relative goal to world frame
        cos_yaw = math.cos(start_yaw)
        sin_yaw = math.sin(start_yaw)
        goal_x = start_x + dx * cos_yaw - dy * sin_yaw
        goal_y = start_y + dx * sin_yaw + dy * cos_yaw
        
        # Calculate absolute goal yaw
        goal_yaw = start_yaw + dyaw
        
        # Normalize goal yaw
        goal_yaw = (goal_yaw + math.pi) % (2 * math.pi) - math.pi
        
        # Shortest path for yaw
        yaw_diff = goal_yaw - start_yaw
        yaw_diff = (yaw_diff + math.pi) % (2 * math.pi) - math.pi
        goal_yaw = start_yaw + yaw_diff
        
        self.start = np.array([start_x, start_y, start_yaw])
        self.goal = np.array([goal_x, goal_y, goal_yaw])
        self.delta = self.goal - self.start
        self.delta[2] = yaw_diff # Use wrapped diff
        
        self.profiles = []
        for i, (distance, v_max, a_max) in enumerate(zip(self.delta, max_vel, max_accel)):
            profile = self._compute_trapezoidal_profile(abs(distance), v_max, a_max)
            profile['sign'] = np.sign(distance) if distance != 0 else 1.0
            self.profiles.append(profile)
        
        self.duration = max(p['duration'] for p in self.profiles)
        
    def _compute_trapezoidal_profile(self, distance, v_max, a_max):
        t_accel = v_max / a_max
        d_accel = 0.5 * a_max * t_accel**2
        d_decel = d_accel
        
        if (d_accel + d_decel) <= distance:
            d_const = distance - d_accel - d_decel
            t_const = d_const / v_max
            t_decel = t_accel
            duration = t_accel + t_const + t_decel
        else:
            t_accel = math.sqrt(distance / a_max)
            t_const = 0.0
            t_decel = t_accel
            duration = t_accel + t_decel
            v_max = a_max * t_accel
            d_accel = 0.5 * a_max * t_accel**2
            d_const = 0.0
            d_decel = d_accel
        
        return {
            't_accel': t_accel, 't_const': t_const, 't_decel': t_decel,
            'duration': duration, 'v_max': v_max, 'a_max': a_max,
            'd_accel': d_accel, 'd_const': d_const, 'd_decel': d_decel,
        }
    
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
                t_in = t - t_a
                p = profile['d_accel'] + v_max * t_in
                v = v_max
            elif t <= (t_a + t_c + profile['t_decel']):
                t_in = t - (t_a + t_c)
                p = profile['d_accel'] + profile['d_const'] + v_max * t_in - 0.5 * a_max * t_in**2
                v = v_max - a_max * t_in
            else:
                p = abs(self.delta[i])
                v = 0.0
            
            if i == 2: # Yaw normalization
                pos[i] = self.start[i] + sign * p
                pos[i] = (pos[i] + math.pi) % (2 * math.pi) - math.pi
                vel[i] = sign * v
            else:
                pos[i] = self.start[i] + sign * p
                vel[i] = sign * v
        
        return pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]
    
    def is_complete(self, t):
        return t >= self.duration

# --- HELPERS ---
def get_robot_pose(env):
    root_pos = env.scene["robot"].data.root_pos_w[0]
    q = env.scene["robot"].data.root_quat_w[0]
    w, qx, qy, qz = q[0].item(), q[1].item(), q[2].item(), q[3].item()
    yaw = math.atan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))
    return [root_pos[0].item(), root_pos[1].item(), yaw]

def world_to_robot_frame(vel_world, robot_yaw):
    vx_w, vy_w, omega = vel_world
    cos_y = math.cos(robot_yaw)
    sin_y = math.sin(robot_yaw)
    vx_r = vx_w * cos_y + vy_w * sin_y
    vy_r = -vx_w * sin_y + vy_w * cos_y
    return vx_r, vy_r, omega

def generate_random_goal(max_distance):
    angle = np.random.uniform(0, 2 * math.pi)
    distance = np.random.uniform(1.0, max_distance)
    dx = distance * math.cos(angle)
    dy = distance * math.sin(angle)
    dyaw = np.random.uniform(-math.pi/2, math.pi/2) # Limit yaw change for Spot
    return [dx, dy, dyaw]

def create_goal_marker(stage, position, yaw, name="goal_marker"):
    prim_path = f"/World/{name}"
    if stage.GetPrimAtPath(prim_path).IsValid():
        stage.RemovePrim(prim_path)
    
    cone = UsdGeom.Cone.Define(stage, prim_path)
    cone.CreateHeightAttr(0.4)
    cone.CreateRadiusAttr(0.1)
    cone.AddTranslateOp().Set(Gf.Vec3d(position[0], position[1], 0.2))
    cone.AddRotateXYZOp().Set(Gf.Vec3f(0, 90, math.degrees(yaw)))
    UsdGeom.Gprim(cone).CreateDisplayColorAttr([(0.0, 1.0, 0.0)])
    return prim_path

def save_results_to_csv(errors, output_dir):
    import csv
    from datetime import datetime
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"spot_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Goal', 'Dist_Err', 'Yaw_Err_Deg', 'Duration'])
        for e in errors:
            writer.writerow([e['goal_num'], f"{e['distance']:.4f}", f"{math.degrees(e['yaw']):.2f}", f"{e['duration']:.2f}"])
    print(f"\n✅ Saved CSV: {path}")

# --- MAIN ---
def main():
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"Error: JIT model not found at {CHECKPOINT_PATH}")
        simulation_app.close()
        return

    # 1. Setup Spot Environment
    env_cfg = parse_env_cfg(TASK_NAME)
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1, 
        physics_material=None, visual_material=None, debug_vis=False
    )
    
    # CRITICAL FIX: Disable Curriculum for Flat Terrain
    env_cfg.curriculum.terrain_levels = None 
    
    # CRITICAL FIX: Spot needs higher spawn point
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.75) 
    env_cfg.scene.num_envs = 1
    
    env = ManagerBasedRLEnv(cfg=env_cfg)
    device = env.device
    dt = env.step_dt

    # 2. Load JIT Policy (Direct Torch Load)
    print(f"[INFO] Loading Spot JIT Policy: {CHECKPOINT_PATH}")
    try:
        policy = torch.jit.load(CHECKPOINT_PATH, map_location=device)
        policy.eval()
    except Exception as e:
        print(f"[ERROR] Failed to load JIT: {e}")
        simulation_app.close()
        return

    # Command Term
    cmd_name = "base_velocity"
    if cmd_name not in env.command_manager._terms:
        cmd_name = list(env.command_manager._terms.keys())[0]

    obs, _ = env.reset()
    stage = omni.usd.get_context().get_stage() if not args_cli.headless else None

    print("\n" + "="*60)
    print(" >>> SPOT TRAJECTORY TRACKING START <<<")
    print("="*60 + "\n")

    # 3. Execution Loop
    with torch.inference_mode():
        # Zero actions init
        env.step(torch.zeros((1, env.action_manager.total_action_dim), device=device))
        
        goal_count = 0
        trajectory = None
        traj_time = 0.0
        total_time = 0.0
        wait_time = 0.0
        state = "WAITING"
        errors = []
        
        while simulation_app.is_running() and goal_count < NUM_GOALS:
            curr_pose = get_robot_pose(env)
            vx_cmd = vy_cmd = omega_cmd = 0.0
            
            if state == "WAITING":
                wait_time += dt
                if wait_time >= WAIT_AT_GOAL or trajectory is None:
                    goal_count += 1
                    rel_goal = generate_random_goal(MAX_RANDOM_DISTANCE)
                    trajectory = ConstantAccelerationTrajectory(
                        curr_pose, rel_goal,
                        max_vel=[MAX_LINEAR_VEL, MAX_LINEAR_VEL, MAX_ANGULAR_VEL],
                        max_accel=[MAX_LINEAR_ACCEL, MAX_LINEAR_ACCEL, MAX_ANGULAR_ACCEL],
                        dt=dt
                    )
                    traj_time = 0.0
                    wait_time = 0.0
                    state = "EXECUTING"
                    
                    if stage:
                        create_goal_marker(stage, trajectory.goal[:2], trajectory.goal[2])
                    
                    print(f"Goal {goal_count}: Rel {rel_goal[0]:.1f}m, {rel_goal[1]:.1f}m, {math.degrees(rel_goal[2]):.1f}° | Dur: {trajectory.duration:.1f}s")

            elif state == "EXECUTING":
                # Get Trajectory Setpoint
                _, _, _, vx_w, vy_w, omega = trajectory.get_state_at_time(traj_time)
                
                # World -> Robot Frame
                vx_cmd, vy_cmd, omega_cmd = world_to_robot_frame([vx_w, vy_w, omega], curr_pose[2])
                
                traj_time += dt
                if trajectory.is_complete(traj_time):
                    state = "EVALUATING"

            elif state == "EVALUATING":
                # Calc Errors
                dx = trajectory.goal[0] - curr_pose[0]
                dy = trajectory.goal[1] - curr_pose[1]
                d_err = math.sqrt(dx**2 + dy**2)
                yaw_err = (trajectory.goal[2] - curr_pose[2] + math.pi) % (2*math.pi) - math.pi
                
                errors.append({'goal_num': goal_count, 'distance': d_err, 'yaw': abs(yaw_err), 'duration': trajectory.duration})
                
                print(f"  -> Done. Pos Err: {d_err:.3f}m, Yaw Err: {math.degrees(abs(yaw_err)):.1f}°")
                state = "WAITING"

            # Apply Command
            term = env.command_manager.get_term(cmd_name)
            term.command[:, 0] = vx_cmd
            term.command[:, 1] = vy_cmd
            term.command[:, 2] = omega_cmd

            # Step Policy (Handle JIT input format)
            input_obs = obs["policy"] if isinstance(obs, dict) else obs
            actions = policy(input_obs)
            obs, _, _, _, _ = env.step(actions)
            
            total_time += dt

        # Wrap up
        if SAVE_RESULTS:
            save_results_to_csv(errors, OUTPUT_DIR)
        
    simulation_app.close()

if __name__ == "__main__":
    main()