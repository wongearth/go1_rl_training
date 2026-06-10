import os
import argparse
import math
import torch
import numpy as np
from isaaclab.app import AppLauncher

# --- CONFIG ---
CHECKPOINT_PATH = "logs/rsl_rl/unitree_go1_flat/model_go1_vel.pt"
TASK_NAME = "Isaac-Velocity-Flat-Unitree-Go1-v0"
TEST_DURATION = 5.0  # seconds per test
# --------------

parser = argparse.ArgumentParser(description="Yaw Tracking Diagnostic")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = False
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

    print("\n" + "="*70)
    print(" "*15 + "YAW TRACKING DIAGNOSTIC TEST")
    print("="*70 + "\n")
    
    # Test cases: [vx, vy, omega_z]
    test_cases = [
        ("Forward only", [0.5, 0.0, 0.0]),
        ("Turn in place (slow)", [0.0, 0.0, 0.5]),
        ("Turn in place (medium)", [0.0, 0.0, 1.0]),
        ("Turn in place (fast)", [0.0, 0.0, 1.5]),
        ("Forward + Turn (slow)", [0.5, 0.0, 0.5]),
        ("Forward + Turn (medium)", [0.5, 0.0, 1.0]),
        ("Strafe only", [0.0, 0.3, 0.0]),
        ("Strafe + Turn", [0.0, 0.3, 0.5]),
    ]
    
    results = []
    
    with torch.inference_mode():
        for test_name, cmd_vel in test_cases:
            # Reset robot
            env.scene["robot"].write_root_pose_to_sim(
                torch.tensor([[0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]], device=device)
            )
            obs, _ = env.reset()
            
            # Initial pose
            start_pose = get_robot_pose(env)
            start_yaw = start_pose[2]
            
            # Command velocities
            vx_cmd, vy_cmd, omega_cmd = cmd_vel
            
            print(f"\n{'='*70}")
            print(f"Test: {test_name}")
            print(f"Command: vx={vx_cmd:.2f} m/s, vy={vy_cmd:.2f} m/s, omega={omega_cmd:.2f} rad/s")
            print(f"{'='*70}")
            
            # Track data
            time_elapsed = 0.0
            vel_data = []
            
            while time_elapsed < TEST_DURATION:
                # Set command
                term = env.command_manager.get_term(cmd_name)
                term.command[:, 0] = vx_cmd
                term.command[:, 1] = vy_cmd
                term.command[:, 2] = omega_cmd
                
                # Get actual velocities
                actual_lin_vel = env.scene["robot"].data.root_lin_vel_w[0]
                actual_ang_vel = env.scene["robot"].data.root_ang_vel_w[0]
                
                current_pose = get_robot_pose(env)
                current_yaw = current_pose[2]
                
                # Transform to robot frame
                cos_yaw = math.cos(current_yaw)
                sin_yaw = math.sin(current_yaw)
                actual_vx = actual_lin_vel[0].item() * cos_yaw + actual_lin_vel[1].item() * sin_yaw
                actual_vy = -actual_lin_vel[0].item() * sin_yaw + actual_lin_vel[1].item() * cos_yaw
                actual_omega = actual_ang_vel[2].item()
                
                vel_data.append({
                    'vx': actual_vx,
                    'vy': actual_vy,
                    'omega': actual_omega
                })
                
                # Step
                actions = policy(obs)
                obs, _, _, _, _ = env.step(actions)
                
                time_elapsed += dt
                
                # Print progress
                if int(time_elapsed / dt) % 25 == 0:
                    print(f"  t={time_elapsed:.1f}s | Actual: vx={actual_vx:+.3f}, vy={actual_vy:+.3f}, omega={actual_omega:+.3f} rad/s", end="\r")
            
            # Final pose
            end_pose = get_robot_pose(env)
            end_yaw = end_pose[2]
            
            # Calculate statistics (skip first 1 second for settling)
            steady_state_data = vel_data[int(1.0/dt):]
            
            avg_vx = np.mean([d['vx'] for d in steady_state_data])
            avg_vy = np.mean([d['vy'] for d in steady_state_data])
            avg_omega = np.mean([d['omega'] for d in steady_state_data])
            
            std_vx = np.std([d['vx'] for d in steady_state_data])
            std_vy = np.std([d['vy'] for d in steady_state_data])
            std_omega = np.std([d['omega'] for d in steady_state_data])
            
            vx_error = abs(avg_vx - vx_cmd)
            vy_error = abs(avg_vy - vy_cmd)
            omega_error = abs(avg_omega - omega_cmd)
            
            # Expected vs actual yaw change
            expected_yaw_change = omega_cmd * TEST_DURATION
            actual_yaw_change = end_yaw - start_yaw
            # Normalize to [-pi, pi]
            actual_yaw_change = (actual_yaw_change + math.pi) % (2 * math.pi) - math.pi
            yaw_change_error = abs(expected_yaw_change - actual_yaw_change)
            
            results.append({
                'name': test_name,
                'vx_cmd': vx_cmd,
                'vy_cmd': vy_cmd,
                'omega_cmd': omega_cmd,
                'avg_vx': avg_vx,
                'avg_vy': avg_vy,
                'avg_omega': avg_omega,
                'vx_error': vx_error,
                'vy_error': vy_error,
                'omega_error': omega_error,
                'omega_error_pct': (omega_error / abs(omega_cmd) * 100) if omega_cmd != 0 else 0,
                'expected_yaw_deg': math.degrees(expected_yaw_change),
                'actual_yaw_deg': math.degrees(actual_yaw_change),
                'yaw_error_deg': math.degrees(yaw_change_error)
            })
            
            print(f"\n  Steady-State Results:")
            print(f"    Commanded:  vx={vx_cmd:.3f}, vy={vy_cmd:.3f}, omega={omega_cmd:.3f} rad/s")
            print(f"    Achieved:   vx={avg_vx:.3f}, vy={avg_vy:.3f}, omega={avg_omega:.3f} rad/s")
            print(f"    Error:      vx={vx_error:.3f}, vy={vy_error:.3f}, omega={omega_error:.3f} rad/s ({(omega_error/abs(omega_cmd)*100) if omega_cmd != 0 else 0:.1f}%)")
            print(f"    Yaw Change: Expected={math.degrees(expected_yaw_change):.1f}°, Actual={math.degrees(actual_yaw_change):.1f}°, Error={math.degrees(yaw_change_error):.1f}°")
    
    # Print summary table
    print("\n\n" + "="*70)
    print(" "*20 + "SUMMARY RESULTS")
    print("="*70)
    print(f"\n{'Test':<30} {'Omega Cmd':>10} {'Omega Avg':>10} {'Error':>10} {'Yaw Err':>10}")
    print(f"{'Name':<30} {'(rad/s)':>10} {'(rad/s)':>10} {'(%)':>10} {'(deg)':>10}")
    print("-"*70)
    
    for r in results:
        print(f"{r['name']:<30} {r['omega_cmd']:>10.2f} {r['avg_omega']:>10.3f} {r['omega_error_pct']:>9.1f}% {r['yaw_error_deg']:>9.1f}°")
    
    print("\n" + "="*70)
    print("DIAGNOSIS:")
    print("="*70)
    
    # Analyze results
    turn_in_place_tests = [r for r in results if "Turn in place" in r['name']]
    avg_omega_error_pct = np.mean([r['omega_error_pct'] for r in turn_in_place_tests])
    
    if avg_omega_error_pct > 50:
        print("❌ CRITICAL: Policy does NOT track yaw rate commands well (>50% error)")
        print("   → The policy was likely trained with insufficient yaw commands")
        print("   → Recommendation: Retrain with higher 'track_ang_vel_z_exp' weight")
    elif avg_omega_error_pct > 20:
        print("⚠️  WARNING: Policy has moderate yaw tracking error (20-50%)")
        print("   → The policy can turn but not accurately")
        print("   → Recommendation: Use closed-loop control or retrain")
    else:
        print("✅ OK: Policy tracks yaw commands reasonably well (<20% error)")
        print("   → High yaw errors in trajectory following are due to open-loop drift")
        print("   → Recommendation: Use closed-loop feedback control")
    
    print("\n" + "="*70 + "\n")
    
    simulation_app.close()


if __name__ == "__main__":
    main()