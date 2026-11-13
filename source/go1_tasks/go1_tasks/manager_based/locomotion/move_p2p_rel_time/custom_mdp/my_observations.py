import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, wrap_to_pi, yaw_quat
from .my_utils import _ensure_relpose_buffers
import math

def rel_pose_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    _ensure_relpose_buffers(env)
    PAUSE_STEPS = 50 
    steps_since_reset = env.common_step_counter - env.command_start_step
    is_active = steps_since_reset >= PAUSE_STEPS
    mask = is_active.unsqueeze(-1).float()
    return env.rel_pose_cmd * mask

def rel_pose_position_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    _ensure_relpose_buffers(env)
    asset = env.scene[asset_cfg.name]
    current_pos_w = asset.data.root_pos_w             
    error_w = env.target_root_pos_w - current_pos_w   
    root_quat = asset.data.root_quat_w                
    return quat_apply_inverse(root_quat, error_w)

def rel_pose_yaw_error_sincos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    _ensure_relpose_buffers(env)
    asset = env.scene[asset_cfg.name]
    root_state = asset.data.root_state_w              
    root_quat = root_state[:, 3:7]
    _, _, yaw = euler_xyz_from_quat(root_quat)        
    yaw_err = wrap_to_pi(env.target_root_yaw - yaw)   
    return torch.stack((torch.sin(yaw_err), torch.cos(yaw_err)), dim=-1)


def time_remaining_normalized(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Provides the policy with a normalized countdown [1.0 to 0.0] based on the dynamic target time."""
    
    # NEW: Fetch the command and extract the target time from the 5D tensor (Index 4)
    command = env.command_manager.get_command(command_name)
    target_time = command[:, 4]

    # env.episode_length_buf tracks how many steps have occurred
    t_elapsed = env.episode_length_buf * env.step_dt
    
    # Calculate the normalized remaining time (Element-wise math for all environments)
    t_rem = torch.clamp(target_time - t_elapsed, min=0.0) / target_time
    
    # Must return a 2D tensor [num_envs, 1] for observations
    return t_rem.unsqueeze(1)

def pose_command_with_drift(
    env: ManagerBasedRLEnv, 
    command_name: str,
    # drift_std_xy: float = 0.01,
    # drift_std_yaw: float = 0.009
    # # for k=2
    # drift_std_xy: float = 0.02,
    # drift_std_yaw: float = 0.018
    # for k=3
    drift_std_xy: float = 0.03,
    drift_std_yaw: float = 0.027
) -> torch.Tensor:
    """Fetches the pose command and applies an accumulating random-walk drift."""
    # 1. Get the clean 5D command: [dx, dy, dz, dyaw, time]
    command = env.command_manager.get_command(command_name)
    
    # 2. Initialize a persistent drift buffer on the environment (runs once)
    if not hasattr(env, "_obs_drift_bias"):
        env._obs_drift_bias = torch.zeros((env.num_envs, 3), device=env.device)
        # Precompute scales: std * sqrt(dt)
        scales = [drift_std_xy, drift_std_xy, drift_std_yaw]
        env._obs_drift_scales = torch.tensor(scales, device=env.device) * math.sqrt(env.step_dt)
        
    # 3. Reset drift to zero for environments that just restarted
    resets = (env.episode_length_buf == 0)
    env._obs_drift_bias[resets] = 0.0
    
    # 4. Accumulate the random walk
    step_noise = torch.randn_like(env._obs_drift_bias) * env._obs_drift_scales
    env._obs_drift_bias += step_noise
    
    # 5. Apply drift to the X, Y, and Yaw channels of the command
    noisy_command = command.clone()
    noisy_command[:, 0:2] += env._obs_drift_bias[:, 0:2] # Corrupt dx, dy
    noisy_command[:, 3] += env._obs_drift_bias[:, 2]     # Corrupt dyaw
    
    return noisy_command