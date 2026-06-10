import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, wrap_to_pi, yaw_quat
from .utils import _ensure_relpose_buffers

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


def local_pose_command(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Transforms the global target pose into the robot's local base frame [local_x, local_y, local_z, local_yaw, time]."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # 1. Calculate Global Position Error (Target - Current Position)
    global_target_pos = torch.zeros_like(asset.data.root_pos_w)
    global_target_pos[:, 0] = command[:, 0]
    global_target_pos[:, 1] = command[:, 1]
    global_target_pos[:, 2] = command[:, 2]
    
    pos_error_w = global_target_pos - asset.data.root_pos_w
    
    # 2. Transform Position Error into the Robot's First-Person View (Base Frame)
    yaw_q = yaw_quat(asset.data.root_quat_w)
    local_pos_error = quat_apply_inverse(yaw_q, pos_error_w)
    
    # 3. Calculate Local Yaw Error (Target Yaw - Current Yaw)
    _, _, robot_yaw = euler_xyz_from_quat(yaw_q)
    local_yaw_error = command[:, 3] - robot_yaw
    
    # Wrap yaw error to keep it cleanly between -pi and pi
    local_yaw_error = torch.atan2(torch.sin(local_yaw_error), torch.cos(local_yaw_error))
    
    # 4. Extract Target Time
    target_time = command[:, 4].unsqueeze(1)
    
    # 5. Return the 5D Local Tensor
    return torch.cat([local_pos_error[:, :3], local_yaw_error.unsqueeze(1), target_time], dim=-1)

def local_pose_error(
    env, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Returns local error [dx, dy, dz, dyaw] in the robot's base frame."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # Position Error
    global_target_pos = command[:, :3]
    pos_error_w = global_target_pos - asset.data.root_pos_w
    yaw_q = yaw_quat(asset.data.root_quat_w)
    local_pos_error = quat_apply_inverse(yaw_q, pos_error_w)
    
    # Yaw Error
    _, _, robot_yaw = euler_xyz_from_quat(yaw_q)
    local_yaw_error = command[:, 3] - robot_yaw
    local_yaw_error = torch.atan2(torch.sin(local_yaw_error), torch.cos(local_yaw_error))
    
    # Return 4D tensor (we usually just pass dx, dy, and dyaw to avoid Z drift issues)
    return torch.cat([local_pos_error[:, :2], local_yaw_error.unsqueeze(1)], dim=-1)
