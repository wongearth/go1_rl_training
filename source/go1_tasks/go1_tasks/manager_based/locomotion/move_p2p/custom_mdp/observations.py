import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, wrap_to_pi
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