import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, wrap_to_pi, yaw_quat
from .my_utils import _ensure_relpose_buffers

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