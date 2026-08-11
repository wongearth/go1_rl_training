import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from .my_utils import _ensure_relpose_buffers

def force_reset_robot_to_origin(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Forces the robot to absolutely teleport to (0,0, 0.38) in world coordinates."""
    asset = env.scene[asset_cfg.name]
    
    # Clone default state to retain correct joint positions
    new_root_state = asset.data.default_root_state[env_ids].clone()
    
    # 1. Force Position to X=0, Y=0, Z=0.38 (Go1 standing height)
    new_root_state[:, 0:2] = 0.0
    new_root_state[:, 2] = 0.30
    
    # 2. Force Orientation to Identity (W=1, X=0, Y=0, Z=0) -> Facing perfectly forward
    new_root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    
    # 3. Force all velocities (linear and angular) to zero
    new_root_state[:, 7:13] = 0.0
    
    # Inject directly into the physics engine
    asset.write_root_state_to_sim(new_root_state, env_ids)