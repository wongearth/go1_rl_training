import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from .utils import _ensure_relpose_buffers

def reset_playback_episode(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, ranges: dict, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    _ensure_relpose_buffers(env)
    
    if env.num_envs < 100:
        print(f"[DEBUG -- RESET] Atomic Reset triggered at step {env.common_step_counter}")

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    
    asset = env.scene[asset_cfg.name]
    device = env.device
    origins = env.scene.env_origins[env_ids]
    
    root_pos = torch.zeros(len(env_ids), 3, device=device)
    root_pos[:, :2] = origins[:, :2] 
    root_pos[:, 2] = origins[:, 2] + 0.35 

    root_quat = torch.zeros(len(env_ids), 4, device=device)
    root_quat[:, 0] = 1.0 
    root_vel = torch.zeros(len(env_ids), 6, device=device) 

    asset.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)

    q_stand = env.nominal_joint_pos[env_ids]
    dq_stand = torch.zeros_like(q_stand)
    asset.write_joint_state_to_sim(q_stand, dq_stand, env_ids=env_ids)

    env.command_start_step[env_ids] = env.common_step_counter
    dx = torch.empty(len(env_ids), device=device).uniform_(*ranges["dx"])
    dy = torch.empty(len(env_ids), device=device).uniform_(*ranges["dy"])
    dz = torch.empty(len(env_ids), device=device).uniform_(*ranges["dz"])
    dyaw = torch.empty(len(env_ids), device=device).uniform_(*ranges["dyaw"])
    
    cmd = torch.stack((dx, dy, dz, dyaw), dim=-1)
    env.rel_pose_cmd[env_ids] = cmd
    env.target_root_pos_w[env_ids] = root_pos + cmd[:, :3]
    env.target_root_yaw[env_ids] = cmd[:, 3]

def reset_rel_pose_command(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, ranges: dict, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> None:
    _ensure_relpose_buffers(env)
    
    dx = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dx"])
    dy = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dy"])
    dz = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dz"])
    dyaw = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dyaw"])
    
    cmd = torch.stack((dx, dy, dz, dyaw), dim=-1)
    env.rel_pose_cmd[env_ids] = cmd
    
    origins = env.scene.env_origins[env_ids]
    target_pos = origins.clone()
    target_pos[:, 0] += cmd[:, 0]
    target_pos[:, 1] += cmd[:, 1]
    target_pos[:, 2] += cmd[:, 2]
    
    env.target_root_pos_w[env_ids] = target_pos
    env.target_root_yaw[env_ids] = cmd[:, 3]

def force_reset_robot_to_origin(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
):
    """Forces the robot to absolutely teleport to (0,0, 0.38) in world coordinates."""
    asset = env.scene[asset_cfg.name]
    
    # Clone default state to retain correct joint positions
    new_root_state = asset.data.default_root_state[env_ids].clone()
    
    # 1. Force Position to X=0, Y=0, Z=0.38 (Go1 standing height)
    new_root_state[:, 0:2] = 0.0
    new_root_state[:, 2] = 0.38
    
    # 2. Force Orientation to Identity (W=1, X=0, Y=0, Z=0) -> Facing perfectly forward
    new_root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)
    
    # 3. Force all velocities (linear and angular) to zero
    new_root_state[:, 7:13] = 0.0
    
    # Inject directly into the physics engine
    asset.write_root_state_to_sim(new_root_state, env_ids)