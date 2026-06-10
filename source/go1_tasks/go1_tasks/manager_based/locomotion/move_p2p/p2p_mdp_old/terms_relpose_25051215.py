# File: isaaclab_tasks/manager_based/locomotion/velocity/mdp/terms_relpose.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import math  
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse, wrap_to_pi

# -------------------------------------------------------------------------
# Simple global step counter for debug printing
# -------------------------------------------------------------------------
_debug_relpose_counter = 0


def _ensure_relpose_buffers(env: ManagerBasedRLEnv):
    """Allocate buffers on the env the first time we use them."""
    device = env.device
    num_envs = env.num_envs

    # 1. Ensure Command Buffers exist
    if not hasattr(env, "rel_pose_cmd"):
        env.rel_pose_cmd = torch.zeros(num_envs, 4, device=device)
        env.target_root_pos_w = torch.zeros(num_envs, 3, device=device)
        env.target_root_yaw = torch.zeros(num_envs, device=device)

    # 2. Ensure Nominal Joint Positions exist (FORCE HARDCODED VALUES)
    if not hasattr(env, "nominal_joint_pos"):
        # Go1 Standard Standing Pose
        # [FL_hip, FL_thigh, FL_calf,  FR_hip, FR_thigh, FR_calf, ...]
        base_pose = [0.1, 0.8, -1.5,   -0.1, 0.8, -1.5,   0.1, 0.8, -1.5,   -0.1, 0.8, -1.5]
        q_des = torch.tensor(base_pose, device=device)
        env.nominal_joint_pos = q_des.unsqueeze(0).repeat(num_envs, 1)

    # 3. Ensure Feet Indices exist (Stored in ENV, not asset.data)
    if not hasattr(env, "feet_indices"):
        try:
            asset = env.scene["robot"]
            # Find all bodies ending in "foot". Returns (indices, names)
            # Standard Go1: FL_foot, FR_foot, RL_foot, RR_foot
            feet_indices, _ = asset.find_bodies(".*foot")
            env.feet_indices = feet_indices
            print(f"[INFO] Found feet indices: {feet_indices}")
        except Exception as e:
            print(f"[WARNING] Could not find feet indices automatically: {e}")
            env.feet_indices = None


# -----------------------------------------------------------------------------
# Event: sample a new Δpose command and compute absolute target pose
# -----------------------------------------------------------------------------

def reset_rel_pose_command(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    ranges: dict,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """On reset, sample a relative pose (dx, dy, dz, dyaw) and store target pose."""

    # print only during play 
    if env.num_envs < 100: 
            print("[DEBUG -- RESET]")    # ---------------------
    
    _ensure_relpose_buffers(env)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    asset = env.scene[asset_cfg.name]

    # 1) Sample Δpose
    dx = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dx"])
    dy = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dy"])
    dz = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dz"])
    dyaw = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dyaw"])

    cmd = torch.stack((dx, dy, dz, dyaw), dim=-1)    # (N, 4)
    env.rel_pose_cmd[env_ids] = cmd

    # 2) Compute absolute target pose in world frame
    root_state = asset.data.root_state_w[env_ids]    # (N, 13)
    root_pos = root_state[:, 0:3]                    # (N, 3)
    root_quat = root_state[:, 3:7]                   # (N, 4)
    _, _, root_yaw = euler_xyz_from_quat(root_quat)  # each (N,)

    # Rotate the local [dx, dy, dz] command into the world frame
    delta_pos_w = quat_apply(root_quat, cmd[:, :3]) 

    target_pos = root_pos + delta_pos_w             
    target_yaw = wrap_to_pi(root_yaw + cmd[:, 3])   

    env.target_root_pos_w[env_ids] = target_pos
    env.target_root_yaw[env_ids] = target_yaw


# -----------------------------------------------------------------------------
# Observation terms
# -----------------------------------------------------------------------------

def rel_pose_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the sampled relative command [dx, dy, dz, dyaw]."""
    _ensure_relpose_buffers(env)

    # return env.rel_pose_cmd

    wait_time_s = 1.0  # Pause duration in seconds
    
    # Calculate how many steps to wait: steps = seconds / dt
    # env.step_dt is typically 0.02s (50Hz)
    wait_steps = wait_time_s / env.step_dt
    
    # Check if we have passed the wait period
    # env.episode_length_buf counts steps since the last reset
    is_active = env.episode_length_buf >= wait_steps
    
    # Create a mask (1.0 if active, 0.0 if waiting)
    # Unsqueeze to match shape (num_envs, 1) so it broadcasts to (num_envs, 4)
    mask = is_active.unsqueeze(-1).float()
    
    # Apply mask: If waiting, command becomes [0, 0, 0, 0]
    return env.rel_pose_cmd * mask


def rel_pose_position_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position error vector: target_pos - current_pos (Rotated to Body Frame)."""
    _ensure_relpose_buffers(env)
    asset = env.scene[asset_cfg.name]
    
    # World frame error
    current_pos_w = asset.data.root_pos_w             
    error_w = env.target_root_pos_w - current_pos_w   
    
    # Get Robot Orientation
    root_quat = asset.data.root_quat_w                
    
    # Body frame error
    error_b = quat_apply_inverse(root_quat, error_w)
    
    return error_b


def rel_pose_yaw_error_sincos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw error encoded as [sin(e), cos(e)]."""
    _ensure_relpose_buffers(env)
    asset = env.scene[asset_cfg.name]

    root_state = asset.data.root_state_w              
    root_quat = root_state[:, 3:7]
    _, _, yaw = euler_xyz_from_quat(root_quat)        

    yaw_err = wrap_to_pi(env.target_root_yaw - yaw)   

    return torch.stack((torch.sin(yaw_err), torch.cos(yaw_err)), dim=-1)


# -----------------------------------------------------------------------------
# Reward terms
# -----------------------------------------------------------------------------

def rel_pose_yaw_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sigma=0.5):
    """Reward = exp( -yaw_err² )."""
    _ensure_relpose_buffers(env)
    asset = env.scene[asset_cfg.name]

    root_state = asset.data.root_state_w
    root_quat = root_state[:, 3:7]
    _, _, yaw = euler_xyz_from_quat(root_quat)

    yaw_err = wrap_to_pi(env.target_root_yaw - yaw)
    
    d2 = yaw_err * yaw_err
    return torch.exp(-d2 / (2 * sigma * sigma))


def rel_pose_position_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.5, 
) -> torch.Tensor:
    """Exponential reward based on position error norm."""
    global _debug_relpose_counter
    _ensure_relpose_buffers(env)

    asset = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w                       
    err = env.target_root_pos_w - pos                 
    d2 = torch.sum(err * err, dim=-1)                 

    # # DEBUG
    # _debug_relpose_counter += 1
    # if _debug_relpose_counter % 200 == 0:
    #     i = 0
    #     dist = float(torch.sqrt(d2[i]).detach().cpu().item())
    #     print(f"\n[RELPOSE DEBUG] Env 0 Dist Error: {dist:.3f} m")
    # ================= DEBUG PRINT BLOCK (Play Mode Only) =================
    _debug_relpose_counter += 1
    
    # Check if this is "Play" mode (heuristic: small number of environments)
    if env.num_envs < 100 and _debug_relpose_counter % 50 == 0:
        i = 0 # Monitor Env 0
        
        # 1. Distance Error
        dist = float(torch.sqrt(d2[i]).detach().cpu().item())
        
        # 2. Yaw Error (Degrees)
        # Get current yaw
        root_quat = asset.data.root_quat_w[i].unsqueeze(0)
        _, _, current_yaw = euler_xyz_from_quat(root_quat)
        target_yaw = env.target_root_yaw[i]
        
        # Calc error and wrap to -pi, pi
        yaw_err_rad = wrap_to_pi(target_yaw - current_yaw).item()
        yaw_err_deg = math.degrees(yaw_err_rad)
        
        print(f"[RELPOSE] Dist Err: {dist:.3f} m | Yaw Err: {yaw_err_deg:.1f} deg")
    # ======================================================================

    return torch.exp(-d2 / (2.0 * sigma * sigma))


def nominal_joint_position_l2_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Returns ||q - q_nominal||^2 (Positive value)."""
    _ensure_relpose_buffers(env)
    asset = env.scene[asset_cfg.name]
    
    q = asset.data.joint_pos[:] # (N, 12)
    q_nominal = env.nominal_joint_pos 
    
    deviation = q - q_nominal
    d2 = torch.sum(deviation * deviation, dim=-1) # (N,)
    
    return d2


def track_trot_symmetry_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize unsynchronized movement of diagonal pairs (Trot Symmetry).
    Pairs: (FL, RR) and (FR, RL).
    """
    _ensure_relpose_buffers(env)
    
    # Check if we successfully found feet indices
    if env.feet_indices is None:
        return torch.zeros(env.num_envs, device=env.device)

    asset = env.scene[asset_cfg.name]
    
    # FIX: Use env.feet_indices, NOT asset.data.feet_indices
    # Shape: (num_envs, 4, 3) -> Z is index 2
    feet_vel = asset.data.body_lin_vel_w[:, env.feet_indices, 2] 

    # Diagonal Pair 1: FL (0) and RR (3)
    diff1 = torch.abs(feet_vel[:, 0] - feet_vel[:, 3])
    
    # Diagonal Pair 2: FR (1) and RL (2)
    diff2 = torch.abs(feet_vel[:, 1] - feet_vel[:, 2])
    
    # Sum the errors. We want this to be 0.
    return diff1 + diff2