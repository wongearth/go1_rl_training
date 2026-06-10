# File: isaaclab_tasks/manager_based/locomotion/velocity/mdp/terms_relpose.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse, wrap_to_pi, quat_from_euler_xyz

# -------------------------------------------------------------------------
# Simple global step counter for debug printing
# -------------------------------------------------------------------------
_debug_relpose_counter = 0


def _ensure_relpose_buffers(env: ManagerBasedRLEnv):
    """Allocate buffers on the env the first time we use them."""
    device = env.device
    num_envs = env.num_envs

    if not hasattr(env, "rel_pose_cmd"):
        env.rel_pose_cmd = torch.zeros(num_envs, 4, device=device)
        env.target_root_pos_w = torch.zeros(num_envs, 3, device=device)
        env.target_root_yaw = torch.zeros(num_envs, device=device)
        # Store step count for robust pausing
        env.command_start_step = torch.zeros(num_envs, device=device)

    # 2. Ensure Nominal Joint Positions exist (Safe Go1 Pose)
    if not hasattr(env, "nominal_joint_pos"):
        # [FL_hip, FL_thigh, FL_calf,  FR_hip, FR_thigh, FR_calf, ...]
        base_pose = [0.1, 0.8, -1.5,   -0.1, 0.8, -1.5,   0.1, 0.8, -1.5,   -0.1, 0.8, -1.5]
        q_des = torch.tensor(base_pose, device=device)
        env.nominal_joint_pos = q_des.unsqueeze(0).repeat(num_envs, 1)

    # 3. Ensure Feet Indices exist (For Symmetry Reward)
    if not hasattr(env, "feet_indices"):
        try:
            asset = env.scene["robot"]
            feet_indices, _ = asset.find_bodies(".*foot")
            env.feet_indices = feet_indices
        except Exception as e:
            print(f"[WARNING] Could not find feet indices: {e}")
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
    """On reset, sample a relative pose and store target pose."""
    
    if env.num_envs < 100:
        print("[DEBUG -- RESET] New Command Generated")

    _ensure_relpose_buffers(env)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    # Record step for pause logic
    env.command_start_step[env_ids] = env.common_step_counter

    # 1) Sample Δpose
    dx = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dx"])
    dy = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dy"])
    dz = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dz"])
    dyaw = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dyaw"])

    cmd = torch.stack((dx, dy, dz, dyaw), dim=-1)    # (N, 4)
    env.rel_pose_cmd[env_ids] = cmd

    # ----------------------------------------------------------------------
    # FIX: Anchor Target to ENVIRONMENT ORIGIN (Spawn Point)
    # ----------------------------------------------------------------------
    # Previous bug: used asset.data.root_pos_w (Current Position) which caused
    # race conditions during reset.
    # Now we use env.scene.env_origins (The exact spawn point).
    
    spawn_pos = env.scene.env_origins[env_ids]       # (N, 3)
    
    # Assume spawn orientation is Identity (0,0,0) for Flat Ground
    # If you use complex terrain later, you might need to adjust this,
    # but for flat Play mode, this is robust.
    spawn_quat = torch.zeros(len(env_ids), 4, device=env.device)
    spawn_quat[:, 0] = 1.0 # (w, x, y, z) -> Identity
    
    # Rotate command into World Frame (based on Spawn Orientation)
    delta_pos_w = quat_apply(spawn_quat, cmd[:, :3]) 
    
    # Calculate Target
    target_pos = spawn_pos + delta_pos_w
    target_yaw = cmd[:, 3] # Relative to 0.0 spawn yaw

    env.target_root_pos_w[env_ids] = target_pos
    env.target_root_yaw[env_ids] = target_yaw


# -----------------------------------------------------------------------------
# Observation terms
# -----------------------------------------------------------------------------

def rel_pose_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """
    Return the sampled relative command [dx, dy, dz, dyaw].
    **Includes Robust 1-second PAUSE logic**.
    """
    _ensure_relpose_buffers(env)
    
    PLAY_CYCLE_S = 5.0
    PAUSE_TIME_S = 1.0
    
    if env.num_envs < 100:
        # PLAY MODE: Modulo Timer
        current_time_s = env.common_step_counter * env.step_dt
        time_in_cycle = current_time_s % PLAY_CYCLE_S
        should_pause = time_in_cycle < PAUSE_TIME_S
        mask_val = 0.0 if should_pause else 1.0
        mask = torch.full((env.num_envs, 1), mask_val, device=env.device)
        
    else:
        # TRAINING MODE: Episode Step
        current_episode_time = env.episode_length_buf * env.step_dt
        is_active = current_episode_time >= PAUSE_TIME_S
        mask = is_active.unsqueeze(-1).float()
    
    return env.rel_pose_cmd * mask


def rel_pose_position_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position error vector: target_pos - current_pos (Rotated to Body Frame)."""
    _ensure_relpose_buffers(env)
    asset = env.scene[asset_cfg.name]
    
    current_pos_w = asset.data.root_pos_w             
    error_w = env.target_root_pos_w - current_pos_w   
    root_quat = asset.data.root_quat_w                
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

    # ================= DEBUG PRINT BLOCK (Play Mode Only) =================
    _debug_relpose_counter += 1
    
    if env.num_envs < 100 and _debug_relpose_counter % 50 == 0:
        i = 0 
        
        # 1. Command
        cmd = env.rel_pose_cmd[i]
        cmd_x, cmd_y = cmd[0].item(), cmd[1].item()
        cmd_yaw_deg = math.degrees(cmd[3].item())
        
        # 2. Current Position relative to Spawn Point (Origin)
        # This clarifies if the robot is actually at (0,0) relative to its start
        spawn_pos = env.scene.env_origins[i]
        rel_x = pos[i, 0].item() - spawn_pos[0].item()
        rel_y = pos[i, 1].item() - spawn_pos[1].item()
        
        root_quat = asset.data.root_quat_w[i].unsqueeze(0)
        _, _, cur_yaw_rad = euler_xyz_from_quat(root_quat)
        cur_yaw_deg = math.degrees(cur_yaw_rad.item())
        
        dist_err = float(torch.sqrt(d2[i]).item())
        yaw_err_rad = wrap_to_pi(env.target_root_yaw[i] - cur_yaw_rad).item()
        yaw_err_deg = math.degrees(yaw_err_rad)

        print("-" * 60)
        print(f"[DEBUG] CMD (Target)  : dx={cmd_x:.2f}m, dy={cmd_y:.2f}m, dyaw={cmd_yaw_deg:.1f} deg")
        print(f"[DEBUG] CUR (RelStart):  x={rel_x:.2f}m,  y={rel_y:.2f}m,  yaw={cur_yaw_deg:.1f} deg")
        print(f"[DEBUG] ERR (Delta)   : dist={dist_err:.3f}m, yaw_err={yaw_err_deg:.1f} deg")
        print("-" * 60)
    # ======================================================================

    return torch.exp(-d2 / (2.0 * sigma * sigma))


def nominal_joint_position_l2_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Returns ||q - q_nominal||^2 (Positive value)."""
    _ensure_relpose_buffers(env)
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:] 
    q_nominal = env.nominal_joint_pos 
    deviation = q - q_nominal
    d2 = torch.sum(deviation * deviation, dim=-1) 
    return d2


def track_trot_symmetry_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize unsynchronized movement of diagonal pairs."""
    _ensure_relpose_buffers(env)
    
    if env.feet_indices is None:
        return torch.zeros(env.num_envs, device=env.device)

    asset = env.scene[asset_cfg.name]
    feet_vel = asset.data.body_lin_vel_w[:, env.feet_indices, 2] 

    diff1 = torch.abs(feet_vel[:, 0] - feet_vel[:, 3])
    diff2 = torch.abs(feet_vel[:, 1] - feet_vel[:, 2])
    
    return diff1 + diff2

# ... (Add this to the end of terms_relpose.py) ...

def reset_joints_to_standing(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """
    Force the robot joints to the hardcoded 'Safe Standing' pose on reset.
    This prevents the 'drop' or 'leg straight' glitch at spawn.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    # 1. Define the Safe Pose (Same as nominal)
    # [FL_hip, FL_thigh, FL_calf,  FR_hip, FR_thigh, FR_calf, ...]
    base_pose = [0.1, 0.8, -1.5,   -0.1, 0.8, -1.5,   0.1, 0.8, -1.5,   -0.1, 0.8, -1.5]
    
    # 2. Prepare tensors
    standing_pos = torch.tensor(base_pose, device=env.device)
    # Expand to shape (num_selected_envs, 12)
    joint_pos = standing_pos.unsqueeze(0).repeat(len(env_ids), 1)
    joint_vel = torch.zeros_like(joint_pos)

    # 3. Write directly to simulation
    asset = env.scene[asset_cfg.name]
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)