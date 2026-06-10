# File: isaaclab_tasks/manager_based/locomotion/velocity/mdp/terms_relpose.py
# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply, quat_apply_inverse, wrap_to_pi
# Change it to import your local folder's mdp functions


# -------------------------------------------------------------------------
# Global Debug Counter
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
        env.command_start_step = torch.zeros(num_envs, device=device) # For Pausing

    # Ensure Nominal Joint Positions exist (Safe Go1 Pose)
    if not hasattr(env, "nominal_joint_pos"):
        # [FL_hip, FL_thigh, FL_calf,  FR_hip, FR_thigh, FR_calf, ...]
        base_pose = [0.1, 0.8, -1.5,   -0.1, 0.8, -1.5,   0.1, 0.8, -1.5,   -0.1, 0.8, -1.5]
        q_des = torch.tensor(base_pose, device=device)
        env.nominal_joint_pos = q_des.unsqueeze(0).repeat(num_envs, 1)

    # Ensure Feet Indices exist
    if not hasattr(env, "feet_indices"):
        try:
            asset = env.scene["robot"]
            feet_indices, _ = asset.find_bodies(".*foot")
            env.feet_indices = feet_indices
        except Exception as e:
            env.feet_indices = None

# -----------------------------------------------------------------------------
# MASTER RESET FUNCTION (The Fix)
# -----------------------------------------------------------------------------

def reset_playback_episode(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    ranges: dict,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """
    ATOMIC RESET: Resets Root, Joints, and Command simultaneously.
    Anchors everything to (0,0,0) world origin.
    """
    _ensure_relpose_buffers(env)
    
    if env.num_envs < 100:
        print(f"[DEBUG -- RESET] Atomic Reset triggered at step {env.common_step_counter}")

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    
    asset = env.scene[asset_cfg.name]
    device = env.device

    # 1. RESET ROOT to Origin (0,0, 0.35m height)
    # -------------------------------------------------------
    # Default Go1 Standing Height ~0.35m
    # root_pos = torch.zeros(len(env_ids), 3, device=device)
    # root_pos[:, 2] = 0.35 

    # Grab the specific spawn origins for the environments being reset
    origins = env.scene.env_origins[env_ids]
    
    # Start with zeros, then add the origins
    root_pos = torch.zeros(len(env_ids), 3, device=device)
    root_pos[:, :2] = origins[:, :2] # Add X and Y grid coordinates
    root_pos[:, 2] = origins[:, 2] + 0.35 # Add Z coordinate + 0.35m standing height

    root_quat = torch.zeros(len(env_ids), 4, device=device)
    root_quat[:, 0] = 1.0 # Identity (w=1, x=0, y=0, z=0)
    
    root_vel = torch.zeros(len(env_ids), 6, device=device) # Linear + Angular velocity = 0

    # Write directly to simulation (Atomic Teleport)
    asset.write_root_pose_to_sim(torch.cat([root_pos, root_quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)

    # 2. RESET JOINTS to Safe Standing Pose
    # -------------------------------------------------------
    q_stand = env.nominal_joint_pos[env_ids]
    dq_stand = torch.zeros_like(q_stand)
    
    asset.write_joint_state_to_sim(q_stand, dq_stand, env_ids=env_ids)

    # 3. RESET COMMAND (Relative to Origin 0,0,0)
    # -------------------------------------------------------
    # Record start step for the 1-second pause logic
    env.command_start_step[env_ids] = env.common_step_counter

    dx = torch.empty(len(env_ids), device=device).uniform_(*ranges["dx"])
    dy = torch.empty(len(env_ids), device=device).uniform_(*ranges["dy"])
    dz = torch.empty(len(env_ids), device=device).uniform_(*ranges["dz"])
    dyaw = torch.empty(len(env_ids), device=device).uniform_(*ranges["dyaw"])
    
    cmd = torch.stack((dx, dy, dz, dyaw), dim=-1)
    env.rel_pose_cmd[env_ids] = cmd

    # Since we just teleported to (0,0,0) with Identity orientation:
    # Target = The robot's SPECIFIC spawn grid center + Command
    env.target_root_pos_w[env_ids] = root_pos + cmd[:, :3]
    env.target_root_yaw[env_ids] = cmd[:, 3]


# -----------------------------------------------------------------------------
# Observation & Reward Terms
# -----------------------------------------------------------------------------

def rel_pose_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return command with 1-second PAUSE logic."""
    _ensure_relpose_buffers(env)
    
    # Pause for 1.0 second (50 steps at 50Hz)
    PAUSE_STEPS = 50 
    
    steps_since_reset = env.common_step_counter - env.command_start_step
    
    # Create mask: 0 if waiting, 1 if active
    is_active = steps_since_reset >= PAUSE_STEPS
    mask = is_active.unsqueeze(-1).float()
    
    return env.rel_pose_cmd * mask

def rel_pose_position_error(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    _ensure_relpose_buffers(env)
    asset = env.scene[asset_cfg.name]
    
    current_pos_w = asset.data.root_pos_w             
    error_w = env.target_root_pos_w - current_pos_w   
    root_quat = asset.data.root_quat_w                
    error_b = quat_apply_inverse(root_quat, error_w)
    return error_b

def rel_pose_yaw_error_sincos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    _ensure_relpose_buffers(env)
    asset = env.scene[asset_cfg.name]
    root_state = asset.data.root_state_w              
    root_quat = root_state[:, 3:7]
    _, _, yaw = euler_xyz_from_quat(root_quat)        
    yaw_err = wrap_to_pi(env.target_root_yaw - yaw)   
    return torch.stack((torch.sin(yaw_err), torch.cos(yaw_err)), dim=-1)

def rel_pose_position_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sigma: float = 0.5) -> torch.Tensor:
    global _debug_relpose_counter
    _ensure_relpose_buffers(env)

    asset = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w                       
    err = env.target_root_pos_w - pos                 
    d2 = torch.sum(err * err, dim=-1)                 

    # DEBUG PRINT BLOCK
    _debug_relpose_counter += 1
    if env.num_envs < 100 and _debug_relpose_counter % 50 == 0:
        i = 0 
        cmd = env.rel_pose_cmd[i]
        cmd_x, cmd_y = cmd[0].item(), cmd[1].item()
        cmd_yaw_deg = math.degrees(cmd[3].item())
        
        # Current Position
        cur_x = pos[i, 0].item()
        cur_y = pos[i, 1].item()
        
        root_quat = asset.data.root_quat_w[i].unsqueeze(0)
        _, _, cur_yaw_rad = euler_xyz_from_quat(root_quat)
        cur_yaw_deg = math.degrees(cur_yaw_rad.item())
        
        dist_err = float(torch.sqrt(d2[i]).item())
        
        print("-" * 60)
        print(f"[DEBUG] CMD (Target) : dx={cmd_x:.2f}m, dy={cmd_y:.2f}m, dyaw={cmd_yaw_deg:.1f} deg")
        print(f"[DEBUG] CUR (Actual) :  x={cur_x:.2f}m,  y={cur_y:.2f}m,  yaw={cur_yaw_deg:.1f} deg")
        print(f"[DEBUG] ERR (Delta)  : dist={dist_err:.3f}m")
        print("-" * 60)

    return torch.exp(-d2 / (2.0 * sigma * sigma))

def rel_pose_yaw_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sigma=0.5):
    _ensure_relpose_buffers(env)
    asset = env.scene[asset_cfg.name]
    root_state = asset.data.root_state_w
    root_quat = root_state[:, 3:7]
    _, _, yaw = euler_xyz_from_quat(root_quat)
    yaw_err = wrap_to_pi(env.target_root_yaw - yaw)
    d2 = yaw_err * yaw_err
    return torch.exp(-d2 / (2 * sigma * sigma))

def nominal_joint_position_l2_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    _ensure_relpose_buffers(env)
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:] 
    q_nominal = env.nominal_joint_pos 
    deviation = q - q_nominal
    d2 = torch.sum(deviation * deviation, dim=-1) 
    return d2

def track_trot_symmetry_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    _ensure_relpose_buffers(env)
    if env.feet_indices is None: return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    feet_vel = asset.data.body_lin_vel_w[:, env.feet_indices, 2] 
    diff1 = torch.abs(feet_vel[:, 0] - feet_vel[:, 3])
    diff2 = torch.abs(feet_vel[:, 1] - feet_vel[:, 2])
    return diff1 + diff2

# Need to expose the reset_rel_pose_command for training config compatibility
# reset_rel_pose_command = reset_playback_episode

def reset_rel_pose_command(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ranges: dict,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """
    Standard training reset: ONLY resets the command and target math.
    Leaves the physical robot spawning to the default `reset_base` event!
    """
    _ensure_relpose_buffers(env)
    
    # 1. Generate new random commands
    dx = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dx"])
    dy = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dy"])
    dz = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dz"])
    dyaw = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dyaw"])
    
    cmd = torch.stack((dx, dy, dz, dyaw), dim=-1)
    env.rel_pose_cmd[env_ids] = cmd
    
    # 2. Set the absolute targets based on their specific grid origins
    # (Since reset_base just placed them in these grid squares)
    origins = env.scene.env_origins[env_ids]
    
    target_pos = origins.clone()
    target_pos[:, 0] += cmd[:, 0] # Add dx to grid center
    target_pos[:, 1] += cmd[:, 1] # Add dy to grid center
    target_pos[:, 2] += cmd[:, 2] # Add dz to grid center
    
    env.target_root_pos_w[env_ids] = target_pos
    env.target_root_yaw[env_ids] = cmd[:, 3]