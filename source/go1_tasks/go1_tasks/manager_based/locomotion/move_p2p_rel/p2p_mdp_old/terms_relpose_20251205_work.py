# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab.utils.math import quat_from_euler_xyz


def _ensure_relpose_buffers(env: ManagerBasedRLEnv):
    """Allocate buffers on the env the first time we use them."""
    if not hasattr(env, "rel_pose_cmd"):
        device = env.device
        num_envs = env.num_envs
        # [dx, dy, dz, dyaw] per env
        env.rel_pose_cmd = torch.zeros(num_envs, 4, device=device)
        # absolute target pose in world frame
        env.target_root_pos_w = torch.zeros(num_envs, 3, device=device)
        env.target_root_yaw = torch.zeros(num_envs, device=device)


# --------------------------------------------------------------------------- #
# Event: sample a new relative pose command and compute absolute target
# --------------------------------------------------------------------------- #

def reset_rel_pose_command(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    ranges: dict,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Event term: on reset, sample Δpose and store target world pose."""

    _ensure_relpose_buffers(env)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    asset = env.scene[asset_cfg.name]

    # sample deltas in the given ranges
    dx = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dx"])
    dy = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dy"])
    dz = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dz"])
    dyaw = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dyaw"])

    cmd = torch.stack((dx, dy, dz, dyaw), dim=-1)     # (N, 4)
    env.rel_pose_cmd[env_ids] = cmd

    # # current base pose in world
    # root_pos = asset.data.root_pos_w[env_ids]         # (N, 3)
    # root_euler = euler_xyz_from_quat(asset.data.root_rot_w[env_ids])  # (N, 3)
    # root_yaw = root_euler[:, 2]


    root_state = asset.data.root_state_w[env_ids]        # (N, 13)
    root_pos = root_state[:, 0:3]                        # position
    root_quat = root_state[:, 3:7]                       # orientation as quat (x,y,z,w)
    _, _, root_yaw = euler_xyz_from_quat(root_quat)      # each is (N,)
    

    # absolute target pose
    target_pos = root_pos + cmd[:, :3]                # apply dx, dy, dz in world
    target_yaw = root_yaw + cmd[:, 3]                 # target yaw

    env.target_root_pos_w[env_ids] = target_pos
    env.target_root_yaw[env_ids] = target_yaw


# --------------------------------------------------------------------------- #
# Observation terms (NO env_ids here!)
# --------------------------------------------------------------------------- #

def rel_pose_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the sampled relative commands [dx, dy, dz, dyaw]."""
    _ensure_relpose_buffers(env)
    return env.rel_pose_cmd


def rel_pose_position_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position error to target, in world frame: target_pos - current_pos."""
    _ensure_relpose_buffers(env)
    asset = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w                         # (N, 3)
    err = env.target_root_pos_w - pos                   # (N, 3)
    return err


def rel_pose_yaw_error_sincos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Yaw error encoded as [sin(e), cos(e)]."""
    _ensure_relpose_buffers(env)
    asset = env.scene[asset_cfg.name]

    root_state = asset.data.root_state_w                # (N, 13)
    root_quat = root_state[:, 3:7]
    _, _, yaw = euler_xyz_from_quat(root_quat)

    yaw_err = env.target_root_yaw - yaw
    yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))

    return torch.stack((torch.sin(yaw_err), torch.cos(yaw_err)), dim=-1)


# --------------------------------------------------------------------------- #
# Reward terms (also NO env_ids; they return (num_envs,) directly)
# --------------------------------------------------------------------------- #

def rel_pose_position_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.1,
) -> torch.Tensor:
    """Exponential reward based on position error norm."""
    _ensure_relpose_buffers(env)
    asset = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w
    err = env.target_root_pos_w - pos          # (N, 3)
    d2 = torch.sum(err * err, dim=-1)          # (N,)
    return torch.exp(-d2 / (2.0 * sigma * sigma))


def rel_pose_yaw_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sigma=0.3):
    """Reward = exp( -yaw_err² )."""
    _ensure_relpose_buffers(env)
    asset = env.scene[asset_cfg.name]

    root_state = asset.data.root_state_w
    root_quat = root_state[:, 3:7]
    _, _, yaw = euler_xyz_from_quat(root_quat)

    yaw_err = env.target_root_yaw - yaw
    yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))

    d2 = yaw_err * yaw_err
    return torch.exp(-d2 / (2 * sigma * sigma))


def reached_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pos_tol: float = 0.1,
    yaw_tol: float = 0.2,
):
    """Termination condition: within tolerance in position and yaw."""
    _ensure_relpose_buffers(env)
    asset = env.scene[asset_cfg.name]

    # position error
    pos = asset.data.root_pos_w
    pos_err = env.target_root_pos_w - pos          # (N, 3)
    dist = torch.linalg.norm(pos_err, dim=-1)      # (N,)

    # yaw error
    root_state = asset.data.root_state_w
    root_quat = root_state[:, 3:7]
    _, _, yaw = euler_xyz_from_quat(root_quat)     # (N,)
    yaw_err = env.target_root_yaw - yaw
    yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))

    # boolean mask (N,)
    done = (dist < pos_tol) & (torch.abs(yaw_err) < yaw_tol)
    return done


def time_penalty(env: ManagerBasedRLEnv, value: float = 1.0):
    """Return a constant per-step value (used with negative weight as time penalty)."""
    return torch.ones(env.num_envs, device=env.device) * value


def stand_height_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.33,
    sigma: float = 0.05,
):
    """Reward for keeping base height near target (encourages not lying on ground)."""
    asset = env.scene[asset_cfg.name]
    # root_pos_w: (N, 3)
    h = asset.data.root_pos_w[:, 2]   # z / height
    diff2 = (h - target_height) * (h - target_height)
    return torch.exp(-diff2 / (2.0 * sigma * sigma))

# def update_relpose_marker(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     visualizer_name: str = "base_command",
# ):
#     """Update the debug arrow/disc to show the rel-pose target pose.

#     - Arrow origin: at target_root_pos_w (slightly above ground).
#     - Arrow direction: target_root_yaw (robot's desired heading).
#     """
#     _ensure_relpose_buffers(env)

#     # target position in world (N, 3)
#     pos = env.target_root_pos_w.clone()

#     # place marker slightly above ground so it doesn't z-fight with the floor
#     pos[:, 2] += 0.05

#     # target yaw
#     yaw = env.target_root_yaw

#     # build quaternion with roll=0, pitch=0, yaw=yaw
#     zeros = torch.zeros_like(yaw)
#     quat = quat_from_euler_xyz(zeros, zeros, yaw)  # (N, 4)

#     # access the existing visualizer
#     marker = env.scene.visualizers[visualizer_name]

#     # set poses for all envs
#     marker.set_world_poses(pos, quat)
