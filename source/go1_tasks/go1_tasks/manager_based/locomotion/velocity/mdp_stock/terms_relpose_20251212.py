# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

# -------------------------------------------------------------------------
# Simple global step counter for debug printing
# -------------------------------------------------------------------------
_debug_relpose_counter = 0


def _ensure_relpose_buffers(env: ManagerBasedRLEnv):
    """Allocate buffers on the env the first time we use them.

    Buffers (all shape: [num_envs, ...]):

    - env.rel_pose_cmd:       [dx, dy, dz, dyaw]
    - env.target_root_pos_w:  [x, y, z] target base position in world
    - env.target_root_yaw:    scalar yaw angle (rad) in world
    """
    if not hasattr(env, "rel_pose_cmd"):
        device = env.device
        num_envs = env.num_envs

        env.rel_pose_cmd = torch.zeros(num_envs, 4, device=device)
        env.target_root_pos_w = torch.zeros(num_envs, 3, device=device)
        env.target_root_yaw = torch.zeros(num_envs, device=device)


# -----------------------------------------------------------------------------
# Event: sample a new Δpose command and compute absolute target pose
# -----------------------------------------------------------------------------

def reset_rel_pose_command(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    ranges: dict,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """On reset, sample a relative pose (dx, dy, dz, dyaw) and store target pose.

    ranges: dict with keys "dx", "dy", "dz", "dyaw", each (min, max).
    """
    _ensure_relpose_buffers(env)

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    asset = env.scene[asset_cfg.name]

    # ------------------------------------------------------------------
    # 1) Sample Δpose for these envs
    # ------------------------------------------------------------------
    dx = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dx"])
    dy = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dy"])
    dz = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dz"])
    dyaw = torch.empty(len(env_ids), device=env.device).uniform_(*ranges["dyaw"])

    cmd = torch.stack((dx, dy, dz, dyaw), dim=-1)    # (N, 4)
    env.rel_pose_cmd[env_ids] = cmd

    # DEBUG: print first few envs at reset
    print("\n[RELPOSE RESET] env_ids:", env_ids[:5])
    print("  rel_pose_cmd (first 3):", env.rel_pose_cmd[env_ids[:3]].detach().cpu().numpy())

    # ------------------------------------------------------------------
    # 2) Compute absolute target pose in world frame
    # ------------------------------------------------------------------
    # current base pose in world
    root_state = asset.data.root_state_w[env_ids]    # (N, 13)
    root_pos = root_state[:, 0:3]                    # (N, 3)
    root_quat = root_state[:, 3:7]                   # (N, 4)
    _, _, root_yaw = euler_xyz_from_quat(root_quat)  # each (N,)

    # target position & yaw
    target_pos = root_pos + cmd[:, :3]               # apply [dx, dy, dz] in world
    target_yaw = root_yaw + cmd[:, 3]                # yaw target

    env.target_root_pos_w[env_ids] = target_pos
    env.target_root_yaw[env_ids] = target_yaw


# -----------------------------------------------------------------------------
# Observation terms
# -----------------------------------------------------------------------------

def rel_pose_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the sampled relative command [dx, dy, dz, dyaw]."""
    _ensure_relpose_buffers(env)
    return env.rel_pose_cmd


def rel_pose_position_error(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Position error vector: target_pos - current_pos (world frame)."""
    _ensure_relpose_buffers(env)
    asset = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w                       # (N, 3)
    err = env.target_root_pos_w - pos                 # (N, 3)
    return err


def rel_pose_yaw_error_sincos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Yaw error encoded as [sin(e), cos(e)]."""
    _ensure_relpose_buffers(env)
    asset = env.scene[asset_cfg.name]

    root_state = asset.data.root_state_w              # (N, 13)
    root_quat = root_state[:, 3:7]
    _, _, yaw = euler_xyz_from_quat(root_quat)        # (N,)

    yaw_err = env.target_root_yaw - yaw               # (N,)
    # normalize to [-pi, pi]
    yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))

    return torch.stack((torch.sin(yaw_err), torch.cos(yaw_err)), dim=-1)


# -----------------------------------------------------------------------------
# Reward terms
# -----------------------------------------------------------------------------

# def rel_pose_position_l2_reward(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Reward = -||pos_error||^2 (larger is better when closer)."""
#     _ensure_relpose_buffers(env)
#     asset = env.scene[asset_cfg.name]
#     pos = asset.data.root_pos_w
#     err = env.target_root_pos_w - pos                 # (N, 3)
#     d2 = torch.sum(err * err, dim=-1)                 # (N,)
#     return -d2


# def rel_pose_yaw_l2_reward(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Reward = -yaw_err^2 (yaw_err normalized to [-pi, pi])."""
#     _ensure_relpose_buffers(env)
#     asset = env.scene[asset_cfg.name]

#     root_state = asset.data.root_state_w
#     root_quat = root_state[:, 3:7]
#     _, _, yaw = euler_xyz_from_quat(root_quat)

#     yaw_err = env.target_root_yaw - yaw
#     yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))

#     return -(yaw_err * yaw_err)

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

# def rel_pose_position_reward(
#     env: ManagerBasedRLEnv,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     sigma: float = 0.1,
# ) -> torch.Tensor:
#     """Exponential reward based on position error norm."""
#     _ensure_relpose_buffers(env)
#     asset = env.scene[asset_cfg.name]
#     pos = asset.data.root_pos_w
#     err = env.target_root_pos_w - pos          # (N, 3)
#     d2 = torch.sum(err * err, dim=-1)          # (N,)
#     return torch.exp(-d2 / (2.0 * sigma * sigma))

def rel_pose_position_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sigma: float = 0.3,
) -> torch.Tensor:
    """Exponential reward based on position error norm.

    Also prints debug info for env 0 every N steps so we can see
    whether the robot is actually going to the commanded pose.
    """
    global _debug_relpose_counter
    _ensure_relpose_buffers(env)

    asset = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w                      # (N, 3)
    err = env.target_root_pos_w - pos               # (N, 3)
    d2 = torch.sum(err * err, dim=-1)               # (N,)

    # ================= DEBUG PRINT BLOCK =================
    _debug_relpose_counter += 1
    if _debug_relpose_counter % 50 == 0:
        # pick env 0 as representative
        i = 0
        cmd = env.rel_pose_cmd[i].detach().cpu().numpy()
        tgt_pos = env.target_root_pos_w[i].detach().cpu().numpy()
        cur_pos = pos[i].detach().cpu().numpy()
        dist = float(torch.sqrt(d2[i]).detach().cpu().item())

        # current yaw vs target yaw
        root_state = asset.data.root_state_w         # (N, 13)
        root_quat = root_state[:, 3:7]
        _, _, yaw_all = euler_xyz_from_quat(root_quat)
        cur_yaw = float(yaw_all[i].detach().cpu().item())
        tgt_yaw = float(env.target_root_yaw[i].detach().cpu().item())

        yaw_err = tgt_yaw - cur_yaw
        yaw_err = float(torch.atan2(
            torch.sin(torch.tensor(yaw_err)),
            torch.cos(torch.tensor(yaw_err)),
        ).item())

        print("\n[RELPOSE DEBUG] Env 0")
        # print(f"  rel_pose_cmd [dx, dy, dz, dyaw]: {cmd}")
        # print(f"  target_pos_w: {tgt_pos}")
        # print(f"  current_pos_w: {cur_pos}")
        print(f"  pos_error_norm: {dist:.3f} m and yaw_error: {yaw_err:.3f} rad")
        # print(f"  target_yaw: {tgt_yaw:.3f} rad, current_yaw: {cur_yaw:.3f} rad")
        # print(f"  yaw_error: {yaw_err:.3f} rad")
    # ====================================================

    return torch.exp(-d2 / (2.0 * sigma * sigma))




def stand_height_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.33,
    sigma: float = 0.05,
) -> torch.Tensor:
    """Reward = - (root_z - target_height)^2 (penalize wrong COM height)."""
    asset = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2]
    diff = h - target_height
    return -(diff * diff)


def time_penalty(env: ManagerBasedRLEnv, value: float = 1.0) -> torch.Tensor:
    """Constant per-step value (used with negative weight as a time penalty)."""
    return torch.ones(env.num_envs, device=env.device) * value
