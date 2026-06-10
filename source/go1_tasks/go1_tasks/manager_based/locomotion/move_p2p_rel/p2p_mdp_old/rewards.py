import math
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi
from .utils import _ensure_relpose_buffers, RelposeDebug
from isaaclab.sensors import ContactSensor


def rel_pose_position_reward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), sigma: float = 0.5) -> torch.Tensor:
    _ensure_relpose_buffers(env)
    asset = env.scene[asset_cfg.name]
    pos = asset.data.root_pos_w                       
    err = env.target_root_pos_w - pos                 
    d2 = torch.sum(err * err, dim=-1)                 

    RelposeDebug.counter += 1
    if env.num_envs < 100 and RelposeDebug.counter % 50 == 0:
        i = 0 
        cmd = env.rel_pose_cmd[i]
        cmd_x, cmd_y = cmd[0].item(), cmd[1].item()
        cmd_yaw_deg = math.degrees(cmd[3].item())
        
        cur_x, cur_y = pos[i, 0].item(), pos[i, 1].item()
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
    return torch.sum(deviation * deviation, dim=-1) 

def track_trot_symmetry_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    _ensure_relpose_buffers(env)
    if env.feet_indices is None: return torch.zeros(env.num_envs, device=env.device)
    asset = env.scene[asset_cfg.name]
    feet_vel = asset.data.body_lin_vel_w[:, env.feet_indices, 2] 
    diff1 = torch.abs(feet_vel[:, 0] - feet_vel[:, 3])
    diff2 = torch.abs(feet_vel[:, 1] - feet_vel[:, 2])
    return diff1 + diff2

def pure_feet_air_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Rewards air time for all 4 feet equally, ignoring velocity commands."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    
    # Check which feet just hit the ground this exact physics frame
    first_contact = contact_sensor.compute_first_contact(env.step_dt)
    
    # Calculate how long the foot was in the air before touching down
    reward = (contact_sensor.data.current_air_time - threshold) * first_contact
    
    # Sum the reward across all 4 legs to force equal movement
    return reward.sum(dim=1)