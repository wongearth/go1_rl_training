import torch
from isaaclab.envs import ManagerBasedRLEnv

class RelposeDebug:
    counter = 0

def _ensure_relpose_buffers(env: ManagerBasedRLEnv):
    """Allocate buffers on the env the first time we use them."""
    device = env.device
    num_envs = env.num_envs

    if not hasattr(env, "rel_pose_cmd"):
        env.rel_pose_cmd = torch.zeros(num_envs, 4, device=device)
        env.target_root_pos_w = torch.zeros(num_envs, 3, device=device)
        env.target_root_yaw = torch.zeros(num_envs, device=device)
        env.command_start_step = torch.zeros(num_envs, device=device)

    if not hasattr(env, "nominal_joint_pos"):
        base_pose = [0.1, 0.8, -1.5,   -0.1, 0.8, -1.5,   0.1, 0.8, -1.5,   -0.1, 0.8, -1.5]
        q_des = torch.tensor(base_pose, device=device)
        env.nominal_joint_pos = q_des.unsqueeze(0).repeat(num_envs, 1)

    if not hasattr(env, "feet_indices"):
        try:
            asset = env.scene["robot"]
            feet_indices, _ = asset.find_bodies(".*foot")
            env.feet_indices = feet_indices
        except Exception:
            env.feet_indices = None