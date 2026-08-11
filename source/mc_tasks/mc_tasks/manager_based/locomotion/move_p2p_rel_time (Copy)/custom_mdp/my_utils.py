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
        # NOTE: previously this was a flat positional list assuming Go1's
        # per-leg-grouped DOF order ([hip0, thigh0, knee0, hip1, ...]).
        # MC's actual DOF order is grouped BY JOINT TYPE instead
        # (['q0','q3','q6','q9', 'q1','q4','q7','q10', 'q2','q5','q8','q11']
        #  i.e. all hips, then all thighs, then all knees) — so assigning a
        # flat list positionally silently applied the wrong angle to the
        # wrong joint for every DOF past index 0. Build it by NAME instead,
        # so it's correct regardless of DOF ordering on any given asset.
        joint_pos_by_name = {
            "q0": 0.1, "q3": -0.1, "q6": -0.1, "q9": 0.1,        # hips
            "q1": 0.8, "q4": -0.8, "q7": 0.8, "q10": -0.8,        # upper legs
            "q2": 1.5, "q5": 1.5, "q8": 1.5, "q11": 1.5,    # lower legs
       }
        asset = env.scene["robot"]
        q_des = torch.zeros(len(asset.data.joint_names), device=device)
        for name, val in joint_pos_by_name.items():
            idx = asset.data.joint_names.index(name)
            q_des[idx] = val
        env.nominal_joint_pos = q_des.unsqueeze(0).repeat(num_envs, 1)
        print(f"\n[DEBUG utils.py] joint_names order : {asset.data.joint_names}")
        print(f"[DEBUG utils.py] nominal_joint_pos   : {q_des.tolist()}\n")

    if not hasattr(env, "feet_indices"):
        try:
            asset = env.scene["robot"]
            feet_indices, _ = asset.find_bodies(".*foot")
            env.feet_indices = feet_indices
        except Exception:
            env.feet_indices = None
