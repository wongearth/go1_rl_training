import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat
from isaaclab.sensors import ContactSensor

def track_pos_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of point-to-point position commands (xy axes) using exponential kernel."""
    asset = env.scene[asset_cfg.name]
    
    # The command returns [x, y, yaw]
    target_pos = env.command_manager.get_command(command_name)[:, :2]
    current_pos = asset.data.root_pos_w[:, :2]
    
    # Calculate squared Euclidean distance to the target point
    pos_error = torch.sum(torch.square(target_pos - current_pos), dim=1)
    return torch.exp(-pos_error / std**2)

def track_heading_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of heading commands using exponential kernel."""
    asset = env.scene[asset_cfg.name]
    target_heading = env.command_manager.get_command(command_name)[:, 3]
    
    # Extract current yaw from the robot's base quaternion
    _, _, current_heading = euler_xyz_from_quat(asset.data.root_quat_w)
    
    # Ensure shortest path difference for angles
    heading_error = wrap_to_pi(target_heading - current_heading)
    return torch.exp(-torch.square(heading_error) / std**2)

def stand_still_joint_deviation_l1(
    env, command_name: str, command_threshold: float = 0.06, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    
    # 1. Get the current command (e.g., distance to target)
    command = env.command_manager.get_command(command_name)
    
    # 2. Calculate the penalty: base deviation * boolean mask
    return mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def feet_slide(env: "ManagerBasedRLEnv", sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding on the ground."""
    # 1. Get the contact sensor and the robot asset
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    # 2. Check if the feet are currently experiencing a force greater than 1.0 Newton (meaning they are on the ground)
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

    # 3. Get the XY linear velocity of the feet links in the world frame
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    
    # 4. Multiply velocity by the contact mask. (If not in contact, penalty is 0)
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    
    return reward

def nominal_joint_position_l2_penalty(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint position deviations from the default/nominal posture using L2-norm."""
    # 1. Extract the robot asset
    asset = env.scene[asset_cfg.name]
    
    # 2. Get the current joint positions and the default joint positions
    current_joint_pos = asset.data.joint_pos
    default_joint_pos = asset.data.default_joint_pos
    
    # 3. Calculate the squared difference (L2 penalty)
    joint_pos_err = current_joint_pos - default_joint_pos
    
    # 4. Sum the errors across all 12 joints
    return torch.sum(torch.square(joint_pos_err), dim=1)


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

def progress_towards_pose_target(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Rewards the robot for moving in the direction of the local pose command."""
    command = env.command_manager.get_command(command_name)
    asset = env.scene[asset_cfg.name]
    
    # The command is naturally in the robot's local frame [dx, dy, dyaw]
    pos_error = command[:, :2]
    distance = torch.norm(pos_error, dim=1)
    
    # Create a normalized vector pointing to the target in the local frame
    target_dir = pos_error / (distance.unsqueeze(1) + 1e-5)
    
    # Get the robot's current local linear velocity
    local_vel = asset.data.root_lin_vel_b[:, :2] 
    
    # Dot product: Reward velocity aligned with the target direction
    # The (distance > 0.1) multiplier ensures it stops getting progress points once it arrives
    progress = torch.sum(local_vel * target_dir, dim=1)
    return progress * (distance > 0.1)

def pose_distance_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float = 1.0
) -> torch.Tensor:
    """Exponential reward for parking at the target position."""
    command = env.command_manager.get_command(command_name)
    distance_sq = torch.sum(torch.square(command[:, :2]), dim=1)
    return torch.exp(-distance_sq / (std**2))

def park_and_face_target_exp(
    env: ManagerBasedRLEnv, command_name: str, pos_std: float = 0.25, yaw_std: float = 0.5
) -> torch.Tensor:
    """Exponential reward for parking at the target position AND facing the correct yaw."""
    command = env.command_manager.get_command(command_name)
    
    # 1. Distance Error (X, Y)
    distance_sq = torch.sum(torch.square(command[:, :2]), dim=1)
    # 2. Yaw Error (Rotation)
    yaw_sq = torch.square(command[:, 3])
    
    # 3. Calculate separate exponential rewards
    pos_reward = torch.exp(-distance_sq / (pos_std**2))
    yaw_reward = torch.exp(-yaw_sq / (yaw_std**2))
    
    # 4. MULTIPLY THEM: 1.0 * 1.0 = 1.0. But if yaw is wrong (0.1), score drops to 0.1!
    return pos_reward * yaw_reward


def stand_still_when_arrived(
    env: ManagerBasedRLEnv, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint velocities heavily ONLY when the robot is parked at the target."""
    command = env.command_manager.get_command(command_name)
    
    # Get distance to target (X, Y)
    distance = torch.norm(command[:, :2], dim=1)
    
    asset = env.scene[asset_cfg.name]
    # Calculate how fast the joints are vibrating
    joint_vel_sq = torch.sum(torch.square(asset.data.joint_vel), dim=1)
    
    # Create a mask: 1.0 if arrived (distance < 0.15m), 0.0 if still running
    is_arrived = (distance < 0.15).float()
    
    # Apply penalty only when arrived!
    return joint_vel_sq * is_arrived