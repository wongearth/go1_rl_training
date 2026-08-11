import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

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

def foot_clearance_reward(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.08,
    std: float = 0.03,
) -> torch.Tensor:
    """Reward a foot for lifting close to `target_height` while it's swinging (not in contact).

    - `contacts` tells us which feet are currently touching the ground (using the
      same >1.0 N force check as `feet_slide`).
    - `foot_z` is each foot's current height.
    - `foot_speed_xy` is how fast the foot is moving horizontally — this makes sure we
      only reward feet that are actively part of a step, not just standing still at
      the right height by accident.
    - The `(~contacts)` mask means: only feet currently in the air count. A foot that's
      planted on the ground contributes zero here, no matter its height.
    """
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0

    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    foot_speed_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)

    clearance_error = torch.square(foot_z - target_height)
    per_foot_reward = torch.exp(-clearance_error / std**2) * foot_speed_xy * (~contacts).float()
    return torch.sum(per_foot_reward, dim=1)


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

def position_tracking_exp(env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Exponential reward for being physically close to the target X/Y coordinates."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    pos_error_sq = torch.sum(torch.square(command[:, :2] - asset.data.root_pos_w[:, :2]), dim=1)
    return torch.exp(-pos_error_sq / std**2)

def heading_tracking_exp(env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """Exponential reward for pointing the robot's nose in the correct target yaw direction."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    _, _, robot_yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    yaw_error = command[:, 3] - robot_yaw
    yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
    
    return torch.exp(-torch.square(yaw_error) / std**2)

def time_remaining_normalized(env, command_name: str) -> torch.Tensor:
    """Observation: Returns countdown [1.0 to 0.0]"""
    command = env.command_manager.get_command(command_name)
    target_time = command[:, 4]
    t_elapsed = env.episode_length_buf * env.step_dt
    t_rem = torch.clamp(target_time - t_elapsed, min=0.0) / target_time
    return t_rem.unsqueeze(1)

def track_timed_trajectory_local_exp(
    env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), std: float = 1.0,
) -> torch.Tensor:
    """Reward: Forces robot to match the exact velocity needed to reach the target precisely on time."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    target_time = command[:, 4]
    t_elapsed = env.episode_length_buf * env.step_dt
    t_remaining = torch.clamp(target_time - t_elapsed, min=0.1) 
    
    # 1. The Desired Velocity is simply the remaining local distance divided by remaining time
    local_pos_error = command[:, :2]
    is_active = (t_elapsed < target_time).unsqueeze(1)
    
    desired_vel_b = torch.where(
        is_active, 
        local_pos_error / t_remaining.unsqueeze(1), 
        torch.zeros_like(local_pos_error) # If time is up, desire 0 velocity (park)
    )
    
    # 2. Actual Velocity in the robot's base frame
    yaw_q = yaw_quat(asset.data.root_quat_w)
    actual_vel_b = quat_apply_inverse(yaw_q, asset.data.root_lin_vel_w)[:, :2]
    
    vel_error = torch.sum(torch.square(desired_vel_b - actual_vel_b), dim=1)
    return torch.exp(-vel_error / std**2)

def arrived_on_time_bonus(
    env, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.15, # 15cm tolerance
    bonus: float = 10.0      # High bonus
) -> torch.Tensor:
    """Gives a large bonus if the robot is within threshold at the end of the time limit."""
    
    # Only reward at the very end of the episode (when the timer hits zero)
    # We check if the episode is ending based on the episode length buffer
    is_at_end = (env.episode_length_buf >= env.max_episode_length - 1)
    
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # Calculate distance error (using the local tracking error from command buffer)
    dist_error = torch.norm(command[:, :2], dim=1)
    
    # Check if we are close enough
    is_success = (dist_error < threshold)
    
    # Return bonus only if both conditions are met
    return is_success * is_at_end * bonus


def impatience_penalty(
    env, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.15 # 15cm
) -> torch.Tensor:
    """Penalizes the robot for being at the target if time is still remaining."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # 1. Get distance to target
    dist = torch.norm(command[:, :2], dim=1)
    
    # 2. Check if time is remaining (more than 0.5s left)
    target_time = command[:, 4]
    t_elapsed = env.episode_length_buf * env.step_dt
    time_remaining = (target_time - t_elapsed)
    
    # 3. Penalty condition: 
    # If the robot is within the threshold AND there is still significant time left
    is_impatient = (dist < threshold) & (time_remaining > 0.5)
    
    # Penalize speed if it's "impatient"
    actual_vel = torch.norm(asset.data.root_lin_vel_w[:, :2], dim=1)
    return is_impatient * actual_vel


