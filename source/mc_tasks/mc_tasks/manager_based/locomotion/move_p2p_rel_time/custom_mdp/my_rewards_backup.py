import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import wrap_to_pi, euler_xyz_from_quat
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from isaaclab.managers import RewardTermCfg


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
    env: ManagerBasedRLEnv, command_name: str, 
    pos_std: float = 0.25, 
    yaw_std: float = 0.15
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
    is_arrived = (distance < 0.05).float()
    
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

# def arrived_on_time_bonus(
#     env, 
#     command_name: str, 
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
#     threshold: float = 0.05, # 15cm tolerance
#     bonus: float = 10.0      # High bonus
# ) -> torch.Tensor:
#     """Gives a large bonus if the robot is within threshold at the end of the time limit."""
    
#     # Only reward at the very end of the episode (when the timer hits zero)
#     # We check if the episode is ending based on the episode length buffer
#     is_at_end = (env.episode_length_buf >= env.max_episode_length - 1)
    
#     asset = env.scene[asset_cfg.name]
#     command = env.command_manager.get_command(command_name)
    
#     # Calculate distance error (using the local tracking error from command buffer)
#     dist_error = torch.norm(command[:, :2], dim=1)
    
#     # Check if we are close enough
#     is_success = (dist_error < threshold)
    
#     # Return bonus only if both conditions are met
#     return is_success * is_at_end * bonus

def arrived_on_time_bonus(
    env, 
    command_name: str, 
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.05,       # 5cm position tolerance
    yaw_threshold: float = 0.087,  # ~5 degrees yaw tolerance (in radians)
    bonus: float = 10.0
) -> torch.Tensor:
    """Gives a large bonus if the robot is within threshold (dist and yaw) at the end of the time limit."""
    
    # Only reward at the very end of the episode
    is_at_end = (env.episode_length_buf >= env.max_episode_length - 1)
    
    command = env.command_manager.get_command(command_name)
    
    # Calculate distance error (X, Y)
    dist_error = torch.norm(command[:, :2], dim=1)
    
    # Calculate yaw error (Absolute value of the angular difference)
    yaw_error = torch.abs(command[:, 3])
    
    # Check if we are close enough in BOTH position and heading
    is_success = (dist_error < threshold) & (yaw_error < yaw_threshold)
    
    # Return bonus only if both conditions are met at the end of the episode
    return is_success.float() * is_at_end.float() * bonus

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

def in_box_bonus(
    env: "ManagerBasedRLEnv",
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.05,        # 5cm position tolerance
    yaw_threshold: float = 0.087,   # ~5 deg
    height_target: float = 0.35,
    height_tol: float = 0.04,       # must be within 4cm of target height
) -> torch.Tensor:
    """Dense per-step reward for being in the goal box at correct height and upright."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    dist_error = torch.norm(command[:, :2], dim=1)
    yaw_error = torch.abs(command[:, 3])

    # height gate: forfeits reward if the robot crouches to hold position
    body_height = asset.data.root_pos_w[:, 2]
    height_ok = torch.abs(body_height - height_target) < height_tol

    in_box = (dist_error < threshold) & (yaw_error < yaw_threshold) & height_ok
    return in_box.float()


# def gait_symmetry_penalty(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
#     """
#     Penalizes non-trotting gaits by evaluating all four feet.
#     A true trot requires diagonal pairs to move together, meaning:
#     1. Front legs alternate
#     2. Rear legs alternate
#     3. Same-side legs alternate
#     """
#     contact_sensor = env.scene.sensors[sensor_cfg.name]
    
#     # Check which feet are touching the ground
#     contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    
#     # WARNING: This assumes your contact indices are: 
#     # 0 = Front-Left (FL)
#     # 1 = Front-Right (FR)
#     # 2 = Rear-Left (RL)
#     # 3 = Rear-Right (RR)
#     fl = contacts[:, 0]
#     fr = contacts[:, 1]
#     rl = contacts[:, 2]
#     rr = contacts[:, 3]
    
#     # 1. Penalize if front legs do the same thing
#     front_penalty = (fl == fr).float()
    
#     # 2. Penalize if rear legs do the same thing
#     rear_penalty = (rl == rr).float()
    
#     # 3. Penalize if the left-side legs do the same thing (prevents pacing)
#     side_penalty = (fl == rl).float()
    
#     # Sum them all together. A perfect diagonal trot results in a penalty of 0.0
#     return front_penalty + rear_penalty + side_penalty

# def stride_length_reward(
#     env,
#     sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="LowerLeg.*"),
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="LowerLeg.*"),
#     max_stride: float = 0.30,     # clip: strides longer than this earn no extra (m)
#     contact_force_n: float = 20.0,  # force below this = foot considered airborne
#     debug: bool = True,
# ) -> torch.Tensor:
#     """Reward horizontal foot displacement over a completed swing (liftoff -> touchdown).

#     Air/contact is determined by thresholding the contact force magnitude, NOT by the
#     sensor's current_air_time (which never registered lift on this robot).
#     Fires once per foot at the touchdown step.
#     """
#     sensor = env.scene.sensors[sensor_cfg.name]
#     asset = env.scene[asset_cfg.name]

#     foot_ids = sensor_cfg.body_ids     # feet indices in the CONTACT sensor
#     body_ids = asset_cfg.body_ids      # feet indices in the ARTICULATION
#     n_env = env.num_envs
#     n_feet = len(foot_ids)

#     # --- contact/air detection by force threshold ---
#     forces = sensor.data.net_forces_w[:, foot_ids].norm(dim=-1)   # (n_env, n_feet)
#     in_contact = forces > contact_force_n
#     in_air = ~in_contact                                          # (n_env, n_feet) bool

#     # current horizontal foot positions (world): (n_env, n_feet, 2)
#     foot_pos_xy = asset.data.body_pos_w[:, body_ids, :2]

#     # --- lazily create persistent buffers on env ---
#     if not hasattr(env, "_stride_liftoff_xy"):
#         env._stride_liftoff_xy = foot_pos_xy.clone()
#         env._stride_was_in_air = torch.zeros((n_env, n_feet), dtype=torch.bool, device=env.device)

#     # feet that just LIFTED OFF this step (were down, now in air): record liftoff xy
#     just_lifted = in_air & (~env._stride_was_in_air)
#     env._stride_liftoff_xy = torch.where(
#         just_lifted.unsqueeze(-1), foot_pos_xy, env._stride_liftoff_xy
#     )

#     # feet that just TOUCHED DOWN this step (were in air, now down)
#     just_landed = (~in_air) & env._stride_was_in_air

#     # horizontal distance traveled during the swing
#     stride = torch.norm(foot_pos_xy - env._stride_liftoff_xy, dim=-1)   # (n_env, n_feet)
#     stride = torch.clamp(stride, max=max_stride)

#     # reward only on the landing step, summed over feet
#     reward = torch.sum(stride * just_landed.float(), dim=1)

#     # --- diagnostic (uses the NEW force-based in_air) ---
#     if debug and env.common_step_counter % 50 == 0:
#         foot_z = asset.data.body_pos_w[:, body_ids, 2]
#         print("force[0]  :", [f"{f:.2f}" for f in forces[0].tolist()])
#         print("in_air[0] :", in_air[0].tolist(),
#               " n_feet_in_air:", int(in_air[0].sum().item()), "of", n_feet)
#         print("foot_z[0] :", [f"{z:.3f}" for z in foot_z[0].tolist()])
#         print("landed[0] :", just_landed[0].tolist(),
#               " stride[0]:", [f"{s:.3f}" for s in stride[0].tolist()])

#     # update air-state memory for next step
#     env._stride_was_in_air = in_air
#     return reward


# def trot_gait_phase_reward(
#     env,
#     sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="LowerLeg.*"),
#     period: float = 0.5,                 # full gait cycle duration (s); ~0.4-0.6 typical
#     diagonal_a: tuple = (0, 3),          # indices (in LowerLeg.* order) of FL, RR
#     diagonal_b: tuple = (1, 2),          # indices of FR, RL  -- VERIFY THESE
#     command_name: str = "base_pose",
# ) -> torch.Tensor:
#     """Reward matching a trot contact schedule via a phase clock.

#     Diagonal pair A should be airborne in the first half of each cycle,
#     diagonal pair B airborne in the second half. Rewards feet whose
#     contact state agrees with the schedule.
#     """
#     sensor = env.scene.sensors[sensor_cfg.name]
#     foot_ids = sensor_cfg.body_ids

#     # is each foot in contact this step? use air time: in contact => current_air_time == 0
#     cur_air = sensor.data.current_air_time[:, foot_ids]       # (n_env, n_feet)
#     in_contact = (cur_air <= 1e-4).float()                    # 1 = stance, 0 = swing

#     # phase clock in [0,1)
#     t_elapsed = env.episode_length_buf * env.step_dt
#     phase = (t_elapsed / period) % 1.0                        # (n_env,)

#     # first half: pair A should SWING (in_contact ~ 0), pair B should STANCE (~1)
#     # second half: reversed
#     a_should_stance = (phase >= 0.5).float()                  # 1 in 2nd half
#     b_should_stance = (phase < 0.5).float()                   # 1 in 1st half

#     a_idx = list(diagonal_a)
#     b_idx = list(diagonal_b)

#     # reward = agreement between desired stance and actual contact, per foot
#     # agreement = 1 - |desired - actual|
#     a_actual = in_contact[:, a_idx]                           # (n_env, 2)
#     b_actual = in_contact[:, b_idx]

#     a_match = 1.0 - torch.abs(a_should_stance.unsqueeze(1) - a_actual)
#     b_match = 1.0 - torch.abs(b_should_stance.unsqueeze(1) - b_actual)

#     # average agreement across all four feet -> [0,1]
#     reward = (a_match.sum(dim=1) + b_match.sum(dim=1)) / 4.0

#     # optional: only reward the gait while the robot is actually moving toward target
#     command = env.command_manager.get_command(command_name)
#     dist = torch.norm(command[:, :2], dim=1)
#     moving = (dist > 0.10).float()          # don't enforce trot once parked in the box
#     return reward * moving

def feet_air_time_force(
    env,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names="LowerLeg.*"),
    command_name: str = "base_pose",
    threshold: float = 0.20,        # target swing duration (s); set BELOW median swing
    contact_force_n: float = 22.0,  # force below this = airborne
) -> torch.Tensor:
    """feet_air_time computed from a force threshold instead of the sensor's own
    contact state (which never registered lift on this robot).

    Rewards each completed swing by (air_time - threshold), counted at touchdown.
    Only active while the robot is still moving toward the target.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    foot_ids = sensor_cfg.body_ids
    n_env = env.num_envs
    n_feet = len(foot_ids)

    forces = sensor.data.net_forces_w[:, foot_ids].norm(dim=-1)
    in_contact = forces > contact_force_n
    in_air = ~in_contact

    # persistent air-time accumulator
    if not hasattr(env, "_fat_air_time"):
        env._fat_air_time = torch.zeros((n_env, n_feet), device=env.device)
        env._fat_was_in_air = torch.zeros((n_env, n_feet), dtype=torch.bool, device=env.device)

    # accumulate air time while airborne, reset to 0 on contact
    env._fat_air_time = torch.where(
        in_air,
        env._fat_air_time + env.step_dt,
        torch.zeros_like(env._fat_air_time),
    )

    # touchdown this step = was in air, now in contact
    just_landed = in_contact & env._fat_was_in_air

    # air time of the swing that just completed (value BEFORE it was zeroed above:
    # capture it from the previous step's accumulator)
    completed_air = torch.where(just_landed, env._prev_fat_air_time, torch.zeros_like(env._fat_air_time)) \
        if hasattr(env, "_prev_fat_air_time") else torch.zeros_like(env._fat_air_time)

    reward = torch.sum((completed_air - threshold) * just_landed.float(), dim=1)

    # gate: only reward stepping while moving toward target (not while parked)
    command = env.command_manager.get_command(command_name)
    dist = torch.norm(command[:, :2], dim=1)
    reward = reward * (dist > 0.10).float()

    # save state for next step
    env._prev_fat_air_time = env._fat_air_time.clone()
    env._fat_was_in_air = in_air
    return reward



class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs defined in :attr:`synced_feet_pair_names`
    to bias the policy towards a desired gait, i.e trotting, bounding, or pacing. Note that this reward is only for
    quadrupedal gaits with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        max_err: float,
        velocity_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3

        # only enforce gait if cmd > 0
        # cmd = torch.norm(env.command_manager.get_command("base_velocity"), dim=1)
        # body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
        # return torch.where(
        #     torch.logical_or(cmd > 0.0, body_vel > self.velocity_threshold), sync_reward * async_reward, 0.0
        # )
        
        body_vel = torch.linalg.norm(self.asset.data.root_lin_vel_b[:, :2], dim=1)
        return torch.where(
            body_vel > self.velocity_threshold, sync_reward * async_reward, 0.0
        )
    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)

def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)


def joint_acceleration_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint accelerations on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_acc), dim=1)

def joint_torques_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint torques on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.applied_torque), dim=1)

def joint_velocity_penalty(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint velocities on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.linalg.norm((asset.data.joint_vel), dim=1)


