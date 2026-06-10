# File: p2p_env_cfg.py

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs.mdp import ObservationTermCfg, EventTermCfg

import isaaclab.envs.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG

# The Critical MDP Import
from .mdp import terms_relpose as relpose_mdp


# ===========================================================================
# 1. BASE ROUGH CONFIGURATION
# ===========================================================================
@configclass
class UnitreeGo1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        if hasattr(self.observations.policy, "commands"):
            self.observations.policy.commands.scale = [2.0, 2.0, 1.0]
                    
        # Physics & Hardware
        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        self.actions.joint_pos.scale = 0.30
        
        # Command Restrictions
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0) 
        self.commands.base_velocity.ranges.lin_vel_y = (1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)

        # Crash Fixes
        self.events.push_robot = None
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
        self.curriculum.terrain_levels = None 

        # Rewards
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh"
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=mdp.track_lin_vel_xy_exp, weight=1.0, params={"command_name": "base_velocity", "std": 1.0},
        )
        self.rewards.track_ang_vel_z_exp = RewTerm(
            func=mdp.track_ang_vel_z_exp, weight=2.5, params={"command_name": "base_velocity", "std": 1.0},
        )
        self.rewards.dof_pos_limits.weight = -0.1 
        self.rewards.ang_vel_xy_l2.weight = -0.001 
        self.rewards.dof_torques_l2.weight = -0.000025
        self.rewards.action_rate_l2.weight = -0.01 
        self.rewards.feet_air_time.weight = 0.5

        # Terminations
        self.terminations.base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"), "threshold": 1.0},
        )


# ===========================================================================
# 2. FLAT CONFIGURATION
# ===========================================================================
@configclass
class UnitreeGo1FlatEnvCfg(UnitreeGo1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Overrides
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2, weight=-2.0, params={"target_height": 0.38}, 
        )

        # Flatten Terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.curriculum.terrain_levels = None


# ===========================================================================
# 3. P2P (RELPOSE) CONFIGURATION
# ===========================================================================
@configclass
class UnitreeGo1P2PEnvCfg(UnitreeGo1FlatEnvCfg):
    """Go1 environment: track a relative pose command (dx, dy, dz, d_yaw)."""

    def __post_init__(self):
        super().__post_init__()

        # Command Ranges & Spacing
        self.command_rel_pose_ranges = {"dx": (-1.0, 1.0), "dy": (-0.5, 0.5), "dz": (0.0, 0.0), "dyaw": (-0.8, 0.8)}
        self.scene.env_spacing = 2.5

        # Observations
        self.observations.policy.rel_pose_cmd = ObservationTermCfg(func=relpose_mdp.rel_pose_command)
        self.observations.policy.rel_pos_error = ObservationTermCfg(func=relpose_mdp.rel_pose_position_error)
        self.observations.policy.rel_yaw_error = ObservationTermCfg(func=relpose_mdp.rel_pose_yaw_error_sincos)
        self.observations.policy.concatenate_terms = True

        # Events
        self.events.reset_rel_pose = EventTermCfg(
            func=relpose_mdp.reset_rel_pose_command,
            mode="reset",
            params={"ranges": self.command_rel_pose_ranges, "asset_cfg": SceneEntityCfg("robot")},
        )

        # Rewards
        self.rewards.rel_pos_tracking = RewTerm(
            func=relpose_mdp.rel_pose_position_reward, weight=4.0, params={"asset_cfg": SceneEntityCfg("robot"), "sigma": 0.5}, 
        )
        self.rewards.rel_yaw_tracking = RewTerm(
            func=relpose_mdp.rel_pose_yaw_reward, weight=2.0, params={"asset_cfg": SceneEntityCfg("robot"), "sigma": 0.5},
        )
        self.rewards.nominal_pos_penalty = RewTerm(
            func=relpose_mdp.nominal_joint_position_l2_penalty, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.trot_symmetry = RewTerm(
            func=relpose_mdp.track_trot_symmetry_penalty, weight=-0.5, params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Stability Tweaks
        if hasattr(self.rewards, "track_lin_vel_xy_exp"): self.rewards.track_lin_vel_xy_exp.weight = 0.0
        if hasattr(self.rewards, "track_ang_vel_z_exp"): self.rewards.track_ang_vel_z_exp.weight = 0.0
        if hasattr(self.rewards, "lin_vel_z_l2"): self.rewards.lin_vel_z_l2.weight = -2.0 
        if hasattr(self.rewards, "ang_vel_xy_l2"): self.rewards.ang_vel_xy_l2.weight = -0.05 
        if hasattr(self.rewards, "feet_air_time_exp"): self.rewards.feet_air_time_exp.weight = -0.25


# ===========================================================================
# 4. P2P PLAY CONFIGURATION
# ===========================================================================
@configclass
class UnitreeGo1P2PEnvCfg_PLAY(UnitreeGo1P2PEnvCfg):
    """Smaller, cleaner scene for visualization / playback."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.command_rel_pose_ranges["dyaw"] = (-0.26, 0.26)
        self.observations.policy.enable_corruption = False
        
        # Cleanup inherited resets
        if hasattr(self.events, "reset_base"): self.events.reset_base = None
        if hasattr(self.events, "reset_robot_joints"): self.events.reset_robot_joints = None
        if hasattr(self.events, "base_external_force_torque"): self.events.base_external_force_torque = None
        if hasattr(self.events, "push_robot"): self.events.push_robot = None
        if hasattr(self.events, "reset_rel_pose"): self.events.reset_rel_pose = None

        # Atomic Resets
        self.events.startup_atomic_reset = EventTermCfg(
            func=relpose_mdp.reset_playback_episode,
            mode="reset",
            params={"ranges": self.command_rel_pose_ranges, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.events.interval_atomic_reset = EventTermCfg(
            func=relpose_mdp.reset_playback_episode,
            mode="interval",
            interval_range_s=(5.0, 5.0),
            params={"ranges": self.command_rel_pose_ranges, "asset_cfg": SceneEntityCfg("robot")},
        )