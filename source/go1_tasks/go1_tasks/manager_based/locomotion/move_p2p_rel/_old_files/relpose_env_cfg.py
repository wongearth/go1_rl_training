# File: isaaclab_tasks/manager_based/locomotion/velocity/config/go1/relpose_env_cfg.py

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.envs.mdp import ObservationTermCfg, RewardTermCfg, EventTermCfg
from isaaclab.managers import SceneEntityCfg

# 2. Change this ONLY IF you copied flat_env_cfg.py into your new workspace.
# If you didn't copy it over, leave this as isaaclab_tasks!
from .flat_env_cfg import UnitreeGo1FlatEnvCfg

# 3. THE CRITICAL FIX: Point this to your custom mdp folder!
from .mdp import terms_relpose as relpose_mdp

@configclass
class UnitreeGo1RelPoseEnvCfg(UnitreeGo1FlatEnvCfg):
    """Go1 environment: track a relative pose command (dx, dy, dz, d_yaw)."""

    def __post_init__(self):
        super().__post_init__()

        # ---------------------------------------------------
        # 0a) Command Ranges
        # ---------------------------------------------------
        self.command_rel_pose_ranges = {
            "dx": (-1.0, 1.0),
            "dy": (-0.5, 0.5),
            "dz": (0.0, 0.0),
            "dyaw": (-0.8, 0.8),
        }
        self.scene.env_spacing = 2.5

        # ---------------------------------------------------
        # 1) Observations
        # ---------------------------------------------------
        self.observations.policy.rel_pose_cmd = ObservationTermCfg(
            func=relpose_mdp.rel_pose_command,
        )
        self.observations.policy.rel_pos_error = ObservationTermCfg(
            func=relpose_mdp.rel_pose_position_error,
        )
        self.observations.policy.rel_yaw_error = ObservationTermCfg(
            func=relpose_mdp.rel_pose_yaw_error_sincos,
        )
        self.observations.policy.concatenate_terms = True

        # ---------------------------------------------------
        # 2) Events
        # ---------------------------------------------------
        # Standard Training Reset (Mode: Reset)
        self.events.reset_rel_pose = EventTermCfg(
            func=relpose_mdp.reset_rel_pose_command,
            mode="reset",
            params={
                "ranges": self.command_rel_pose_ranges,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # ---------------------------------------------------
        # 3) Rewards
        # ---------------------------------------------------
        self.rewards.rel_pos_tracking = RewardTermCfg(
            func=relpose_mdp.rel_pose_position_reward,
            weight=4.0,   
            params={"asset_cfg": SceneEntityCfg("robot"), "sigma": 0.5}, 
        )
        self.rewards.rel_yaw_tracking = RewardTermCfg(
            func=relpose_mdp.rel_pose_yaw_reward,
            weight=2.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "sigma": 0.5},
        )
        self.rewards.nominal_pos_penalty = RewardTermCfg(
            func=relpose_mdp.nominal_joint_position_l2_penalty, 
            weight=-0.1, 
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        # Symmetry Reward (Anti-Three-Legged Walk)
        self.rewards.trot_symmetry = RewardTermCfg(
            func=relpose_mdp.track_trot_symmetry_penalty,
            weight=-0.5,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )

        # Stability Tweaks
        if hasattr(self.rewards, "track_lin_vel_xy_exp"):
            self.rewards.track_lin_vel_xy_exp.weight = 0.0
        if hasattr(self.rewards, "track_ang_vel_z_exp"):
            self.rewards.track_ang_vel_z_exp.weight = 0.0
        if hasattr(self.rewards, "lin_vel_z_l2"):
            self.rewards.lin_vel_z_l2.weight = -2.0 
        if hasattr(self.rewards, "ang_vel_xy_l2"):
            self.rewards.ang_vel_xy_l2.weight = -0.05 
        if hasattr(self.rewards, "feet_air_time_exp"):
            self.rewards.feet_air_time_exp.weight = -0.25


@configclass
class UnitreeGo1RelPoseEnvCfg_PLAY(UnitreeGo1RelPoseEnvCfg):
    """Smaller, cleaner scene for visualization / playback."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # Override Yaw Range (+/- 15 deg)
        self.command_rel_pose_ranges["dyaw"] = (-0.26, 0.26)

        self.observations.policy.enable_corruption = False
        
        # ==================================================================
        # 🧹 AGGRESSIVE CLEANUP: KILL ALL INHERITED RESET EVENTS
        # ==================================================================
        # We must disable these so they don't fight our Atomic Reset.
        
        # 1. Kill Base Resets (from FlatEnvCfg)
        if hasattr(self.events, "reset_base"):
            self.events.reset_base = None
        if hasattr(self.events, "reset_robot_joints"):
            self.events.reset_robot_joints = None
            
        # 2. Kill Random Pushes
        if hasattr(self.events, "base_external_force_torque"):
            self.events.base_external_force_torque = None
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None
            
        # 3. Kill the Parent's 'reset_rel_pose' (Mode: Reset)
        # This one runs at startup and uses the OLD logic. We replace it below.
        if hasattr(self.events, "reset_rel_pose"):
            self.events.reset_rel_pose = None

        # ==================================================================
        # 🚀 ATOMIC RESET: The ONLY Thing That Should Move The Robot
        # ==================================================================
        
        # A. Startup Reset (Runs ONCE at the very beginning)
        # Guarantees robot spawns at (0,0,0) immediately.
        self.events.startup_atomic_reset = EventTermCfg(
            func=relpose_mdp.reset_playback_episode,
            mode="reset", # <--- Runs on env.reset()
            params={
                "ranges": self.command_rel_pose_ranges,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        # B. Interval Reset (Runs every 5.0 seconds)
        # Resets everything again periodically.
        self.events.interval_atomic_reset = EventTermCfg(
            func=relpose_mdp.reset_playback_episode,
            mode="interval",
            interval_range_s=(5.0, 5.0),
            params={
                "ranges": self.command_rel_pose_ranges,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )