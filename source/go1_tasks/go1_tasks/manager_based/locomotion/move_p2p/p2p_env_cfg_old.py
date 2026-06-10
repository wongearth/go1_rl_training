# File: p2p_env_cfg.py

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.envs.mdp import ObservationTermCfg, EventTermCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG

# from .mdp import terms_relpose as p2p_mdp
from . import mdp as p2p_mdp

@configclass
class UnitreeGo1P2PEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Go1 environment: track a relative pose command on flat ground."""

    def __post_init__(self):
        # Initialize the core locomotion buffers
        super().__post_init__()

        # ===========================================================================
        # 1. SCENE & PHYSICS (Merged Rough + Flat)
        # ===========================================================================
        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.env_spacing = 2.5
        self.actions.joint_pos.scale = 0.30

        # Flatten Terrain
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.curriculum.terrain_levels = None

        # ===========================================================================
        # 2. COMMANDS & OBSERVATIONS
        # ===========================================================================
        # The custom P2P ranges
        self.command_rel_pose_ranges = {"dx": (-1.0, 1.0), "dy": (-0.5, 0.5), "dz": (0.0, 0.0), "dyaw": (-0.8, 0.8)}

        # Restrict old velocity commands
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0) 
        self.commands.base_velocity.ranges.lin_vel_y = (1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)

        # Observations
        self.observations.policy.height_scan = None # Disabled for flat ground
        if hasattr(self.observations.policy, "commands"):
            self.observations.policy.commands.scale = [2.0, 2.0, 1.0]
            
        self.observations.policy.rel_pose_cmd = ObservationTermCfg(func=p2p_mdp.rel_pose_command)
        self.observations.policy.rel_pos_error = ObservationTermCfg(func=p2p_mdp.rel_pose_position_error)
        self.observations.policy.rel_yaw_error = ObservationTermCfg(func=p2p_mdp.rel_pose_yaw_error_sincos)
        self.observations.policy.concatenate_terms = True

        # ===========================================================================
        # 3. EVENTS (Resets & Pushes)
        # ===========================================================================
        # self.events.push_robot = None # Disable random pushes
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
        
        # Add custom P2P command reset
        self.events.reset_rel_pose = EventTermCfg(
            func=p2p_mdp.reset_rel_pose_command,
            mode="reset",
            params={"ranges": self.command_rel_pose_ranges, "asset_cfg": SceneEntityCfg("robot")},
        )

        # ===========================================================================
        # 4. REWARDS & TERMINATIONS
        # ===========================================================================
        # Setup specific body parts
        # self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        # self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh"

        # # P2P Specific Rewards
        # self.rewards.rel_pos_tracking = RewTerm(
        #     func=p2p_mdp.rel_pose_position_reward, 
        #     weight=4.0, 
        #     params={"asset_cfg": SceneEntityCfg("robot"), "sigma": 0.5}, 
        # )
        # self.rewards.rel_yaw_tracking = RewTerm(
        #     func=p2p_mdp.rel_pose_yaw_reward, 
        #     weight=2.0, 
        #     params={"asset_cfg": SceneEntityCfg("robot"), "sigma": 0.5},
        # )
        # self.rewards.nominal_pos_penalty = RewTerm(
        #     func=p2p_mdp.nominal_joint_position_l2_penalty, 
        #     weight=-0.2, 
        #     params={"asset_cfg": SceneEntityCfg("robot")},
        # )
        # self.rewards.trot_symmetry = RewTerm(
        #     func=p2p_mdp.track_trot_symmetry_penalty,
        #         weight=-0.25, 
        #         params={"asset_cfg": SceneEntityCfg("robot")},
        # )

        # # 1. Punish the robot HEAVILY if it drags a foot
        # self.rewards.feet_slide = RewTerm(
        #     func=mdp.feet_slide, 
        #     weight=-0.5,   # <-- Raised from -1.0 to make sliding hurt more
        #     params={
        #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"), 
        #         "asset_cfg": SceneEntityCfg("robot", body_names=".*foot")            
        #     }
        # )

        # # 2. NEW: Forbid the calves, thighs, and hips from touching the ground
        # self.rewards.undesired_contacts = RewTerm(
        #     func=mdp.undesired_contacts,
        #     weight=-2.0, 
        #     params={
        #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*(thigh|calf|hip)"),
        #         "threshold": 1.0
        #     }
        # )

        # # 3. Reward the robot for picking all 4 feet up (Fixing the math trap)
        # self.rewards.equal_feet_air_time = RewTerm(
        #     func=p2p_mdp.pure_feet_air_time, 
        #     weight=0.5, 
        #     params={
        #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*foot"), 
        #         "threshold": 0.15  # <-- Lowered to 0.15s so normal, fast steps yield positive points!
        #     }
        # )


        # # Flattened/Adjusted Base Penalties
        # self.rewards.flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
        # self.rewards.base_height_l2 = RewTerm(func=mdp.base_height_l2, 
        #     weight=-1.0, params={"target_height": 0.38})
        # self.rewards.dof_pos_limits.weight = -0.1 
        # self.rewards.ang_vel_xy_l2.weight = -0.005 
        # self.rewards.dof_torques_l2.weight = -0.00005
        # self.rewards.action_rate_l2.weight = -0.02 
        # # self.rewards.feet_air_time.weight = 0.5


        # ==========================================================
        # 1. THE OBJECTIVE (Go to the point and stop)
        # ==========================================================
        self.rewards.rel_pos_tracking = RewTerm(
            func=p2p_mdp.rel_pose_position_reward, weight=4.0, 
            params={"asset_cfg": SceneEntityCfg("robot"), "sigma": 0.5}
        )
        self.rewards.rel_yaw_tracking = RewTerm(
            func=p2p_mdp.rel_pose_yaw_reward, weight=2.0, 
            params={"asset_cfg": SceneEntityCfg("robot"), "sigma": 0.5}
        )

        # ==========================================================
        # 2. POSTURE (Look like a standing dog)
        # ==========================================================
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2, weight=-1.0, 
            params={"target_height": 0.38}
        )
        self.rewards.flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-2.5)
        self.rewards.nominal_pos_penalty = RewTerm(
            func=p2p_mdp.nominal_joint_position_l2_penalty, weight=-0.5, 
            params={"asset_cfg": SceneEntityCfg("robot")}
        )
        
        # Keep the Commando Crawl penalty so it doesn't drag its knees
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts, weight=-1.0, 
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*(thigh|calf|hip)"), "threshold": 1.0}
        )

        # ==========================================================
        # 3. ENERGY EFFICIENCY (The true cure for dragging)
        # ==========================================================
        # We increase the torque penalty significantly. Dragging fights friction and burns 
        # torque. The robot will be forced to lift its feet to conserve energy.
        self.rewards.dof_torques_l2.weight = -0.0002 
        
        # We punish frantic, jerky movements to force smooth stepping
        self.rewards.action_rate_l2.weight = -0.05 

        # ==========================================================
        # 4. DELETE THE MATH TRAPS
        # ==========================================================
        self.rewards.feet_slide = None
        self.rewards.equal_feet_air_time = None
        self.rewards.trot_symmetry = None


        # Kill unwanted velocity rewards
        if hasattr(self.rewards, "track_lin_vel_xy_exp"): self.rewards.track_lin_vel_xy_exp.weight = 0.0
        if hasattr(self.rewards, "track_ang_vel_z_exp"): self.rewards.track_ang_vel_z_exp.weight = 0.0
        if hasattr(self.rewards, "lin_vel_z_l2"): self.rewards.lin_vel_z_l2.weight = -2.0 
        if hasattr(self.rewards, "feet_air_time_exp"): self.rewards.feet_air_time_exp.weight = -0.25

        # Terminations
        self.terminations.base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"), "threshold": 1.0},
        )


        # ===========================================================================
        # 5. BAGGAGE CLEANUP (Destroying the inherited Velocity stuff)
        # ===========================================================================
        # 1. Delete the velocity commands generator completely
        self.commands.base_velocity = None 
        
        # 2. Delete velocity commands from the neural network's observations
        if hasattr(self.observations.policy, "velocity_commands"):
            self.observations.policy.velocity_commands = None
            
        # 3. Delete the velocity tracking rewards completely
        if hasattr(self.rewards, "track_lin_vel_xy_exp"): self.rewards.track_lin_vel_xy_exp = None
        if hasattr(self.rewards, "track_ang_vel_z_exp"): self.rewards.track_ang_vel_z_exp = None

        # 4. Delete the velocity-dependent air time reward
        self.rewards.feet_air_time = None

# ===========================================================================
# PLAY CONFIGURATION (For Visualization)
# ===========================================================================
@configclass
class UnitreeGo1P2PEnvCfg_PLAY(UnitreeGo1P2PEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 50
        self.command_rel_pose_ranges["dyaw"] = (-0.26, 0.26)
        self.observations.policy.enable_corruption = False
        
        # Cleanup inherited resets
        if hasattr(self.events, "reset_base"): self.events.reset_base = None
        if hasattr(self.events, "reset_robot_joints"): self.events.reset_robot_joints = None
        if hasattr(self.events, "reset_rel_pose"): self.events.reset_rel_pose = None

        # Atomic Resets for smooth video playback
        self.events.startup_atomic_reset = EventTermCfg(
            func=p2p_mdp.reset_playback_episode,
            mode="reset",
            params={"ranges": self.command_rel_pose_ranges, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.events.interval_atomic_reset = EventTermCfg(
            func=p2p_mdp.reset_playback_episode,
            mode="interval",
            interval_range_s=(5.0, 5.0),
            params={"ranges": self.command_rel_pose_ranges, "asset_cfg": SceneEntityCfg("robot")},
        )