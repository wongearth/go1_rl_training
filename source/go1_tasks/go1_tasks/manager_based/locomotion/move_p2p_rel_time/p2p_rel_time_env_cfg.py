# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.envs import ViewerCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.actuators import ImplicitActuatorCfg # Need to disable Actuator Net that cause limping robot 
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

# Reusing the advanced math specifically designed for Spot's gait
# import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
# import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
# from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from . import custom_mdp as custom_mdp
from . import mdp as mdp

##
# Pre-defined configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG  # isort: skip

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.05), noise_step=0.02, border_width=0.25
        ),
    },
)


@configclass
class Go1CommandsCfg:
    """Command specifications for the environment."""
    base_pose = custom_mdp.LocalPoseTimeCommandCfg(
        class_type=custom_mdp.LocalPoseTimeCommand,
        resampling_time_range=(10000.0, 10000.0),
        ranges=custom_mdp.LocalPoseTimeCommandCfg.Ranges(
            pos_x=(-1.5, 1.5),
            pos_y=(-1.5, 1.5),
            heading=(-3.14, 3.14),
            # time=(1.5, 4.0) # Timed goals!
            # time=(1.5, 3.0) # Timed goals!
            time=(1.2, 2.5) # <--- The missing 5th dimension!
        )
    )

@configclass
class Go1ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=0.30, 
        use_default_offset=True)



@configclass
class Go1ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        # ... keep other standard observations ...
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity, 
            noise=Unoise(n_min=-0.05, n_max=0.05))
        
        # pose_commands = ObsTerm(
        #     func=mdp.generated_commands, 
        #     params={"command_name": "base_pose"})

        # For IEEE ACCESS R1
        # UPDATE: Observe the generated pose command with built-in accumulating drift
        pose_commands = ObsTerm(
            func=custom_mdp.pose_command_with_drift, 
            params={
                "command_name": "base_pose",
                "drift_std_xy": 0.01,
                "drift_std_yaw": 0.009
            }
        )

        
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
            noise=Unoise(n_min=-0.01, n_max=0.01))
        
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, 
            noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)
        
        time_to_target = ObsTerm(
            func=custom_mdp.time_remaining_normalized, 
            params={"command_name": "base_pose"}
        )

        # ... keep height_scan if utilizing the 2D LiDAR / depth data ...

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class Go1EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"), # Go1 uses "trunk" instead of "body"
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-0.1, 0.1),
            # "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


@configclass
class Go1RewardsCfg:
    """Reward terms for the MDP."""
    
    # -- penalties (Keep standard locomotion constraints)
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2, 
        weight=-2.0)
    # ang_vel_xy_l2 = RewTerm(
    #     func=mdp.ang_vel_xy_l2, 
    #     weight=-0.05)
    # dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, 
    #     weight=-1.0e-6) # Was -1.0e-5
    # dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, 
    #     weight=-1.25e-7)        # Was -2.5e-7

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, 
        weight=-0.01)   # Was -0.01

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10.0,
        params={
            # CHANGE: ".*THIGH" to ".*_thigh" 
            # (You can also add ".*_calf" or "trunk" if you want to penalize those hitting the ground)
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), 
            "threshold": 1.0
        },
    )

    is_alive = RewTerm(
        func=mdp.is_alive, 
        weight=0.1
    )

    # Penalize tilting. This forces the robot to fight to keep its back flat.
    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2, 
        weight=-5.0
    )

    feet_slide = RewTerm(
        func=custom_mdp.feet_slide,
        # weight=-0.5, 
        weight=0.0, 
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            # CHANGE: Add body_names here so it only grabs the 4 feet velocities!
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot") 
        }
    )

    # 1. The Magnet: Constant pull towards the target
    track_progress = RewTerm(
        func=custom_mdp.progress_towards_pose_target,
        weight=2.0, 
        params={"command_name": "base_pose", "asset_cfg": SceneEntityCfg("robot")}
    )

    # 1. Timed Trajectory Tracking (Replaces standard track_progress)
    track_timed_trajectory = RewTerm(
        # func=custom_mdp.track_timed_trajectory_local_exp_v2,
        func=custom_mdp.track_spline_trajectory_local_exp_v3,
        weight=4.0, 
        params={"command_name": "base_pose",
            "std": 0.5, 
            "asset_cfg": SceneEntityCfg("robot")}
    )


    # # 2. The Strict Parker (Kicks in hard when time is up and desired_vel drops to 0)
    # rel_pos_tracking = RewTerm(
    #     func=custom_mdp.park_and_face_target_exp,
    #     # weight=2.0, 
    #     weight=2.0, 
    #     params={"command_name": "base_pose", 
    #             # "pos_std": 0.25, 
    #             "pos_std": 0.05, 
    #             "yaw_std": 0.15}, 
    # )

    # 1. The Wide Net (Tolerates the 3-4cm sensor drift)
    rel_pos_tracking_wide = RewTerm(
        func=custom_mdp.park_and_face_target_exp,
        weight=2.0, 
        params={"command_name": "base_pose", 
                "pos_std": 0.10, # 10cm width catches the drifting observation
                "yaw_std": 0.15}, 
    )

    # 2. The Bullseye (Forces sub-centimeter physical accuracy)
    rel_pos_tracking_strict = RewTerm(
        func=custom_mdp.park_and_face_target_exp,
        weight=3.0, 
        params={"command_name": "base_pose", 
                "pos_std": 0.02, # Ultra-strict 2cm width
                "yaw_std": 0.05}, 
    )



    # Reward for facing the exact target angle
    track_yaw = RewTerm(
        func=custom_mdp.track_heading_exp, # Or wherever your track_heading_exp is located
        weight=4.0,  # Make this strong enough that the robot cares
        params={
            "command_name": "base_pose", 
            "std": 0.5, # Tight standard deviation (must be within ~2-3 degrees)
            "asset_cfg": SceneEntityCfg("robot")
        }
    )

    # -- Added for IEEE ACCESS R1
    zero_vel_arrival = RewTerm(
        func=custom_mdp.terminal_velocity_penalty, # Or wherever your track_heading_exp is located
        weight=-5.0,  
        params={
            "command_name": "base_pose", 
            "asset_cfg": SceneEntityCfg("robot")
        }
    )


    # # 1. Timed Trajectory Tracking (Replaces standard track_progress)
    # track_timed_trajectory = RewTerm(
    #     func=custom_mdp.track_timed_trajectory_local_exp,
    #     weight=4.0, 
    #     params={"command_name": "base_pose", "std": 0.5, "asset_cfg": SceneEntityCfg("robot")}
    # )


    # # 2. The Strict Parker (Kicks in hard when time is up and desired_vel drops to 0)
    # rel_pos_tracking = RewTerm(
    #     func=custom_mdp.park_and_face_target_exp,
    #     weight=2.0,   
    #     params={"command_name": "base_pose", "pos_std": 0.25, "yaw_std": 0.5}, 
    # )


    # 3. The 3-Legged Fix: Force the knees to bend
    nominal_pos_penalty = RewTerm(
        func=custom_mdp.nominal_joint_position_l2_penalty, 
        weight=-0.05, 
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # # 4. The Limp Fix: Reward high, distinct steps
    # feet_air_time = RewTerm(
    #     func=mdp.feet_air_time,
    #     weight=0.5, # Ensure this is strong enough to encourage picking all 4 feet up
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "command_name": "base_pose", 
    #         "threshold": 0.5,
    #     },
    # )

    # The Statue Penalty: Forces the robot to freeze solid when it reaches the goal
    stop_jitter = RewTerm(
        func=custom_mdp.stand_still_when_arrived,
        # weight=-0.005, # Very heavy penalty for vibrating
        weight=-0.05, # Very heavy penalty for vibrating
        params={"command_name": "base_pose", "asset_cfg": SceneEntityCfg("robot")}
    )

    arrived_on_time = RewTerm(
    func=custom_mdp.in_box_bonus,
    weight=2.0,   # dense now, so a SMALL weight — not 10
    params={
        "command_name": "base_pose",
        "threshold": 0.05,
        "yaw_threshold": 0.087,
        "height_target": 0.30,
        "height_tol": 0.04,
        },
    )

    # Penalize the robot for arriving too early
    too_early_penalty = RewTerm(
        func=custom_mdp.impatience_penalty,
        weight=-2.0, # Adjust this negative weight to make it "feel" the penalty
        params={"command_name": "base_pose", "threshold": 0.15}
    )


    # Force the robot to keep its belly off the ground
    base_height = RewTerm(
        func=mdp.base_height_l2, 
        weight=-30.0, # Strong penalty for crouching/falling
        # weight=0.0, # Strong penalty for crouching/falling
        params={"target_height": 0.35, "asset_cfg": SceneEntityCfg("robot")}
    )


    gait = RewTerm(
        func=custom_mdp.GaitReward,
        weight=2.0,
        params={
            "std": 0.025,
            "max_err": 0.05,
            # "velocity_threshold": 0.5,
            "distance_threshold": 0.05,
            # MC specific trot pairing: (Front-Left + Rear-Right) & (Front-Right + Rear-Left)
            "synced_feet_pair_names": (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")), 
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", 
                    body_names=".*foot"),
        },
    )

    foot_clearance = RewTerm(
        func=custom_mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
        },
    )

    action_smoothness = RewTerm(func=custom_mdp.action_smoothness_penalty, 
        weight=-1.0)


    joint_acc = RewTerm(
        func=custom_mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    # joint_vel = RewTerm(
    #     func=custom_mdp.joint_velocity_penalty,
    #     weight=-1.0e-2,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    # )

    joint_torques = RewTerm(
        func=custom_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )



@configclass
class Go1TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        # Terminate if the trunk, thighs, or calves hit the floor
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["trunk", ".*_thigh", ".*_calf"]), "threshold": 1.0},
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )

    # reached_goal = DoneTerm(
    #     func=custom_mdp.reached_goal_and_stopped, # You would need to write this
    #     params={
    #         "command_name": "base_pose",
    #         "distance_threshold": 0.1, # Within 10cm
    #         "velocity_threshold": 0.05 # Almost zero speed
    #     }
    # )

# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

#     terrain_levels = CurrTerm(func=custom_mdp.terrain_levels_pose)



@configclass
class Go1P2PRelTimeEnvCfg(ManagerBasedRLEnvCfg):

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)

    # Basic settings
    observations: Go1ObservationsCfg = Go1ObservationsCfg()
    actions: Go1ActionsCfg = Go1ActionsCfg()
    commands: Go1CommandsCfg = Go1CommandsCfg()

    # MDP setting
    rewards: Go1RewardsCfg = Go1RewardsCfg()
    terminations: Go1TerminationsCfg = Go1TerminationsCfg()
    events: Go1EventCfg = Go1EventCfg()
    # curriculum: CurriculumCfg = CurriculumCfg()

    # Viewer
    # viewer = ViewerCfg(eye=(4.5, 4.5, 4.5), origin_type="world", env_index=0, asset_name="robot")
    viewer = ViewerCfg(eye=(4.5, 4.5, 4.5), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 4  # 50 Hz control over 200Hz sim
        self.episode_length_s = 6.0
        # ==========================================================
        # 2. DISABLE THE COMMAND TIMER (The Target)
        # ==========================================================
        # By setting the resample time to 10,000 seconds, the target 
        # will NEVER change on its own. It will ONLY change when the 
        # 6.0-second episode timer runs out and triggers a hard reset.
        self.commands.base_pose.resampling_time_range = (10000.0, 10000.0)
        # ==========================================================

        self.observations.policy.enable_corruption = False


        # simulation settings
        self.sim.dt = 0.005  # 200 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        self.scene.contact_forces.update_period = self.sim.dt

        # switch robot to Go1
        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


        # # terrain (Flat)
        # self.scene.terrain = TerrainImporterCfg(
        #     prim_path="/World/ground",
        #     terrain_type="generator",
        #     terrain_generator=COBBLESTONE_ROAD_CFG,
        #     max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        #     collision_group=-1,
        #     physics_material=sim_utils.RigidBodyMaterialCfg(
        #         friction_combine_mode="multiply",
        #         restitution_combine_mode="multiply",
        #         static_friction=1.0,
        #         dynamic_friction=1.0,
        #     ),
        #     visual_material=sim_utils.MdlFileCfg(
        #         mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
        #         project_uvw=True,
        #         texture_scale=(0.25, 0.25),
        #     ),
        #     debug_vis=True,
        # )

        # # no height scan for flat ground
        # # self.scene.height_scanner = None

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane", # This tells Isaac Lab to use an infinite flat floor
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            # Keep the visual material so you have a nice grid/texture to look at
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )

        # Ensure the height scanner is disabled since a flat plane doesn't need to be scanned
        self.scene.height_scanner = None

        # ==========================================================
        # [NEW] OVERRIDE INHERITED COMMAND DEFAULTS
        # ==========================================================
        import math
        # Loop through all inherited commands and force 360-degree headings
        for cmd_name, cmd_cfg in vars(self.commands).items():
            if hasattr(cmd_cfg, "simple_heading"):
                cmd_cfg.simple_heading = False
            if hasattr(cmd_cfg, "ranges"):
                cmd_cfg.ranges.heading = (-math.pi, math.pi)

@configclass
class Go1P2PRelTimeEnvCfg_PLAY(Go1P2PRelTimeEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        
        # ==========================================================
        # 1. FLAT PLANE & TOP-DOWN CAMERA
        # ==========================================================
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        
        from isaaclab.envs import ViewerCfg
        self.viewer = ViewerCfg(eye=(0.0, 0.00, 12.0), lookat=(0.0, 0.0, 0.0))

        # ==========================================================
        # 2. BASIC PLAYBACK SETTINGS
        # ==========================================================
        self.scene.num_envs = 1
        self.episode_length_s = 5.0 
        self.observations.policy.enable_corruption = False

        # ==========================================================
        # 3. FORCE EXPLICIT COMMAND GENERATION
        # ==========================================================
        # from isaaclab.envs import mdp
        from . import custom_mdp
        import math
        
        # MUST use the 5D LocalPoseTimeCommandCfg, NOT the default mdp one!
        self.commands.base_pose = custom_mdp.LocalPoseTimeCommandCfg(
            class_type=custom_mdp.LocalPoseTimeCommand,
            resampling_time_range=(10000.0, 10000.0), 
            ranges=custom_mdp.LocalPoseTimeCommandCfg.Ranges(
                pos_x=(-1.5, 1.5),
                pos_y=(-1.5, 1.5),
                heading=(-math.pi, math.pi),
                # time=(2.5, 4.0) # <--- The missing 5th dimension!
                # time=(1.5, 3.0) # <--- The missing 5th dimension!
                time=(1.2, 2.5) # <--- The missing 5th dimension!
            )
        )

        # ==========================================================
        # 4. FORCE PERFECT SPAWN AT X=0, Y=0, YAW=0
        # ==========================================================
        from isaaclab.managers import EventTermCfg as EventTerm
        from isaaclab.managers import SceneEntityCfg

        self.events.reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "pose_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)},
                "velocity_range": {
                    "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                    "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
                },
            },
        )

        self.events.reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (1.0, 1.0), 
                "velocity_range": (0.0, 0.0),
            },
        )
        
        if hasattr(self.events, "base_external_force_torque"):
            self.events.base_external_force_torque = None
        if hasattr(self.events, "push_robot"):
            self.events.push_robot = None


        # import math
        # for cmd_name, cmd_cfg in vars(self.commands).items():
        #     if hasattr(cmd_cfg, "simple_heading"):
        #         cmd_cfg.simple_heading = False
        #     if hasattr(cmd_cfg, "ranges"):
        #         cmd_cfg.ranges.heading = (-math.pi, math.pi)