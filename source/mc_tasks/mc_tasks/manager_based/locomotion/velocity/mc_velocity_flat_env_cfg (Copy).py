# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import sys
import os
# Add the 'source/mc_tasks' directory to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from data.robots.mc_quadruped.mc_quadruped_cfg import MC_QUADRUPED_CFG
import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.envs import ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.actuators import ImplicitActuatorCfg # Need to disable Actuator Net that cause limping robot 
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


# Reusing the advanced math specifically designed for Spot's gait
# import isaaclab_tasks.manager_based.locomotion.velocity.config.spot.mdp as spot_mdp
# import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
# from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from . import custom_mdp as custom_mdp
from . import mdp as mdp
from .velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
# from isaaclab_assets.robots.unitree import UNITREE_MC_CFG  # isort: skip

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
    robot = MC_QUADRUPED_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # sensors
    # height_scanner = RayCasterCfg(
    #     prim_path="{ENV_REGEX_NS}/Robot/base",
    #     offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    #     ray_alignment="yaw",
    #     pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    #     debug_vis=False,
    #     mesh_prim_paths=["/World/ground"],
    # )

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/QuadrupedRobot/(Body|UpperLeg.*|LowerLeg.*)",
        update_period=0.005,
        history_length=3,
        track_air_time=True,
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
class MCActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.30, use_default_offset=True)


@configclass
class MCCommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # Fixed the Y-axis bug here to stop permanent strafing
            # lin_vel_x=(-1.0, 2.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-2.0, 2.0)
            # lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.2, 0.2), ang_vel_z=(-0.5, 0.5)
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.8, 0.8)
        ),
    )


@configclass
class MCObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1)
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class MCEventCfg:
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
            "asset_cfg": SceneEntityCfg("robot", body_names="Body"), # MC uses "Body" instead of "body"
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Body"),
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
                "x": (-1.5, 1.5), "y": (-1.0, 1.0), "z": (-0.5, 0.5),
                "roll": (-0.7, 0.7), "pitch": (-0.7, 0.7), "yaw": (-1.0, 1.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=custom_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (-2.5, 2.5),
            "asset_cfg": SceneEntityCfg("robot"),
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
class MCRewardsCfg:
    # -- task
    air_time = RewardTermCfg(
        func=custom_mdp.air_time_reward,
        weight=5.0,
        params={
            "mode_time": 0.3,
            "velocity_threshold": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces",
                    body_names="UpperLeg.*"), 
            "threshold": 2.0
        },
    )
    base_angular_velocity = RewardTermCfg(
        func=custom_mdp.base_angular_velocity_reward,
        weight=5.0,
        params={"std": 2.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_linear_velocity = RewardTermCfg(
        func=custom_mdp.base_linear_velocity_reward,
        weight=5.0,
        params={"std": 1.0, "ramp_rate": 0.5, "ramp_at_vel": 1.0, "asset_cfg": SceneEntityCfg("robot")},
    )
    foot_clearance = RewardTermCfg(
        func=custom_mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names="LowerLeg.*"),
        },
    )
    gait = RewardTermCfg(
        func=custom_mdp.GaitReward,
        weight=10.0,
        params={
            "std": 0.025,
            "max_err": 0.05,
            "velocity_threshold": 0.5,
            # MC specific trot pairing: (Front-Left + Rear-Right) & (Front-Right + Rear-Left)
            "synced_feet_pair_names": (("LowerLeg0", "LowerLeg3"), ("LowerLeg1", "LowerLeg2")), 
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", 
                    body_names="LowerLeg.*"),
        },
    )

    # -- penalties
    action_smoothness = RewardTermCfg(func=custom_mdp.action_smoothness_penalty, weight=-1.0)

    # air_time_variance = RewardTermCfg(
    #     func=custom_mdp.air_time_variance_penalty,
    #     weight=-1.0,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot")},
    # )
    base_motion = RewardTermCfg(
        func=custom_mdp.base_motion_penalty, 
        # weight=-5.0, 
        weight=-1.0, 
        params={"asset_cfg": SceneEntityCfg("robot")}
    )
    base_orientation = RewardTermCfg(
        func=custom_mdp.base_orientation_penalty, weight=-3.0, params={"asset_cfg": SceneEntityCfg("robot")}
    )
    foot_slip = RewardTermCfg(
        func=custom_mdp.foot_slip_penalty,
        # weight=-0.5,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="LowerLeg.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="LowerLeg.*"),
            "threshold": 1.0,
        },
    )
    # joint_acc = RewardTermCfg(
    #     func=custom_mdp.joint_acceleration_penalty,
    #     weight=-1.0e-4,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    # )

    joint_acc = RewardTermCfg(
        func=custom_mdp.joint_acceleration_penalty,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    # ...
    joint_vel = RewardTermCfg(
        func=custom_mdp.joint_velocity_penalty,
        weight=-1.0e-2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    joint_torques = RewardTermCfg(
        func=custom_mdp.joint_torques_penalty,
        weight=-5.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )

    # # Default weight is usually around 1.0 or 0.125 depending on the base class. 
    # feet_air_time = RewardTermCfg(
    #     func=mdp.feet_air_time,
    #     weight=0.2,  
    #     params={
    #         # Changed to body_names!
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "command_name": "base_velocity",
    #         "threshold": 0.4,
    #     },
    # )

    # 2. THE TWITCH FIX (Action Rate)
    # # Default weight is usually -0.01. 
    # # Making it more negative heavily penalizes the neural network for sending
    # action_rate_l2 = RewardTermCfg(
    #     func=mdp.action_rate_l2,
    #     weight=-0.05, # <--- MAKE THIS MORE NEGATIVE
    # )

    # Penalize high energy usage
    dof_torques_l2 = RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-0.0002, # Make this slightly more negative to encourage efficient, flowing movement
    )

    # # The Root-Cause Limping Fix: Penalize asymmetric air times!
    # air_time_variance = RewardTermCfg(
    #     func=custom_mdp.rewards.feet_air_time_variance, # <--- Pointing to YOUR new custom function!
    #     weight=-5.0, # Make this heavily negative
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
    #         "command_name": "base_velocity",
    #     },
    # )

    # # Penalize slamming the feet into the ground
    # contact_force_spikes = RewardTermCfg(
    #     func=custom_mdp.contact_forces_penalty,
    #     weight=-0.01, 
    #     params={
    #         "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"), 
    #         "max_contact_force": 300.0, # Adjust this based on MC's weight. If it still limps, lower this to 200.0
    #     },
    # )


@configclass
class MCTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        # Terminate if the trunk, thighs, or calves hit the floor
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Body", "UpperLeg.*"]), "threshold": 1.0},
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class MCVelocityFlatEnvCfg(LocomotionVelocityRoughEnvCfg):

    # Basic settings
    observations: MCObservationsCfg = MCObservationsCfg()
    actions: MCActionsCfg = MCActionsCfg()
    commands: MCCommandsCfg = MCCommandsCfg()

    # MDP setting
    rewards: MCRewardsCfg = MCRewardsCfg()
    terminations: MCTerminationsCfg = MCTerminationsCfg()
    events: MCEventCfg = MCEventCfg()

    # Viewer
    viewer = ViewerCfg(eye=(4.5, 4.5, 4.5), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # general settings
        self.decimation = 4  # 50 Hz control over 200Hz sim
        self.episode_length_s = 20.0

        # simulation settings
        self.sim.dt = 0.005  # 200 Hz
        self.sim.render_interval = self.decimation
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "multiply"
        self.sim.physics_material.restitution_combine_mode = "multiply"
        self.scene.contact_forces.update_period = self.sim.dt

        # switch robot to MC
        self.scene.robot = MC_QUADRUPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # # Kill the Actuator Net and force the MC to use standard Ideal PD motors
        # self.scene.robot.actuators = {
        #     "legs": ImplicitActuatorCfg(
        #         joint_names_expr=[".*"], # Apply to all joints
        #         stiffness=20.0,          # Standard MC Kp
        #         damping=0.5,             # Standard MC Kd
        #         effort_limit=23.7,       # MC max torque limit
        #         velocity_limit=30.0,     # MC max velocity limit
        #     )
        # }


        # terrain (Flat)
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=COBBLESTONE_ROAD_CFG,
            max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
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
            debug_vis=True,
        )

        # no height scan for flat ground
        self.scene.height_scanner = None


class MCVelocityFlatEnvCfg_PLAY(MCVelocityFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None

        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.push_robot = None