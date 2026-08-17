# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
Standalone MC quadruped velocity-tracking (flat) environment.

This file does NOT inherit the stock LocomotionVelocityRoughEnvCfg. That base
class carries Go1/ANYmal body names ("base", ".*FOOT", ".*THIGH") in its default
EventCfg / RewardsCfg / scene, which fail to resolve against the MC robot bodies
(Body, Hip0-3, UpperLeg0-3, LowerLeg0-3). Inheriting ManagerBasedRLEnvCfg
directly means only the MC-specific config groups below are ever instantiated.
"""

import sys
import os
import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))
from data.robots.mc_quadruped.mc_quadruped_cfg import MC_QUADRUPED_CFG

from . import custom_mdp as custom_mdp
from . import mdp as mdp

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


##
# Terrain
##

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


##
# Scene
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with the MC quadruped."""

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

    # robot (MC quadruped)
    robot = MC_QUADRUPED_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # contact sensor (matches the proven P2P path)
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/QuadrupedRobot/(Body|UpperLeg.*|LowerLeg.*/Foot/Contact"),
        update_period=0.005,
        history_length=3,
        track_air_time=True,
        # force_threshold=20.0,
    )

    # flat ground -> no height scanner
    height_scanner = None

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class MCActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], 
        scale=0.30,
        use_default_offset=True
    )


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
        lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-0.8, 0.8)
        # lin_vel_x=(-1.5, 1.5), lin_vel_y=(-0.75, 0.75), ang_vel_z=(-3.14, 3.14)
        # lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0), ang_vel_z=(-0.8, 0.8)
        ),
    )


@configclass
class MCObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0),
        )

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-100.0, 100.0)
        )

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0), # Added
        )

        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, 
            params={"asset_cfg": SceneEntityCfg("robot")}, 
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0), # Added

        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")}, noise=Unoise(n_min=-0.5, n_max=0.5),
            clip=(-100.0, 100.0)
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
            "asset_cfg": SceneEntityCfg("robot", body_names="Body"),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )

    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Body"),
            # "force_range": (-5.0, 5.0),
            # "torque_range": (-1.0, 1.0),
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
                # "x": (-1.5, 1.5), "y": (-1.0, 1.0), "z": (0.0, 0.0),
                # "roll": (-0.7, 0.7), "pitch": (-0.7, 0.7), "yaw": (-1.0, 1.0),
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),            },
        },
    )

    reset_robot_joints = EventTerm(
        func=custom_mdp.reset_joints_around_default,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            # "velocity_range": (-2.5, 2.5),
            "velocity_range": (-0.1, 0.1),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        # interval_range_s=(3.0, 5.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )


@configclass
class MCRewardsCfg:
    # -- task
    # -- task
    track_lin_vel_xy_exp = RewardTermCfg(
        func=mdp.track_lin_vel_xy_exp, 
        # weight=1.0, 
        weight=4.0, 
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    track_ang_vel_z_exp = RewardTermCfg(
        func=mdp.track_ang_vel_z_exp, 
        # weight=0.5, 
        weight=2.0, 
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- penalties
    lin_vel_z_l2 = RewardTermCfg(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewardTermCfg(func=mdp.ang_vel_xy_l2, weight=-0.05)
    dof_torques_l2 = RewardTermCfg(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewardTermCfg(func=mdp.action_rate_l2, 
        # weight=-0.01)
        weight=-0.005)

    feet_air_time = RewardTermCfg(
        func=mdp.feet_air_time,
        # weight=0.125,
        weight=0.3,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="LowerLeg.*/Foot/Contact"),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    undesired_contacts = RewardTermCfg(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="UpperLeg.*"), 
            "threshold": 1.0},
    )
    # -- optional penalties
    flat_orientation_l2 = RewardTermCfg(
        func=mdp.flat_orientation_l2, 
        weight=-5.0)

    dof_pos_limits = RewardTermCfg(
        func=mdp.joint_pos_limits, 
        weight=0.0)

    base_height_crouch = RewardTermCfg(
        func=custom_mdp.base_height_below_target_penalty,
        # weight=-80.0,
        weight=-150.0,
        params={"target_height": 0.32, "asset_cfg": SceneEntityCfg("robot")},
    )

    hip_deviation = RewardTermCfg(
        func=mdp.joint_deviation_l1,
        # weight=-0.2,   # start modest; raise if stance stays narrow
        weight=-0.0,   # start modest; raise if stance stays narrow
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["q0", "q3", "q6", "q9"])},
    )

    gait = RewardTermCfg(
        func=custom_mdp.GaitReward,   # your P2P GaitReward, or the Spot one in your custom_mdp
        weight=3.0,                   # start moderate; task terms must still dominate
        params={
            "std": 0.025,
            "max_err": 0.05,
            "velocity_threshold": 0.1,
            "synced_feet_pair_names": (("LowerLeg0", "LowerLeg3"), ("LowerLeg1", "LowerLeg2")),
            "asset_cfg": SceneEntityCfg("robot", body_names="LowerLeg.*/Foot/Contact"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="LowerLeg.*/Foot/Contact"),
        },
    )

    foot_clearance = RewardTermCfg(
        func=custom_mdp.foot_clearance_reward,
        weight=0.5,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names="LowerLeg.*/Foot/Contact"),
        },
    )


@configclass
class MCTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Body", "UpperLeg.*"]), "threshold": 1.0},
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class MCCurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


##
# Environment configuration
##


@configclass
class MCVelocityFlatEnvCfg(ManagerBasedRLEnvCfg):
    """MC quadruped velocity-tracking on flat terrain (standalone, no Go1 base)."""

    # Scene
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)

    # MDP groups
    observations: MCObservationsCfg = MCObservationsCfg()
    actions: MCActionsCfg = MCActionsCfg()
    commands: MCCommandsCfg = MCCommandsCfg()
    rewards: MCRewardsCfg = MCRewardsCfg()
    terminations: MCTerminationsCfg = MCTerminationsCfg()
    events: MCEventCfg = MCEventCfg()
    # curriculum: MCCurriculumCfg = MCCurriculumCfg()
    curriculum: MCCurriculumCfg = None

    # Viewer
    viewer = ViewerCfg(eye=(4.5, 4.5, 4.5), origin_type="world", env_index=0, asset_name="robot")

    def __post_init__(self):
        """Post initialization (formerly provided by the base class)."""
        # general settings
        self.episode_length_s = 20.0

        # simulation settings
        self.sim.dt = 0.005          # 200 Hz
        self.decimation = 4          
        # self.sim.dt = 0.002          # 200 Hz
        # self.decimation = 10          

        # Tame PhysX at lower frequencies to prevent NaN explosions
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.max_depenetration_velocity = 1.0 # Crucial: stops the robot from being launched to infinity

        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # sensor update periods tick at the physics rate
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply", restitution_combine_mode="multiply",
                static_friction=1.0, dynamic_friction=1.0,
            ),
            debug_vis=False,
        )

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

        # keep sim material in sync with the (new) terrain material
        self.sim.physics_material = self.scene.terrain.physics_material

        # enable terrain curriculum if present
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True


@configclass
class MCVelocityFlatEnvCfg_PLAY(MCVelocityFlatEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()

        # smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.push_robot = None
        self.curriculum = None 
