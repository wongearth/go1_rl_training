# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo1RoughEnvCfg
from isaaclab.managers import RewardTermCfg as RewTerm
import isaaclab.envs.mdp as mdp

@configclass
class UnitreeGo1FlatEnvCfg(UnitreeGo1RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # ---------------------------------------------------------
        # FIX FOR HEIGHT (The correct way)
        # ---------------------------------------------------------
        # We use "=" to create the reward term from scratch, 
        # because it didn't exist in the parent config.
        self.rewards.base_height_l2 = RewTerm(
            func=mdp.base_height_l2,
            weight=-2.0,  
            params={"target_height": 0.38}, 
        )
        # ---------------------------------------------------------

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class UnitreeGo1FlatEnvCfg_PLAY(UnitreeGo1FlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
