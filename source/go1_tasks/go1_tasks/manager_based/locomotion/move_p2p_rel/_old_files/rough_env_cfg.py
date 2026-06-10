# File: source/extensions/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go1/rough_env_cfg.py

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg
from isaaclab_assets.robots.unitree import UNITREE_GO1_CFG
import isaaclab.envs.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm 
from isaaclab.managers import SceneEntityCfg

@configclass
class UnitreeGo1RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        if hasattr(self.observations.policy, "commands"):
                    # [v_x, v_y, w_z] -> Scale w_z UP so it's "loud"
                    self.observations.policy.commands.scale = [2.0, 2.0, 1.0]
                    
        # ============================================================
        # 1. PHYSICS: THE "SPOT" STANDARD
        # ============================================================
        self.scene.robot = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/trunk"
        
        # ACTION SCALE: 0.50 (Matches Spot)
        # This gives the Go1 the "mechanical authority" to turn sharply.
        self.actions.joint_pos.scale = 0.30
        
        # 1. Force Linear Velocity to ZERO
        # The robot is NOT ALLOWED to walk forward or sideways.
        # It must learn to turn in place.
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0) 
        self.commands.base_velocity.ranges.lin_vel_y = (1.0, 1.0)

        self.commands.base_velocity.ranges.ang_vel_z = (-2.5, 2.5)

        # ============================================================
        # 2. CRASH FIXES (Curriculum & Names)
        # ============================================================
        
        # --- FIX 1: Body Names ---
        self.events.push_robot = None
        self.events.add_base_mass.params["asset_cfg"].body_names = "trunk"
        self.events.base_external_force_torque.params["asset_cfg"].body_names = "trunk"
        
        # --- FIX 2: Disable Curriculum (Prevents "NoneType" Crash) ---
        # If we use a flat patch or simple rough terrain, we don't need levels.
        # This stops the code from looking for "terrain.cfg.terrain_generator.size"
        self.curriculum.terrain_levels = None 

        # ============================================================
        # 3. REWARDS: UNCHAINED
        # ============================================================
        
        # Fix Regex for Go1 (lowercase)
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = ".*_foot"
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ".*_thigh"

        # Turn Dominance (2.5 vs 1.0)
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=mdp.track_lin_vel_xy_exp,
            weight=1.0, 
            params={"command_name": "base_velocity", "std": 1.0},
        )
        self.rewards.track_ang_vel_z_exp = RewTerm(
            func=mdp.track_ang_vel_z_exp,
            weight=2.5,
            params={"command_name": "base_velocity", "std": 1.0},
        )
        
        # RELAXED LIMITS (The "Spot Freedom")
        # -0.1 allows the robot to reach wide angles for turning.
        self.rewards.dof_pos_limits.weight = -0.1 
        
        # ALLOW LEANING
        self.rewards.ang_vel_xy_l2.weight = -0.001 

        # Standard Penalties
        self.rewards.dof_torques_l2.weight = -0.000025
        self.rewards.action_rate_l2.weight = -0.01 
        self.rewards.feet_air_time.weight = 0.5


        # Terminations
        self.terminations.base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="trunk"), "threshold": 1.0},
        )

@configclass
class UnitreeGo1RoughEnvCfg_PLAY(UnitreeGo1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None