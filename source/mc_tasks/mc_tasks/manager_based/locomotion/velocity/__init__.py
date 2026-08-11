import gymnasium as gym
from . import agents  # This imports the hyperparameters for the RL algorithm

# Register the Training Environment
gym.register(
    id="Isaac-MC-Velocity-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mc_velocity_flat_env_cfg:MCVelocityFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MCFlatPPORunnerCfg",
    },
)

# Register the Play/Visualization Environment
gym.register(
    id="Isaac-MC-Velocity-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mc_velocity_flat_env_cfg:MCVelocityFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:MCFlatPPORunnerCfg",
    },
)