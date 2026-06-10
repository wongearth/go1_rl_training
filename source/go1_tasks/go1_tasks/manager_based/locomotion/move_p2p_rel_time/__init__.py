import gymnasium as gym

gym.register(
    id="Isaac-Go1-P2P_Rel_Time-v0",  
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # This now correctly points to your new unified file and class
        "env_cfg_entry_point": f"{__name__}.p2p_rel_time_env_cfg:Go1P2PRelTimeEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UnitreeGo1RoughPPORunnerCfg",
    },
)

# 2. ADD THIS: Your Play Environment
gym.register(
    id="Isaac-Go1-P2P_Rel_Time-Play-v0",  # Notice the "-Play-" addition
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        # Notice this points to the PLAY class!
        "env_cfg_entry_point": f"{__name__}.p2p_rel_time_env_cfg:Go1P2PRelTimeEnvCfg_PLAY", 
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:UnitreeGo1RoughPPORunnerCfg",
    },
)