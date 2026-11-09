play_vel_p2p_single_replanning.py [-h] [--video] [--video_length VIDEO_LENGTH]
                                         [--disable_fabric] [--num_envs NUM_ENVS]
                                         [--task TASK] [--agent AGENT] [--seed SEED]
                                         [--use_pretrained_checkpoint] [--real-time]
                                         [--max_steps MAX_STEPS]
                                         [--plot_mode {save,live,both}]
                                         [--export_csv EXPORT_CSV]
                                         [--ortho_size ORTHO_SIZE]
                                         [--target_time_range TARGET_TIME_RANGE TARGET_TIME_RANGE]
                                         [--target_x_range TARGET_X_RANGE TARGET_X_RANGE]
                                         [--target_y_range TARGET_Y_RANGE TARGET_Y_RANGE]
                                         [--target_yaw_range TARGET_YAW_RANGE TARGET_YAW_RANGE]
                                         [--hold_time HOLD_TIME]
                                         [--max_lin_vel MAX_LIN_VEL]
                                         [--max_ang_vel MAX_ANG_VEL]
                                         [--num_episodes NUM_EPISODES]
                                         [--commands_csv COMMANDS_CSV]
                                         [--commands_seed COMMANDS_SEED]
                                         [--results_csv RESULTS_CSV]
                                         [--play_episode PLAY_EPISODE] [--cycle] [--once]
                                         [--append] [--random_targets]
                                         [--target_seed TARGET_SEED]
                                         [--noise_model {none,gaussian,drift}]
                                         [--noise_k NOISE_K] [--sigma_xy SIGMA_XY]
                                         [--sigma_yaw SIGMA_YAW]
                                         [--drift_xy_per_s DRIFT_XY_PER_S]
                                         [--drift_yaw_per_s DRIFT_YAW_PER_S]
                                         [--noise_seed NOISE_SEED]
                                         [--ts_episode TS_EPISODE]

Play a velocity policy as a P2P timed reach with cubic-spline velocity profiles.

optional arguments:
  -h, --help            show this help message and exit
  --video               Record videos during training. (default: False)[cite: 6]
  --video_length VIDEO_LENGTH
                        Length of the recorded video (in steps). (default: 200)[cite: 6]
  --disable_fabric      Disable fabric and use USD I/O operations. (default: False)[cite: 6]
  --num_envs NUM_ENVS   Number of environments to simulate (default 1 for single-robot viewing). (default: 1)[cite: 6]
  --task TASK           Name of the task. (default: None)[cite: 6]
  --agent AGENT         Name of the RL agent configuration entry point. (default: rsl_rl_cfg_entry_point)[cite: 6]
  --seed SEED           Seed used for the environment (default: None)[cite: 6]
  --use_pretrained_checkpoint
                        Use the pre-trained checkpoint from Nucleus. (default: False)[cite: 6]
  --real-time           Run in real-time, if possible. (default: False)[cite: 6]
  --max_steps MAX_STEPS
                        Total steps to run before auto-exiting (default: video_length+50 if --video, else 100000). (default: None)[cite: 6]
  --plot_mode {save,live,both}
                        How to output the per-episode error plot: 'save' writes PNG files (default, headless-safe), 'live' shows an interactive updating window instead, 'both' does both. (default: save)[cite: 6]
  --export_csv EXPORT_CSV
                        If set, write per-step time-series (episode, time, err_x/y/yaw, height, roll, pitch) to this CSV. (default: "")[cite: 6]
  --ortho_size ORTHO_SIZE
                        If > 0, use an orthographic top-down camera with this vertical view span in meters (e.g. 3.0 tight on the robot, 12.0 wide). 0 = leave the perspective camera as-is. (default: 0.0)[cite: 6]
  --target_time_range TARGET_TIME_RANGE TARGET_TIME_RANGE
                        Random per-episode time budget (s) for each cubic-spline reach; sampled uniformly. (default: [2.5, 4.0])[cite: 6]
  --target_x_range TARGET_X_RANGE TARGET_X_RANGE
                        Target x range (m, world/base-init frame). (default: [-2.0, 2.0])[cite: 6]
  --target_y_range TARGET_Y_RANGE TARGET_Y_RANGE
                        Target y range (m). (default: [-2.0, 2.0])[cite: 6]
  --target_yaw_range TARGET_YAW_RANGE TARGET_YAW_RANGE
                        Target yaw range (rad). (default: [-3.141592653589793, 3.141592653589793])[cite: 6]
  --hold_time HOLD_TIME
                        Extra time (s) held at zero command after the reach, before resampling a new target. (default: 1.0)[cite: 6]
  --max_lin_vel MAX_LIN_VEL
                        Clamp on planned linear velocity command magnitude (m/s). (default: 1.0)[cite: 6]
  --max_ang_vel MAX_ANG_VEL
                        Clamp on planned yaw-rate command (rad/s). (default: 1.0)[cite: 6]
  --num_episodes NUM_EPISODES
                        Number of episodes to run before exiting (batch mode). (default: 100)[cite: 6]
  --commands_csv COMMANDS_CSV
                        Shared command file; created (seeded) if missing, else reused. (default: p2p_commands.csv)[cite: 6]
  --commands_seed COMMANDS_SEED
                        Seed used ONLY when creating the shared command file. (default: 42)[cite: 6]
  --results_csv RESULTS_CSV
                        Per-episode results output for this (velocity) run. (default: results_velocity.csv)[cite: 6]
  --play_episode PLAY_EPISODE
                        Play ONLY this row index from the commands CSV, looping it forever so you can watch it. Omit to play all rows once (or use --cycle to loop through all rows forever). (default: None)[cite: 6]
  --cycle               Cycle through all rows 0->end->0 forever (ignored if --play_episode is set). (default: False)[cite: 6]
  --once                With --play_episode N: play row N exactly once, write its result, and exit (instead of looping). Use in a shell loop for one-fresh-process-per-episode collection. (default: False)[cite: 6]
  --append              Append to --results_csv instead of overwriting (write header only if file is new). Lets many --once runs accumulate into one results file. (default: False)[cite: 6]
  --random_targets      Sample targets randomly from the ranges (seeded by --seed) instead of reading --commands_csv. Like the P2P play script's default behavior. (default: False)[cite: 6]
  --target_seed TARGET_SEED
                        RNG seed for --random_targets (change for a different sequence). (default: 42)[cite: 6]
  --noise_model {none,gaussian,drift}
                        Noise model on the observation. (default: none)[cite: 6]
  --noise_k NOISE_K     Scale multiplier on base sigmas. (default: 1.0)[cite: 6]
  --sigma_xy SIGMA_XY   Base standard deviation in meters for dx/dy channels. (default: 0.02)[cite: 6]
  --sigma_yaw SIGMA_YAW
                        Base standard deviation in radians for heading. (default: 0.026)[cite: 6]
  --drift_xy_per_s DRIFT_XY_PER_S
                        Drift random-walk standard deviation for dx/dy. (default: 0.01)[cite: 6]
  --drift_yaw_per_s DRIFT_YAW_PER_S
                        Drift random-walk standard deviation for heading. (default: 0.009)[cite: 6]
  --noise_seed NOISE_SEED
                        Base seed for the noise RNG. (default: 0)[cite: 6]
  --ts_episode TS_EPISODE
                        If set, ONLY save timeseries data for this specific episode index (prevents giant CSVs). (default: None)[cite: 6]
                        





collect_via_single_replanning.sh 
usage: bash collect_via_single_replanning.sh TASK NOISE_MODEL NOISE_K [NUM] [CHECKPOINT] [RESULTS_CSV] [COMMANDS_CSV]

Collect velocity-based Point-to-Point (P2P) results with noise injection, running one episode per fresh process.

positional arguments (required):
  TASK              The Isaac Lab task ID to run (e.g., Isaac-Go1-Velocity-Flat-Play-v0)
  NOISE_MODEL       The observation noise model to apply. Choices: {none, gaussian, drift}
  NOISE_K           Scale multiplier on the base noise sigmas (e.g., 0, 1, 2)

optional arguments (positional):
  NUM               Number of batch mode episodes to run. (default: 100)
  CHECKPOINT        Path to a specific model checkpoint to load. (default: "" [auto-loads latest])
  RESULTS_CSV       Path to save the per-episode results CSV. (default: results_velocity_replanning_${NOISE_MODEL}_k${NOISE_K}.csv)
  COMMANDS_CSV      Path to the shared command file to read targets from. (default: p2p_commands.csv)

  -h, --help        show this help message and exit

Examples:
  # Run clean baseline for 100 episodes
  bash collect_via_single_replanning.sh Isaac-Go1-Velocity-Flat-Play-v0 none 0

  # Run drift noise (k=2) for 50 episodes with custom output files
  bash collect_via_single_replanning.sh Isaac-Go1-Velocity-Flat-Play-v0 drift 2 50 "" custom_results.csv custom_cmds.csv