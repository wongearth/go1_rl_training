
# IEEE ACCESS data collection R1

# DATA COLLECTION: 

python generate_commands.py --num_episodes 100 --seed 42 --out p2p_commands.csv --x_range -1.5 1.5 --y_range -0.75 0.75 --yaw_range -3.14 3.14 --time_range 1.5 3.5
# --> files produced 100 episodes of commands in the file named p2p_commands.csv 
# --> p2p_commands.csv 

# To play velocity replanning policy 
python scripts/rsl_rl/play_vel_p2p_single_replanning.py --task Isaac-Go1-Velocity-Flat-Play-v0 --num_envs 1
# --> this will show the results for each run

# ========= Collect data Velocity replanning policy with noise drifts   
# <--- the command above need file play_vel_p2p_single_replanning.py
# without noise
bash scripts/rsl_rl/collect_via_single_replanning.sh Isaac-Go1-Velocity-Flat-Play-v0 "" 100 && 
# --> files produced
# --> result_vel_replanning.csv

# #with noise 
bash scripts/rsl_rl/collect_via_single_replanning.sh Isaac-Go1-Velocity-Flat-Play-v0 gaussian 1 100 "" &&
bash scripts/rsl_rl/collect_via_single_replanning.sh Isaac-Go1-Velocity-Flat-Play-v0 gaussian 2 100 "" &&
bash scripts/rsl_rl/collect_via_single_replanning.sh Isaac-Go1-Velocity-Flat-Play-v0 gaussian 3 100 "" &&
# --> files produced
# --> result_vel_replanning_noise_k1.csv
# --> result_vel_replanning_noise_k2.csv
# --> result_vel_replanning_noise_k3.csv

# with drifts
bash scripts/rsl_rl/collect_via_single_replanning.sh Isaac-Go1-Velocity-Flat-Play-v0 drift 1 100 "" &&
bash scripts/rsl_rl/collect_via_single_replanning.sh Isaac-Go1-Velocity-Flat-Play-v0 drift 2 100 "" &&
bash scripts/rsl_rl/collect_via_single_replanning.sh Isaac-Go1-Velocity-Flat-Play-v0 drift 3 100 ""
# --> files produced
# --> result_vel_replanning_drift_k1.csv
# --> result_vel_replanning_drift_k2.csv
# --> result_vel_replanning_drift_k3.csv

# ========= Collect data P2P-T policy with noise
# <-- need file play_p2pt_single.py
bash scripts/rsl_rl/collect_p2pt_noise.sh Isaac-Go1-P2P_Rel_Time-Play-v0 none 0 100 "" results_p2p_clean.csv &&
bash scripts/rsl_rl/collect_p2pt_noise.sh Isaac-Go1-P2P_Rel_Time-Play-v0 all 1 100 "" results_p2p_allsensors_k1.csv &&
bash scripts/rsl_rl/collect_p2pt_noise.sh Isaac-Go1-P2P_Rel_Time-Play-v0 all 2 100 "" results_p2p_allsensors_k2.csv &&
bash scripts/rsl_rl/collect_p2pt_noise.sh Isaac-Go1-P2P_Rel_Time-Play-v0 all 3 100 "" results_p2p_allsensors_k3.csv
# --> files produced
# --> result_p2pt.csv
# --> result_p2pt_allsensors_k1.csv
# --> result_p2pt_allsensors_k2.csv
# --> result_p2pt_allsensors_k3.csv

# ========= Collect data P2P-T policy with drifts
# <-- need file play_p2pt_single.py
bash scripts/rsl_rl/collect_p2pt_noise.sh Isaac-Go1-P2P_Rel_Time-Play-v0 none 0 100 "" results_p2pt_clean.csv &&
bash scripts/rsl_rl/collect_p2pt_noise.sh Isaac-Go1-P2P_Rel_Time-Play-v0 drift 1 100 "" results_p2pt_drift_k1.csv &&
bash scripts/rsl_rl/collect_p2pt_noise.sh Isaac-Go1-P2P_Rel_Time-Play-v0 drift 2 100 "" results_p2pt_drift_k2.csv &&
bash scripts/rsl_rl/collect_p2pt_noise.sh Isaac-Go1-P2P_Rel_Time-Play-v0 drift 3 100 "" results_p2pt_drift_k3.csv
# --> files produced
# --> result_p2pt.csv
# --> result_p2pt_drift_k1.csv
# --> result_p2pt_drift_k2.csv
# --> result_p2pt_drift_k3.csv

# ================== RESULT DISPLAY
# ++++++++++++++++++ Time-series plot
# Per-episode time series (single column, with target-time line):

# --- Cubic polynomial
# First generate data
python scripts/rsl_rl/play_vel_p2p_single.py \
    --task Isaac-Go1-Velocity-Flat-Play-v0 \
    --play_episode 84 \
    --once \
    --headless \
    --export_csv ts_vel_ep84.csv
# Then plot
python scripts/rsl_rl/plot_ieee_ts.py timeseries \
    --csv ts_vel_ep48.csv \
    --episode 84 \
    --margin 0.2 \
    --legend right \
    --out fig_ts_vel_ep84

# --- Velocity replanning
# ++ clean
# First generate data
python scripts/rsl_rl/play_vel_p2p_single_replanning.py \
    --task Isaac-Go1-Velocity-Flat-Play-v0 \
    --play_episode 84 \
    --once \
    --headless \
    --export_csv ts_vel_replanning_ep84.csv
# Then plot
python scripts/rsl_rl/plot_ieee_ts.py timeseries \
    --csv ts_vel_replanning_ep84.csv \
    --episode 84 \
    --margin 0.2 \
    --legend right \
    --out fig_ts_vel_replanning_ep84

# ++ drift
# k = 2
# First generate data
python scripts/rsl_rl/play_vel_p2p_single_replanning.py \
    --task Isaac-Go1-Velocity-Flat-Play-v0 \
    --play_episode 4 \
    --once \
    --noise_model drift \
    --headless \
    --noise_k 2.0 \
    --export_csv ts_vel_replanning_drift_k2_ep4.csv
# Then plot
python scripts/rsl_rl/plot_ieee_ts.py timeseries \
    --csv ts_vel_replanning_drift_k2_ep4.csv \
    --episode 4 \
    --margin 0.2 \
    --legend right \
    --out fig_ts_vel_replanning_drift_k2_ep4

# k = 3
# First generate data
python scripts/rsl_rl/play_vel_p2p_single_replanning.py \
    --task Isaac-Go1-Velocity-Flat-Play-v0 \
    --play_episode 4 \
    --once \
    --noise_model drift \
    --headless \
    --noise_k 3.0 \
    --export_csv ts_vel_replanning_drift_k3_ep4.csv
# Then plot
python scripts/rsl_rl/plot_ieee_ts.py timeseries \
    --csv ts_vel_replanning_drift_k3_ep4.csv \
    --episode 4 \
    --margin 0.2 \
    --legend right \
    --out fig_ts_vel_replanning_drift_k3_ep4

# --- P2PT
# clean
# Collect data
python scripts/rsl_rl/play_p2pt_single.py \
    --task Isaac-Go1-P2P_Rel_Time-Play-v0 \
    --commands_csv p2p_commands.csv \
    --play_episode 84 \
    --headless \
    --once \
    --export_csv ts_p2pt_ep84.csv
# Then plot
python scripts/rsl_rl/plot_ieee_ts.py timeseries \
    --csv ts_p2pt_ep84.csv \
    --episode 84 \
    --margin 0.2 \
    --legend right \
    --out fig_ts_p2pt_ep84

# drift
# Collect data
python scripts/rsl_rl/play_p2pt_single.py \
    --task Isaac-Go1-P2P_Rel_Time-Play-v0 \
    --commands_csv p2p_commands.csv \
    --play_episode 4 \
    --headless \
    --once \
    --noise_model drift \
    --noise_k 2.0 \
    --export_csv ts_p2pt_drift_k2_ep4.csv
# Then plot
python scripts/rsl_rl/plot_ieee_ts.py timeseries \
    --csv ts_p2pt_drift_k2_ep4.csv \
    --episode 4 \
    --margin 0.2 \
    --legend right \
    --out fig_ts_p2pt_drift_k2_ep4


# OR call the batch script to generate the time series figures for comparison of both velocity replanning v.s. p2pt
# example episode 4
# for clean data, example episode 84
bash scripts/rsl_rl/generate_fig_ts_compare_clean.sh 84

# for drift
bash scripts/rsl_rl/generate_fig_ts_compare.sh 15 drift 3.0  
bash scripts/rsl_rl/generate_fig_ts_compare.sh 15 none 0.0




# velocity
python scripts/rsl_rl/play_vel_p2p_single.py --task Isaac-Go1-Velocity-Flat-Play-v0 --random_targets --target_seed 42 --export_csv ts_vel.csv


# -> this produces: vel_data.csv (timeseries) + vel_data_commands.csv (targets used)
python scripts/rsl_rl/plot_ieee.py timeseries --csv ts_vel.csv --episode 4  --margin 0.2 --out fig_ts_vel_ep4

# To watch episode 4 loop continuously on your screen so you can observe the robot's behavior, run:
# Bash
python scripts/rsl_rl/play_vel_p2p_single_replanning.py --task Isaac-Go1-Velocity-Flat-Play-v0 --play_episode 4

# If you are running the time-series extraction setup we just built and want it to play episode 4 exactly once, save the data, and exit, add the --once and --ts_episode flags:
# Bash
python scripts/rsl_rl/play_vel_p2p_single.py --task Isaac-Go1-Velocity-Flat-Play-v0 --play_episode 4 --once --ts_episode 4 --export_csv ts_vel_ep4.csv
# You can append any of your other flags (like --noise_model drift --noise_k 2) directly to these commands to test specific conditions on that single episode.

















# Fig 4 version 2
# compare with velocity Replanning
python scripts/rsl_rl/compare_results.py --vel results_velocity_replanning.csv --p2p results_p2p_clean.csv --out_prefix compare_p2p_vs_vel --dist_floor 0.1
# --> the figure is saved in file named compare_p2p_vs_vel_overview.pdf


