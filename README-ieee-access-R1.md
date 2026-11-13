# ###################### Git management
Backup 
cd ~/go1_rl_training

# 1. Is this a git repo at all?
git status

# 2. Is a remote (GitHub) configured, and what URL?
git remote -v

# 3. What branch are you on, and is it tracking a remote branch?
git branch -vv

# If github already set up then
git add -A
git commit -m "write some note"
git push



# ################################ IEEE ACCESS data collection R1
# DATA COLLECTION: 

# # generate command
python generate_commands.py --num_episodes 100 --seed 42 --out p2p_commands.csv --x_range -1.5 1.5 --y_range -0.75 0.75 --yaw_range -3.14 3.14 --time_range 1.5 3.5
# --> files produced 100 episodes of commands in the file named p2p_commands.csv 
# --> p2p_commands.csv 


# ###################### VELOCITY Open Loop
# == play
python scripts/rsl_rl/play_vel_p2p_OL.py --task Isaac-Go1-Velocity-Flat-Play-v0 --num_envs 1
# --> this will show the results for each run

# == Collect data Velocity CL policy with noise drifts   
bash scripts/rsl_rl/collect_vel_OL.sh Isaac-Go1-Velocity-Flat-Play-v0 "" 100
# --> result_velocity_OL.csv

# ###################### VELOCITY Closed Loop
# To play velocity  policy 
python scripts/rsl_rl/play_vel_p2p_CL.py --task Isaac-Go1-Velocity-Flat-Play-v0 --num_envs 1
# --> this will show the results for each run


# without noise
# <--- the command above need file play_vel_p2p_CL.py
bash scripts/rsl_rl/collect_vel_CL.sh Isaac-Go1-Velocity-Flat-Play-v0 "" 100 && 
# --> files produced
# --> result_vel_CL.csv

<!-- # #with noise 
<!-- bash scripts/rsl_rl/collect_vel_CL.sh Isaac-Go1-Velocity-Flat-Play-v0 gaussian 1 100 "" &&
bash scripts/rsl_rl/collect_vel_CL.sh Isaac-Go1-Velocity-Flat-Play-v0 gaussian 2 100 "" &&
bash scripts/rsl_rl/collect_vel_CL.sh Isaac-Go1-Velocity-Flat-Play-v0 gaussian 3 100 "" && -->
<!-- # -> files produced
# -> result_vel_CL_noise_k1.csv
# -> result_vel_CL_noise_k2.csv
# -> result_vel_CL_noise_k3.csv --> 

# with drifts
bash scripts/rsl_rl/collect_vel_CL.sh Isaac-Go1-Velocity-Flat-Play-v0 drift 0 100 "" && 
bash scripts/rsl_rl/collect_vel_CL.sh Isaac-Go1-Velocity-Flat-Play-v0 drift 1 100 "" &&
bash scripts/rsl_rl/collect_vel_CL.sh Isaac-Go1-Velocity-Flat-Play-v0 drift 2 100 "" &&
bash scripts/rsl_rl/collect_vel_CL.sh Isaac-Go1-Velocity-Flat-Play-v0 drift 3 100 ""
# --> files produced
# --> result_vel_CL_drift_k1.csv
# --> result_vel_CL_drift_k2.csv
# --> result_vel_CL_drift_k3.csv

# ========= Collect data P2P-T policy with noise
# <-- need file play_p2pt_single.py
<!-- bash scripts/rsl_rl/collect_p2pt_noise.sh Isaac-Go1-P2P_Rel_Time-Play-v0 none 0 100 "" results_p2p_clean.csv &&
bash scripts/rsl_rl/collect_p2pt_noise.sh Isaac-Go1-P2P_Rel_Time-Play-v0 all 1 100 "" results_p2p_allsensors_k1.csv &&
bash scripts/rsl_rl/collect_p2pt_noise.sh Isaac-Go1-P2P_Rel_Time-Play-v0 all 2 100 "" results_p2p_allsensors_k2.csv &&
bash scripts/rsl_rl/collect_p2pt_noise.sh Isaac-Go1-P2P_Rel_Time-Play-v0 all 3 100 "" results_p2p_allsensors_k3.csv
# -> files produced
# -> result_p2pt.csv
# -> result_p2pt_allsensors_k1.csv
# -> result_p2pt_allsensors_k2.csv
# -> result_p2pt_allsensors_k3.csv -->

# ========= Collect data P2P-T policy with drifts
# <-- need file play_p2pt_single.py
# use default saved location
bash scripts/rsl_rl/collect_p2pt_noise.sh Isaac-Go1-P2P_Rel_Time-Play-v0 drift 0 100 "" &&
bash scripts/rsl_rl/collect_p2pt_noise.sh Isaac-Go1-P2P_Rel_Time-Play-v0 drift 1 100 "" &&
bash scripts/rsl_rl/collect_p2pt_noise.sh Isaac-Go1-P2P_Rel_Time-Play-v0 drift 2 100 "" &&
bash scripts/rsl_rl/collect_p2pt_noise.sh Isaac-Go1-P2P_Rel_Time-Play-v0 drift 3 100 ""


# --> files produced
# --> result_p2pt.csv
# --> result_p2pt_drift_k1.csv
# --> result_p2pt_drift_k2.csv
# --> result_p2pt_drift_k3.csv

# ############################################################################################

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

# --- Velocity CL
# ++ clean
# First generate data
python scripts/rsl_rl/play_vel_p2p_single_CL.py \
    --task Isaac-Go1-Velocity-Flat-Play-v0 \
    --play_episode 84 \
    --once \
    --headless \
    --export_csv ts_vel_CL_ep84.csv
# Then plot
python scripts/rsl_rl/plot_ieee_ts.py timeseries \
    --csv ts_vel_CL_ep84.csv \
    --episode 84 \
    --margin 0.2 \
    --legend right \
    --out fig_ts_vel_CL_ep84

# ++ drift
# k = 2
# First generate data
python scripts/rsl_rl/play_vel_p2p_single_CL.py \
    --task Isaac-Go1-Velocity-Flat-Play-v0 \
    --play_episode 4 \
    --once \
    --noise_model drift \
    --headless \
    --noise_k 2.0 \
    --export_csv ts_vel_CL_drift_k2_ep4.csv
# Then plot
python scripts/rsl_rl/plot_ieee_ts.py timeseries \
    --csv ts_vel_CL_drift_k2_ep4.csv \
    --episode 4 \
    --margin 0.2 \
    --legend right \
    --out fig_ts_vel_CL_drift_k2_ep4

# k = 3
# First generate data
python scripts/rsl_rl/play_vel_p2p_single_CL.py \
    --task Isaac-Go1-Velocity-Flat-Play-v0 \
    --play_episode 4 \
    --once \
    --noise_model drift \
    --headless \
    --noise_k 3.0 \
    --export_csv ts_vel_CL_drift_k3_ep4.csv
# Then plot
python scripts/rsl_rl/plot_ieee_ts.py timeseries \
    --csv ts_vel_CL_drift_k3_ep4.csv \
    --episode 4 \
    --margin 0.2 \
    --legend right \
    --out fig_ts_vel_CL_drift_k3_ep4

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


# OR call the batch script to generate the time series figures for comparison of both velocity CL v.s. p2pt
# example episode 4
# for clean data, example episode 84
bash scripts/rsl_rl/generate_fig_ts_compare_clean.sh 84

# for drift
bash scripts/rsl_rl/generate_fig_ts_compare.sh 15 drift 3.0  
bash scripts/rsl_rl/generate_fig_ts_compare.sh 15 none 0.0

# for side-by-side comparison
bash scripts/rsl_rl/generate_fig_ts_sidebyside_cubic_p2pt.sh 4 

# Velocity CL vs p2pt
bash scripts/rsl_rl/generate_fig_ts_sidebyside.sh 4 drift 0.0 &&
bash scripts/rsl_rl/generate_fig_ts_sidebyside.sh 4 drift 1.0 &&
bash scripts/rsl_rl/generate_fig_ts_sidebyside.sh 4 drift 2.0 &&
bash scripts/rsl_rl/generate_fig_ts_sidebyside.sh 4 drift 3.0

# compare all 3 methods
bash scripts/rsl_rl/generate_fig_ts_sidebyside_3methods.sh 4 drift 0


# ################## OVERVIEW COMPARATIVE PLOT
bash scripts/rsl_rl/generate_result_overview3x3.sh drift 0 &&
bash scripts/rsl_rl/generate_result_overview3x3.sh drift 1 &&
bash scripts/rsl_rl/generate_result_overview3x3.sh drift 2 &&
bash scripts/rsl_rl/generate_result_overview3x3.sh drift 3

# ---- plot degradation trends as k increase
bash scripts/rsl_rl/plot_degradation_trends.sh
# -> this generates data/fig/fig_degradation_trends.pdf file


# ######################### 1 second after the target time
# ######################## FOr analysis of the robot 1 SECOND after the target time
python scripts/rsl_rl/analyze_post_arrival.py \
    --csv_vel data/results_velocity_CL_drift_k3.csv \
    --csv_p2p data/results_p2pt_drift_k3.csv \
    --out_pdf data/fig/fig_post_arrival_k3.pdf


# ############## To see the effects if a push (spike in velocity) 0.2 second before the target time
# <-- need file play_p2pt_single_push.py
bash collect_boundary_perturbation.sh Isaac-Go1-P2P_Rel_Time-Play-v0
# --> data is  
