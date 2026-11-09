#!/usr/bin/env bash
# Generate IEEE time-series plots for both Velocity Replanning and P2P-T policies.
# Usage: bash generate_comparison_figures.sh <EPISODE> <NOISE_MODEL> <NOISE_K> [COMMANDS_CSV]

set -u

EPISODE=${1:?Need episode number as arg 1 (e.g., 4)}
NOISE_MODEL=${2:?Need noise model as arg 2 (none, gaussian, drift)}
NOISE_K=${3:?Need noise k level as arg 3 (e.g., 3.0)}
COMMANDS_CSV=${4:-p2p_commands.csv}

# Format k for filenames to avoid ugly decimals (e.g., 3.0 -> 3)
K_STR=${NOISE_K%.*}

echo "=================================================="
echo "Generating Figures for Episode ${EPISODE} | Noise: ${NOISE_MODEL} (k=${NOISE_K})"
echo "=================================================="

# Create the data and figure directories if they don't exist yet
mkdir -p data/fig

# --------------------------------------------------
# 1. VELOCITY REPLANNING
# --------------------------------------------------
VEL_CSV="data/ts_ep${EPISODE}_vel_replanning_${NOISE_MODEL}_k${K_STR}.csv"
VEL_FIG="data/fig/fig_ts_ep${EPISODE}_vel_replanning_${NOISE_MODEL}_k${K_STR}"

echo "[1/4] Running Velocity Replanning Simulation..."
python scripts/rsl_rl/play_vel_p2p_single_replanning.py \
    --task Isaac-Go1-Velocity-Flat-Play-v0 \
    --commands_csv "${COMMANDS_CSV}" \
    --play_episode "${EPISODE}" \
    --once \
    --headless \
    --noise_model "${NOISE_MODEL}" \
    --noise_k "${NOISE_K}" \
    --export_csv "${VEL_CSV}"

echo "[2/4] Plotting Velocity Replanning Figure..."
python scripts/rsl_rl/plot_ieee_ts.py timeseries \
    --csv "${VEL_CSV}" \
    --episode "${EPISODE}" \
    --margin 0.2 \
    --legend right \
    --out "${VEL_FIG}"

# --------------------------------------------------
# 2. POINT-TO-POINT (P2P-T)
# --------------------------------------------------
P2P_CSV="data/ts_ep${EPISODE}_p2pt_${NOISE_MODEL}_k${K_STR}.csv"
P2P_FIG="data/fig/fig_ts_ep${EPISODE}_p2pt_${NOISE_MODEL}_k${K_STR}"

echo "[3/4] Running P2P-T Simulation..."
python scripts/rsl_rl/play_p2pt_single.py \
    --task Isaac-Go1-P2P_Rel_Time-Play-v0 \
    --commands_csv "${COMMANDS_CSV}" \
    --play_episode "${EPISODE}" \
    --once \
    --headless \
    --noise_model "${NOISE_MODEL}" \
    --noise_k "${NOISE_K}" \
    --export_csv "${P2P_CSV}"

echo "[4/4] Plotting P2P-T Figure..."
python scripts/rsl_rl/plot_ieee_ts.py timeseries \
    --csv "${P2P_CSV}" \
    --episode "${EPISODE}" \
    --margin 0.2 \
    --legend right \
    --out "${P2P_FIG}"

echo "=================================================="
echo "DONE! Files generated:"
echo " - ${VEL_CSV}"
echo " - ${VEL_FIG}.png"
echo " - ${P2P_CSV}"
echo " - ${P2P_FIG}.png"
echo "=================================================="