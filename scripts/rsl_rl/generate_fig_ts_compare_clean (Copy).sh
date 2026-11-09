#!/usr/bin/env bash
# Generate clean (no-noise) IEEE time-series plots for Open-Loop Velocity vs Velocity Replanning.
# Usage: bash generate_clean_vel_figures.sh <EPISODE> [COMMANDS_CSV]

set -u

EPISODE=${1:?Need episode number as arg 1 (e.g., 84)}
COMMANDS_CSV=${2:-p2p_commands.csv}

echo "=================================================="
echo "Generating Clean Velocity Figures for Episode ${EPISODE}"
echo "=================================================="

# Create the data and figure directories if they don't exist yet
mkdir -p data/fig

# --------------------------------------------------
# 1. OPEN-LOOP CUBIC POLYNOMIAL (play_vel_p2p_single.py)
# --------------------------------------------------
OPEN_CSV="data/ts_vel_ep${EPISODE}.csv"
OPEN_FIG="data/fig/fig_ts_vel_ep${EPISODE}"

echo "[1/4] Running Open-Loop Velocity Simulation..."
python scripts/rsl_rl/play_vel_p2p_single.py \
    --task Isaac-Go1-Velocity-Flat-Play-v0 \
    --commands_csv "${COMMANDS_CSV}" \
    --play_episode "${EPISODE}" \
    --once \
    --headless \
    --export_csv "${OPEN_CSV}"

echo "[2/4] Plotting Open-Loop Velocity Figure..."
python scripts/rsl_rl/plot_ieee_ts.py timeseries \
    --csv "${OPEN_CSV}" \
    --episode "${EPISODE}" \
    --margin 0.2 \
    --legend right \
    --out "${OPEN_FIG}"

# --------------------------------------------------
# 2. VELOCITY REPLANNING (play_vel_p2p_single_replanning.py)
# --------------------------------------------------
REPLAN_CSV="data/ts_vel_replanning_ep${EPISODE}.csv"
REPLAN_FIG="data/fig/fig_ts_vel_replanning_ep${EPISODE}"

echo "[3/4] Running Velocity Replanning Simulation..."
python scripts/rsl_rl/play_vel_p2p_single_replanning.py \
    --task Isaac-Go1-Velocity-Flat-Play-v0 \
    --commands_csv "${COMMANDS_CSV}" \
    --play_episode "${EPISODE}" \
    --once \
    --headless \
    --export_csv "${REPLAN_CSV}"

echo "[4/4] Plotting Velocity Replanning Figure..."
python scripts/rsl_rl/plot_ieee_ts.py timeseries \
    --csv "${REPLAN_CSV}" \
    --episode "${EPISODE}" \
    --margin 0.2 \
    --legend right \
    --out "${REPLAN_FIG}"

echo "=================================================="
echo "DONE! Files generated:"
echo " - ${OPEN_CSV}"
echo " - ${OPEN_FIG}.png"
echo " - ${REPLAN_CSV}"
echo " - ${REPLAN_FIG}.png"
echo "=================================================="