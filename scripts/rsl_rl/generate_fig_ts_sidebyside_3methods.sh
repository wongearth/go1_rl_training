#!/usr/bin/env bash
# Generate a side-by-side IEEE 2-column wide PDF comparing 3 methods:
# OL-VB (Open-Loop), CL-VTB (Closed-Loop Replanning), and P2P-T.
# Shared primary y-axes for cm/deg and cm/s / deg/s.
# Usage: bash generate_ieee_pdf_3methods.sh <EPISODE> <NOISE_MODEL> <NOISE_K> [COMMANDS_CSV]

set -u

EPISODE=${1:?Need episode number as arg 1 (e.g., 4)}
NOISE_MODEL=${2:?Need noise model as arg 2 (none, gaussian, drift)}
NOISE_K=${3:?Need noise k level as arg 3 (e.g., 2.0)}
COMMANDS_CSV=${4:-p2p_commands.csv}

K_STR=${NOISE_K%.*}

echo "=================================================="
echo "Generating IEEE 3-Method Comparison PDF for Episode ${EPISODE} | Noise: ${NOISE_MODEL} (k=${NOISE_K})"
echo "=================================================="

mkdir -p data/fig

CUBIC_CSV="data/ts_ep${EPISODE}_cubic_${NOISE_MODEL}_k${K_STR}.csv"
VEL_CSV="data/ts_ep${EPISODE}_vel_CL_${NOISE_MODEL}_k${K_STR}.csv"
P2P_CSV="data/ts_ep${EPISODE}_p2pt_${NOISE_MODEL}_k${K_STR}.csv"
OUT_PDF="data/fig/fig_ts_ep${EPISODE}_compare_3methods_${NOISE_MODEL}_k${K_STR}.pdf"

echo "[1/4] Running Open-Loop Cubic Velocity (OL-VB) Simulation..."
python scripts/rsl_rl/play_vel_p2p_OL.py \
    --task Isaac-Go1-Velocity-Flat-Play-v0 \
    --commands_csv "${COMMANDS_CSV}" \
    --play_episode "${EPISODE}" \
    --once \
    --headless \
    --export_csv "${CUBIC_CSV}"

echo "[2/4] Running Velocity Replanning (CL-VTB) Simulation..."
python scripts/rsl_rl/play_vel_p2p_CL.py \
    --task Isaac-Go1-Velocity-Flat-Play-v0 \
    --commands_csv "${COMMANDS_CSV}" \
    --play_episode "${EPISODE}" \
    --once \
    --headless \
    --noise_model "${NOISE_MODEL}" \
    --noise_k "${NOISE_K}" \
    --export_csv "${VEL_CSV}"

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

echo "[4/4] Generating IEEE 2-column wide PDF (2x3 Grid)..."

# Execute inline Python script (LaTeX dollar signs escaped for bash!)
python - <<EOF
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEE 2-column width is exactly 7.16 inches. 
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'lines.linewidth': 1.0,
})

def load_data(filepath):
    # Columns: 0:ep, 1:t, 2:tg, 3:ex, 4:ey, 5:eyaw ... 12:vx, 13:vy ... 17:wz
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    t = data[:, 1]
    tg = data[0, 2]
    ex, ey, eyaw = data[:, 3], data[:, 4], data[:, 5]
    vx, vy, wz = data[:, 12], data[:, 13], data[:, 17]
    return t, tg, ex, ey, eyaw, vx, vy, wz

t_cub, tg_cub, ex_cub, ey_cub, eyaw_cub, vx_cub, vy_cub, wz_cub = load_data("${CUBIC_CSV}")
t_vel, tg_vel, ex_vel, ey_vel, eyaw_vel, vx_vel, vy_vel, wz_vel = load_data("${VEL_CSV}")
t_p2p, tg_p2p, ex_p2p, ey_p2p, eyaw_p2p, vx_p2p, vy_p2p, wz_p2p = load_data("${P2P_CSV}")

# CONVERT VELOCITY UNITS: m/s to cm/s, and rad/s to deg/s
for v in [vx_cub, vy_cub, vx_vel, vy_vel, vx_p2p, vy_p2p]:
    v *= 100.0
for w in [wz_cub, wz_vel, wz_p2p]:
    w *= (180.0 / np.pi)

# Create 2x3 grid fitting 7.16 inches wide by 3.6 inches tall
fig, axs = plt.subplots(2, 3, figsize=(7.16, 3.6), sharex=True)

# ---------------- ROW 1: ERROR (cm & deg on SAME scale) ----------------
axs[0, 1].sharey(axs[0, 0])
axs[0, 2].sharey(axs[0, 0])

def plot_err(ax, t, tg, ex, ey, eyaw, title):
    ax.plot(t, ex, label='Err X (cm)', color='C0')
    ax.plot(t, ey, label='Err Y (cm)', color='C1')
    ax.plot(t, eyaw, label='Yaw Err (deg)', color='C2', linestyle='--')
    
    ax.axvline(tg, color='red', linestyle=':', linewidth=1.0)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

plot_err(axs[0, 0], t_cub, tg_cub, ex_cub, ey_cub, eyaw_cub, 'OL-VB')
plot_err(axs[0, 1], t_vel, tg_vel, ex_vel, ey_vel, eyaw_vel, 'CL-VTB')
plot_err(axs[0, 2], t_p2p, tg_p2p, ex_p2p, ey_p2p, eyaw_p2p, 'P2P-T')

axs[0, 0].set_ylabel('Error (cm & deg)')
axs[0, 1].tick_params(labelleft=False)
axs[0, 2].tick_params(labelleft=False)

# ---------------- ROW 2: VELOCITY (cm/s & deg/s on SAME scale) ----------------
axs[1, 1].sharey(axs[1, 0])
axs[1, 2].sharey(axs[1, 0])

def plot_vel(ax, t, tg, vx, vy, wz):
    ax.plot(t, vx, label='\$v_x\$ (cm/s)', color='C3')
    ax.plot(t, vy, label='\$v_y\$ (cm/s)', color='C4')
    ax.plot(t, wz, label='\$\omega_z\$ (deg/s)', color='C5', linestyle='--')
    
    ax.axvline(tg, color='red', linestyle=':', linewidth=1.0)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)

plot_vel(axs[1, 0], t_cub, tg_cub, vx_cub, vy_cub, wz_cub)
plot_vel(axs[1, 1], t_vel, tg_vel, vx_vel, vy_vel, wz_vel)
plot_vel(axs[1, 2], t_p2p, tg_p2p, vx_p2p, vy_p2p, wz_p2p)

axs[1, 0].set_ylabel('Velocity (cm/s & deg/s)')
axs[1, 1].tick_params(labelleft=False)
axs[1, 2].tick_params(labelleft=False)

# ---------------- UNIFIED LEGEND ----------------
lines_1, labels_1 = axs[0, 0].get_legend_handles_labels()
lines_2, labels_2 = axs[1, 0].get_legend_handles_labels()

# Combine all 6 labels into a single block at the top
fig.legend(lines_1 + lines_2, labels_1 + labels_2, 
           loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.12), frameon=False)

# ---------------- X-AXIS CLIPPING ----------------
dt = t_cub[1] - t_cub[0] if len(t_cub) > 1 else 0.02
x_max = max(tg_cub, tg_vel, tg_p2p) + (20 * dt)

for ax in axs.flat:
    ax.set_xlim(0, x_max)
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)

fig.tight_layout(pad=0.3)
fig.savefig("${OUT_PDF}", format='pdf', bbox_inches='tight')
EOF

echo "=================================================="
echo "DONE! Generated PDF:"
echo " - ${OUT_PDF}"
echo "=================================================="