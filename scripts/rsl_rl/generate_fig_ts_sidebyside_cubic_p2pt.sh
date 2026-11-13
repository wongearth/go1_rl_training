#!/usr/bin/env bash
# Generate IEEE 1-column PDF comparing Open-Loop Cubic Velocity Planning vs P2P-T (No Noise).
# Usage: bash generate_ieee_pdf_cubic_vs_p2pt.sh <EPISODE> [COMMANDS_CSV]

set -u

EPISODE=${1:?Need episode number as arg 1 (e.g., 84)}
COMMANDS_CSV=${2:-p2p_commands.csv}

echo "=================================================="
echo "Generating IEEE Comparison PDF (Cubic vs P2P-T) for Episode ${EPISODE} | Clean (No Noise)"
echo "=================================================="

mkdir -p data/fig

CUBIC_CSV="data/ts_ep${EPISODE}_cubic_none_k0.csv"
P2P_CSV="data/ts_ep${EPISODE}_p2pt_none_k0.csv"
OUT_PDF="data/fig/fig_ts_ep${EPISODE}_compare_cubic_vs_p2pt_clean.pdf"

echo "[1/3] Running Open-Loop Cubic Velocity Simulation..."
python scripts/rsl_rl/play_vel_p2p_single.py \
    --task Isaac-Go1-Velocity-Flat-Play-v0 \
    --commands_csv "${COMMANDS_CSV}" \
    --play_episode "${EPISODE}" \
    --once \
    --headless \
    --export_csv "${CUBIC_CSV}"

echo "[2/3] Running P2P-T Simulation (Clean)..."
python scripts/rsl_rl/play_p2pt_single.py \
    --task Isaac-Go1-P2P_Rel_Time-Play-v0 \
    --commands_csv "${COMMANDS_CSV}" \
    --play_episode "${EPISODE}" \
    --once \
    --headless \
    --noise_model "none" \
    --noise_k "0" \
    --export_csv "${P2P_CSV}"

echo "[3/3] Generating IEEE 1-column PDF (2x2 Grid)..."

# Execute inline Python script
python - <<EOF
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEE 1-column width is exactly 3.5 inches (90mm). 
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 8,
    'legend.fontsize': 6,
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
t_p2p, tg_p2p, ex_p2p, ey_p2p, eyaw_p2p, vx_p2p, vy_p2p, wz_p2p = load_data("${P2P_CSV}")

# CONVERT VELOCITY UNITS: m/s to cm/s, and rad/s to deg/s
vx_cub *= 100.0
vy_cub *= 100.0
wz_cub *= (180.0 / np.pi)

vx_p2p *= 100.0
vy_p2p *= 100.0
wz_p2p *= (180.0 / np.pi)

# Create 2x2 grid fitting 3.5 inches wide by 3.4 inches tall
fig, axs = plt.subplots(2, 2, figsize=(3.5, 3.4), sharex=True)

# Link all primary y-axes horizontally across rows
axs[0, 1].sharey(axs[0, 0])
axs[1, 1].sharey(axs[1, 0])

# ---------------- ROW 1: ERROR (cm & deg on SAME scale) ----------------
def plot_err(ax, t, tg, ex, ey, eyaw, title):
    ax.plot(t, ex, label='Err X (cm)', color='C0')
    ax.plot(t, ey, label='Err Y (cm)', color='C1')
    ax.plot(t, eyaw, label='Yaw Err (deg)', color='C2', linestyle='--')
    
    ax.axvline(tg, color='red', linestyle=':', linewidth=1.0)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

plot_err(axs[0, 0], t_cub, tg_cub, ex_cub, ey_cub, eyaw_cub, 'Cubic Velocity Planning')
plot_err(axs[0, 1], t_p2p, tg_p2p, ex_p2p, ey_p2p, eyaw_p2p, 'P2P-T Policy')

axs[0, 0].set_ylabel('Error (cm & deg)')
axs[0, 1].tick_params(labelleft=False)

# ---------------- ROW 2: VELOCITY (cm/s & deg/s on SAME scale) ----------------
def plot_vel(ax, t, tg, vx, vy, wz):
    ax.plot(t, vx, label='\$v_x\$ (cm/s)', color='C3')
    ax.plot(t, vy, label='\$v_y\$ (cm/s)', color='C4')
    ax.plot(t, wz, label='\$\omega_z\$ (deg/s)', color='C5', linestyle='--')
    
    ax.axvline(tg, color='red', linestyle=':', linewidth=1.0)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.grid(True, alpha=0.3)

plot_vel(axs[1, 0], t_cub, tg_cub, vx_cub, vy_cub, wz_cub)
plot_vel(axs[1, 1], t_p2p, tg_p2p, vx_p2p, vy_p2p, wz_p2p)

axs[1, 0].set_ylabel('Velocity (cm/s & deg/s)')
axs[1, 1].tick_params(labelleft=False)

# ---------------- UNIFIED LEGEND ----------------
lines_1, labels_1 = axs[0, 0].get_legend_handles_labels()
lines_2, labels_2 = axs[1, 0].get_legend_handles_labels()

fig.legend(lines_1 + lines_2, labels_1 + labels_2, 
           loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.13), frameon=False)

# ---------------- X-AXIS CLIPPING ----------------
dt = t_cub[1] - t_cub[0] if len(t_cub) > 1 else 0.02
x_max = max(tg_cub, tg_p2p) + (20 * dt)

for ax in axs.flat:
    ax.set_xlim(0, x_max)
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)

fig.tight_layout(pad=0.2)
fig.savefig("${OUT_PDF}", format='pdf', bbox_inches='tight')
EOF

echo "=================================================="
echo "DONE! Generated PDF:"
echo " - ${OUT_PDF}"
echo "=================================================="