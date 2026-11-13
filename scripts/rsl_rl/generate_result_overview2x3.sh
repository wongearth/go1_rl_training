#!/usr/bin/env bash
# Generate IEEE 2-column wide 2x3 overview plot for 100-episode results.
# Usage: bash generate_results_overview.sh <NOISE_MODEL> <NOISE_K>

set -u

NOISE_MODEL=${1:-drift}
NOISE_K=${2:-2}

K_STR=${NOISE_K%.*}

echo "=================================================="
echo "Generating Results Overview | Noise: ${NOISE_MODEL} (k=${NOISE_K})"
echo "=================================================="

mkdir -p data/fig


VEL_CSV="data/results_velocity_replanning_${NOISE_MODEL}_k${K_STR}.csv"
P2P_CSV="data/results_p2pt_${NOISE_MODEL}_k${K_STR}.csv"
OUT_PDF="data/fig/fig_results_overview_${NOISE_MODEL}_k${K_STR}.pdf"

# Check if files exist before plotting
if [[ ! -f "${VEL_CSV}" || ! -f "${P2P_CSV}" ]]; then
    echo "[ERROR] Missing one or both CSV files:"
    echo "  - ${VEL_CSV}"
    echo "  - ${P2P_CSV}"
    exit 1
fi

echo "[1/1] Parsing 100-episode CSVs and generating 2x3 PDF..."

python - <<EOF
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEE full-page width is ~7.16 inches
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'lines.linewidth': 1.2,
})

def load_metrics(filepath):
    # CSV Indices assumed from standard Isaac Lab evaluation:
    # 4: err_x, 5: err_y, 6: err_yaw, 7: term_vx, 8: term_vy
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    
    # Calculate Distances in cm
    dist_err = np.hypot(data[:, 4], data[:, 5]) * 100.0
    # Calculate Yaw in degrees
    yaw_err = np.abs(data[:, 6]) * (180.0 / np.pi)
    
    # Calculate Terminal Arrival Velocity in cm/s
    if data.shape[1] > 8:
        term_vel = np.hypot(data[:, 7], data[:, 8]) * 100.0
    else:
        term_vel = np.zeros_like(dist_err)
        
    return dist_err, yaw_err, term_vel

d_vel, y_vel, v_vel = load_metrics("${VEL_CSV}")
d_p2p, y_p2p, v_p2p = load_metrics("${P2P_CSV}")

fig, axs = plt.subplots(2, 3, figsize=(7.16, 4.5))

bins_dist = np.linspace(0, max(np.max(d_vel), np.max(d_p2p)), 20)
bins_yaw = np.linspace(0, max(np.max(y_vel), np.max(y_p2p)), 20)
bins_vel = np.linspace(0, max(np.max(v_vel), np.max(v_p2p)), 20)

# --- (a) Distance Error Distribution ---
axs[0, 0].hist(d_vel, bins=bins_dist, alpha=0.6, label='Velocity Replanning', color='C0')
axs[0, 0].hist(d_p2p, bins=bins_dist, alpha=0.6, label='P2P-T', color='C1')
axs[0, 0].set_title('(a) Distance error distribution')
axs[0, 0].set_xlabel('Distance error (cm)')
axs[0, 0].set_ylabel('Episodes')
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].legend()

# --- (b) Yaw Error Distribution ---
axs[0, 1].hist(y_vel, bins=bins_yaw, alpha=0.6, color='C0')
axs[0, 1].hist(y_p2p, bins=bins_yaw, alpha=0.6, color='C1')
axs[0, 1].set_title('(b) Yaw error distribution')
axs[0, 1].set_xlabel('|Yaw| error (deg)')
axs[0, 1].set_ylabel('Episodes')
axs[0, 1].grid(True, alpha=0.3)

# --- (c) Distance Error CDF ---
def plot_cdf(ax, data1, data2, xlabel, title):
    x1, y1 = np.sort(data1), np.arange(1, len(data1) + 1) / len(data1)
    x2, y2 = np.sort(data2), np.arange(1, len(data2) + 1) / len(data2)
    ax.plot(x1, y1, label='Velocity Replanning', color='C0')
    ax.plot(x2, y2, label='P2P-T', color='C1')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative fraction')
    ax.grid(True, alpha=0.3)

plot_cdf(axs[0, 2], d_vel, d_p2p, 'Distance error (cm)', '(c) Distance-error CDF')

# --- (d) Paired Error (Scatter) ---
axs[1, 0].scatter(d_vel, d_p2p, alpha=0.6, s=10, color='C0')
max_val = max(np.max(d_vel), np.max(d_p2p))
axs[1, 0].plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='equal')
axs[1, 0].set_title('(d) Paired distance error')
axs[1, 0].set_xlabel('Velocity dist. err (cm)')
axs[1, 0].set_ylabel('P2P dist. err (cm)')
axs[1, 0].grid(True, alpha=0.3)
axs[1, 0].legend(loc='lower right')

# --- (e) Arrival Velocity Distribution ---
axs[1, 1].hist(v_vel, bins=bins_vel, alpha=0.6, color='C0')
axs[1, 1].hist(v_p2p, bins=bins_vel, alpha=0.6, color='C1')
axs[1, 1].set_title('(e) Arrival velocity distribution')
axs[1, 1].set_xlabel('Terminal linear velocity (cm/s)')
axs[1, 1].set_ylabel('Episodes')
axs[1, 1].grid(True, alpha=0.3)

# --- (f) Arrival Velocity CDF ---
plot_cdf(axs[1, 2], v_vel, v_p2p, 'Terminal linear velocity (cm/s)', '(f) Arrival velocity CDF')

fig.tight_layout(pad=0.5)
fig.savefig("${OUT_PDF}", format='pdf', bbox_inches='tight')
EOF

echo "=================================================="
echo "DONE! Generated PDF: ${OUT_PDF}"
echo "=================================================="