#!/usr/bin/env bash
# Generate IEEE 2-column wide 3x3 overview plot for 100-episode results.
# Usage: bash generate_results_overview.sh <NOISE_MODEL> <NOISE_K>

set -u

NOISE_MODEL=${1:-drift}
NOISE_K=${2:-2}

K_STR=${NOISE_K%.*}

echo "=================================================="
echo "Generating 3x3 Results Overview | Noise: ${NOISE_MODEL} (k=${NOISE_K})"
echo "=================================================="

mkdir -p data/fig

VEL_CSV="data/results_velocity_CL_${NOISE_MODEL}_k${K_STR}.csv"
P2P_CSV="data/results_p2pt_${NOISE_MODEL}_k${K_STR}.csv"
OUT_PDF="data/fig/fig_results_overview_${NOISE_MODEL}_k${K_STR}.pdf"


if [[ ! -f "${VEL_CSV}" || ! -f "${P2P_CSV}" ]]; then
    echo "[ERROR] Missing one or both CSV files:"
    echo "  - ${VEL_CSV}"
    echo "  - ${P2P_CSV}"
    exit 1
fi

echo "[1/1] Parsing 100-episode CSVs and generating 3x3 PDF..."

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
    # CSV Indices assumed: 4:err_x, 5:err_y, 6:err_yaw, 7:term_vx, 8:term_vy, 9:term_wz
    data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
    
    # Position and Yaw Errors
    dist_err = np.hypot(data[:, 4], data[:, 5]) * 100.0
    yaw_err = np.abs(data[:, 6]) * (180.0 / np.pi)
    
    # Terminal Linear Velocity (cm/s)
    if data.shape[1] > 8:
        term_lin_vel = np.hypot(data[:, 7], data[:, 8]) * 100.0
    else:
        term_lin_vel = np.zeros_like(dist_err)
        
    # Terminal Angular Velocity (deg/s)
    if data.shape[1] > 9:
        term_ang_vel = np.abs(data[:, 9]) * (180.0 / np.pi)
    else:
        term_ang_vel = np.zeros_like(dist_err)
        
    return dist_err, yaw_err, term_lin_vel, term_ang_vel

d_vel, y_vel, lv_vel, av_vel = load_metrics("${VEL_CSV}")
d_p2p, y_p2p, lv_p2p, av_p2p = load_metrics("${P2P_CSV}")

# 3x3 Grid
fig, axs = plt.subplots(3, 3, figsize=(7.16, 4.5))

bins_dist = np.linspace(0, max(np.max(d_vel), np.max(d_p2p)), 20)
bins_yaw = np.linspace(0, max(np.max(y_vel), np.max(y_p2p)), 20)
bins_lvel = np.linspace(0, max(np.max(lv_vel), np.max(lv_p2p)), 20)
bins_avel = np.linspace(0, max(np.max(av_vel), np.max(av_p2p)), 20)

def plot_cdf(ax, data1, data2, xlabel, title):
    x1, y1 = np.sort(data1), np.arange(1, len(data1) + 1) / len(data1)
    x2, y2 = np.sort(data2), np.arange(1, len(data2) + 1) / len(data2)
    ax.plot(x1, y1, label='CL-VTB', color='C0')
    ax.plot(x2, y2, label='P2P-T', color='C1')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative fraction')
    ax.grid(True, alpha=0.3)
    if ax == axs[0, 1]:  # Only add legend to the first CDF
        ax.legend()

def plot_scatter(ax, data1, data2, xlabel, ylabel, title):
    ax.scatter(data1, data2, alpha=0.6, s=10, color='C0')
    max_val = max(np.max(data1), np.max(data2))
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, label='equal')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

# ================= ROW 1: DISTANCE ERROR =================
axs[0, 0].hist(d_vel, bins=bins_dist, alpha=0.6, label='CL-VTB', color='C0')
axs[0, 0].hist(d_p2p, bins=bins_dist, alpha=0.6, label='P2P-T', color='C1')
axs[0, 0].set_title('(a) Distance error distribution')
axs[0, 0].set_xlabel('Distance error (cm)')
axs[0, 0].set_ylabel('Episodes')
axs[0, 0].grid(True, alpha=0.3)

plot_cdf(axs[0, 1], d_vel, d_p2p, 'Distance error (cm)', '(b) Distance error CDF')
plot_scatter(axs[0, 2], d_vel, d_p2p, 'CL-VTB dist. err (cm)', 'P2P-T dist. err (cm)', '(c) Paired distance error')

# ================= ROW 2: YAW ERROR =================
axs[1, 0].hist(y_vel, bins=bins_yaw, alpha=0.6, color='C0')
axs[1, 0].hist(y_p2p, bins=bins_yaw, alpha=0.6, color='C1')
axs[1, 0].set_title('(d) Yaw error distribution')
axs[1, 0].set_xlabel('|Yaw| error (deg)')
axs[1, 0].set_ylabel('Episodes')
axs[1, 0].grid(True, alpha=0.3)

plot_cdf(axs[1, 1], y_vel, y_p2p, '|Yaw| error (deg)', '(e) Yaw error CDF')
plot_scatter(axs[1, 2], y_vel, y_p2p, 'Velocity yaw err (deg)', 'P2P-T yaw err (deg)', '(f) Paired yaw error')

# ================= ROW 3: ARRIVAL VELOCITY =================
axs[2, 0].hist(lv_vel, bins=bins_lvel, alpha=0.6, color='C0')
axs[2, 0].hist(lv_p2p, bins=bins_lvel, alpha=0.6, color='C1')
axs[2, 0].set_title('(g) Terminal linear velocity')
axs[2, 0].set_xlabel('Linear velocity (cm/s)')
axs[2, 0].set_ylabel('Episodes')
axs[2, 0].grid(True, alpha=0.3)

axs[2, 1].hist(av_vel, bins=bins_avel, alpha=0.6, color='C0')
axs[2, 1].hist(av_p2p, bins=bins_avel, alpha=0.6, color='C1')
axs[2, 1].set_title('(h) Terminal angular velocity')
axs[2, 1].set_xlabel('Angular velocity (deg/s)')
axs[2, 1].set_ylabel('Episodes')
axs[2, 1].grid(True, alpha=0.3)

plot_cdf(axs[2, 2], lv_vel, lv_p2p, 'Linear velocity (cm/s)', '(i) Terminal linear vel CDF')

# fig.tight_layout(pad=0.5)
fig.tight_layout(pad=0.2, h_pad=0.4, w_pad=0.3)

fig.savefig("${OUT_PDF}", format='pdf', bbox_inches='tight')
EOF

echo "=================================================="
echo "DONE! Generated PDF: ${OUT_PDF}"
echo "=================================================="