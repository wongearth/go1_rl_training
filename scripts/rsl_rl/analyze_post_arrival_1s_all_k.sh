#!/usr/bin/env bash
# Generate IEEE 1-column PDF for Post-Arrival (+1s) CDFs.
# Loops through noise k=0, 1, 2 to generate a 3x2 grid in a single figure.
# Usage: bash generate_ieee_post_arrival_cdf.sh [NOISE_MODEL]

set -u

NOISE_MODEL=${1:-drift}

echo "=================================================="
echo "Generating IEEE 3x2 Post-Arrival CDF PDF | Noise: ${NOISE_MODEL} (k=0,1,2)"
echo "=================================================="

# Ensure output directory for figures exists
mkdir -p data/fig

OUT_PDF="data/fig/fig_post_arrival_cdf_${NOISE_MODEL}_all_k.pdf"

# Execute inline Python script to parse CSVs and generate the 3x2 plot
python - <<EOF
import sys
import os
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

def load_post_arrival_data(csv_path):
    if not os.path.exists(csv_path):
        print(f"[WARNING] Missing file {csv_path}. Returning empty arrays.")
        return np.array([]), np.array([])
        
    # Load raw data array (skipping the header)
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    
    if data.size == 0:
        return np.array([]), np.array([])
        
    # Dynamically find the correct indices based on the CSV width
    cols = data.shape[1]
    
    if cols >= 16:
        # Standard format (16 or 20 columns)
        # +1s data starts at index 10: err_x, err_y, err_yaw, vel_x, vel_y, vel_yaw
        err_x_idx = 10
        err_y_idx = 11
        err_yaw_idx = 12
    else:
        # Fallback if the CSV is missing +1s data entirely
        print(f"[WARNING] {csv_path} has only {cols} columns. Missing +1s data. Returning zeros.")
        return np.zeros(len(data)), np.zeros(len(data))
    
    # Calculate +1s Metrics
    dist_1s = np.hypot(data[:, err_x_idx], data[:, err_y_idx]) * 100.0   # cm
    yaw_1s = np.abs(data[:, err_yaw_idx]) * (180.0 / np.pi)              # deg
    
    return dist_1s, yaw_1s

# Create 3x2 grid fitting 3.5 inches wide by 4.5 inches tall
fig, axs = plt.subplots(3, 2, figsize=(3.5, 4.5), sharex='col')

# Shared Y axis within each row
for i in range(3):
    axs[i, 1].sharey(axs[i, 0])
    
def plot_cdf(ax, data_vel, data_p2p, xlabel, title, ylabel=None):
    if len(data_vel) > 0 and len(data_p2p) > 0:
        # Sort data for CDF
        x_vel = np.sort(data_vel)
        y_vel = np.arange(1, len(x_vel) + 1) / len(x_vel)
        
        x_p2p = np.sort(data_p2p)
        y_p2p = np.arange(1, len(x_p2p) + 1) / len(x_p2p)
        
        # Colors Swapped: P2P-T is C0, Velocity is C1 (as requested previously)
        ax.plot(x_vel, y_vel, label='CL-VTB', color='C1')
        ax.plot(x_p2p, y_p2p, label='P2P-T', color='C0')
        
    if title:
        ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
        
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

for k in range(3):
    vel_csv = f"results_velocity_CL_${NOISE_MODEL}_k{k}.csv"
    p2p_csv = f"results_p2pt_${NOISE_MODEL}_k{k}.csv"
    
    # Fallback to data/ dir
    if not os.path.exists(vel_csv): vel_csv = f"data/{vel_csv}"
    if not os.path.exists(p2p_csv): p2p_csv = f"data/{p2p_csv}"
        
    dist_vel, yaw_vel = load_post_arrival_data(vel_csv)
    dist_p2p, yaw_p2p = load_post_arrival_data(p2p_csv)
    
    title_dist = 'Steady-State Distance' if k == 0 else None
    title_yaw = 'Steady-State Yaw' if k == 0 else None
    # FIX: Use a double-backslash so Bash passes the literal \n to Python
    ylabel = f'k={k}\\nCumulative Fraction'
    
    plot_cdf(axs[k, 0], dist_vel, dist_p2p, '', title_dist, ylabel)
    plot_cdf(axs[k, 1], yaw_vel, yaw_p2p, '', title_yaw)
    
    axs[k, 1].tick_params(labelleft=False)
    
    # Set x-labels only on the bottom row
    if k == 2:
        axs[k, 0].set_xlabel('Distance Error (cm)')
        axs[k, 1].set_xlabel('Yaw Error (deg)')

# UNIFIED LEGEND
lines, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(lines, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.08), frameon=False)

fig.tight_layout(pad=0.2)
fig.savefig("${OUT_PDF}", format='pdf', bbox_inches='tight')
EOF

echo "=================================================="
echo "DONE! Generated PDF:"
echo " - ${OUT_PDF}"
echo "=================================================="