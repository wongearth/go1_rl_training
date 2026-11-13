#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# IEEE 1-column formatting (exactly 3.5 inches wide)
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
    # Load raw data array (skipping the header)
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    
    # Dynamically find the correct indices based on the CSV width
    cols = data.shape[1]
    
    if cols >= 16:
        # Standard format (16 or 20 columns)
        # +1s data starts at index 10: err_x, err_y, err_yaw, vel_x, vel_y, vel_yaw
        err_x_idx = 10
        err_y_idx = 11
        err_yaw_idx = 12
    else:
        # Fallback if the CSV is missing +1s data entirely (prevents the crash)
        print(f"[WARNING] {csv_path} has only {cols} columns. Missing +1s data. Returning zeros.")
        return np.zeros(len(data)), np.zeros(len(data))
    
    # Calculate +1s Metrics
    dist_1s = np.hypot(data[:, err_x_idx], data[:, err_y_idx]) * 100.0   # cm
    yaw_1s = np.abs(data[:, err_yaw_idx]) * (180.0 / np.pi)              # deg
    
    return dist_1s, yaw_1s

def print_stats(name, dist, yaw):
    print(f"=== {name} : Post-Arrival (+1s) Statistics (100 episodes) ===")
    print(f"  Distance Err (cm) : Mean {np.mean(dist):.2f} | Median {np.median(dist):.2f} | Max {np.max(dist):.2f}")
    print(f"  Yaw Err (deg)     : Mean {np.mean(yaw):.2f} | Median {np.median(yaw):.2f} | Max {np.max(yaw):.2f}\n")

def plot_cdf(ax, data_vel, data_p2p, xlabel, title):
    # Sort data for CDF
    x_vel = np.sort(data_vel)
    y_vel = np.arange(1, len(x_vel) + 1) / len(x_vel)
    
    x_p2p = np.sort(data_p2p)
    y_p2p = np.arange(1, len(x_p2p) + 1) / len(x_p2p)
    
    # Matching your established color scheme
    ax.plot(x_vel, y_vel, label='CL-VTB', color='C1')
    ax.plot(x_p2p, y_p2p, label='P2P-T', color='C0')
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Cumulative Fraction')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare +1s Post-Arrival Data")
    parser.add_argument("--csv_vel", required=True, help="Path to CL-VTB results CSV")
    parser.add_argument("--csv_p2p", required=True, help="Path to P2P-T results CSV")
    parser.add_argument("--out_pdf", default="fig_post_arrival_compare.pdf", help="Output PDF name")
    args = parser.parse_args()

    # Load data
    dist_vel, yaw_vel = load_post_arrival_data(args.csv_vel)
    dist_p2p, yaw_p2p = load_post_arrival_data(args.csv_p2p)

    # Print terminal statistics
    print_stats("CL-VTB", dist_vel, yaw_vel)
    print_stats("P2P-T", dist_p2p, yaw_p2p)

    # Generate 1x2 IEEE CDF Plot (3.5 inches wide to fit a single column)
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 1.8), sharey=True)

    plot_cdf(axs[0], dist_vel, dist_p2p, 'Distance Error (cm)', 'Steady-State Distance')
    plot_cdf(axs[1], yaw_vel, yaw_p2p, 'Yaw Error (deg)', 'Steady-State Yaw')

    # Remove inner y-axis labels for a cleaner layout
    axs[1].set_ylabel('')
    axs[1].tick_params(labelleft=False)

    # Single shared legend positioned at the top center
    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.20), frameon=False)

    fig.tight_layout(pad=0.2)
    fig.savefig(args.out_pdf, format='pdf', bbox_inches='tight')
    print(f"[PLOT] Generated {args.out_pdf}")