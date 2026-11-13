#!/usr/bin/env bash
# Generate IEEE 2x2 trend analysis for k=0, 1, 2, 3 odometry drift.
# Usage: bash plot_degradation_trends.sh

set -u

echo "=================================================="
echo "Generating Degradation Trends (k=0 to 3)"
echo "=================================================="

mkdir -p data/fig
OUT_PDF="data/fig/fig_degradation_trends.pdf"

python - <<EOF
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
})

def extract_metrics(k):
    vel_csv = f"data/results_velocity_replanning_drift_k{k}.csv"
    p2p_csv = f"data/results_p2pt_drift_k{k}.csv"
    
    metrics = {'vel': {}, 'p2p': {}}
    
    for name, filepath in zip(['vel', 'p2p'], [vel_csv, p2p_csv]):
        if not os.path.exists(filepath):
            print(f"[WARNING] Missing {filepath}. Padding with NaNs.")
            metrics[name] = {'dist': np.nan, 'yaw': np.nan, 'pct': np.nan, 'vel': np.nan, 
                             'dist_std': np.nan, 'yaw_std': np.nan}
            continue
            
        # Indices: 0:cmd_x, 1:cmd_y, 4:err_x, 5:err_y, 6:err_yaw, 7:term_vx, 8:term_vy
        data = np.genfromtxt(filepath, delimiter=',', skip_header=1)
        
        cmd_dist = np.hypot(data[:, 0], data[:, 1])
        dist_err = np.hypot(data[:, 4], data[:, 5]) * 100.0  # cm
        yaw_err = np.abs(data[:, 6]) * (180.0 / np.pi)       # deg
        
        valid_idx = cmd_dist >= 0.1
        pct_err = np.zeros_like(dist_err)
        pct_err[valid_idx] = (dist_err[valid_idx] / (cmd_dist[valid_idx] * 100.0)) * 100.0 # pure percentage
        
        term_lin_vel = np.hypot(data[:, 7], data[:, 8]) * 100.0 if data.shape[1] > 8 else np.zeros_like(dist_err)
        
        metrics[name]['dist'] = np.mean(dist_err)
        metrics[name]['dist_std'] = np.std(dist_err)
        metrics[name]['yaw'] = np.mean(yaw_err)
        metrics[name]['yaw_std'] = np.std(yaw_err)
        metrics[name]['pct'] = np.mean(pct_err[valid_idx]) if np.any(valid_idx) else np.nan
        metrics[name]['vel'] = np.mean(term_lin_vel)
        
    return metrics

k_vals = [0, 1, 2, 3]
results = {k: extract_metrics(k) for k in k_vals}

# Extract arrays for plotting
def get_array(policy, metric):
    return np.array([results[k][policy][metric] for k in k_vals])

v_dist, p_dist = get_array('vel', 'dist'), get_array('p2p', 'dist')
v_dist_s, p_dist_s = get_array('vel', 'dist_std'), get_array('p2p', 'dist_std')
v_yaw, p_yaw = get_array('vel', 'yaw'), get_array('p2p', 'yaw')
v_yaw_s, p_yaw_s = get_array('vel', 'yaw_std'), get_array('p2p', 'yaw_std')
v_pct, p_pct = get_array('vel', 'pct'), get_array('p2p', 'pct')
v_vel, p_vel = get_array('vel', 'vel'), get_array('p2p', 'vel')

fig, axs = plt.subplots(2, 2, figsize=(7.16, 4.5))

def plot_trend(ax, y_vel, y_vel_std, y_p2p, y_p2p_std, title, ylabel, show_legend=False):
    ax.plot(k_vals, y_vel, 'o-', color='C0', label='CL-TVB')
    if y_vel_std is not None:
        ax.fill_between(k_vals, y_vel - y_vel_std, y_vel + y_vel_std, color='C1', alpha=0.15)
        
    ax.plot(k_vals, y_p2p, 's-', color='C1', label='P2P-T')
    if y_p2p_std is not None:
        ax.fill_between(k_vals, y_p2p - y_p2p_std, y_p2p + y_p2p_std, color='C0', alpha=0.15)
        
    ax.set_title(title)
    ax.set_xlabel('Odometry Drift Scale (k)')
    ax.set_ylabel(ylabel)
    ax.set_xticks(k_vals)
    ax.grid(True, alpha=0.3)
    if show_legend:
        ax.legend(loc='upper left')

plot_trend(axs[0, 0], v_dist, v_dist_s, p_dist, p_dist_s, '(a) Mean Distance Error', 'Error (cm)', show_legend=True)
plot_trend(axs[0, 1], v_yaw, v_yaw_s, p_yaw, p_yaw_s, '(b) Mean Yaw Error', 'Error (deg)')
plot_trend(axs[1, 0], v_pct, None, p_pct, None, '(c) Mean Percentage Error', 'Error (% of Cmd Distance)')
plot_trend(axs[1, 1], v_vel, None, p_vel, None, '(d) Mean Terminal Velocity', 'Linear Velocity (cm/s)')

fig.tight_layout(pad=0.2, h_pad=0.6, w_pad=0.4)
fig.savefig("${OUT_PDF}", format='pdf', bbox_inches='tight')
EOF

echo "[DONE] Generated ${OUT_PDF}"