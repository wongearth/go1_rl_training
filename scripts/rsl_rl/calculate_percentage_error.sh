#!/usr/bin/env bash
# Calculate error as a percentage of the commanded distance.
# Usage: bash calculate_percentage_error.sh <NOISE_MODEL> <NOISE_K>

set -u

NOISE_MODEL=${1:-drift}
NOISE_K=${2:-2}
K_STR=${NOISE_K%.*}

VEL_CSV="data/results_velocity_replanning_${NOISE_MODEL}_k${K_STR}.csv"
P2P_CSV="data/results_p2pt_${NOISE_MODEL}_k${K_STR}.csv"

echo "=================================================="
echo "Percentage Error Analysis | Noise: ${NOISE_MODEL} (k=${NOISE_K})"
echo "=================================================="

if [[ ! -f "${VEL_CSV}" || ! -f "${P2P_CSV}" ]]; then
    echo "[ERROR] Missing CSV files:"
    echo "  - ${VEL_CSV}"
    echo "  - ${P2P_CSV}"
    exit 1
fi

python - <<EOF
import csv
import math
import numpy as np

def analyze_percentage(filepath, policy_name):
    with open(filepath, 'r') as f:
        data = list(csv.reader(f))[1:] # Skip header
        
    pct_errors = []
    
    for r in data:
        # Indices: cmd_x (0), cmd_y (1), err_x (4), err_y (5)
        cmd_dist = math.hypot(float(r[0]), float(r[1]))
        
        # Exclude near-zero distances to prevent inflated percentages
        if cmd_dist >= 0.1:
            err_dist = math.hypot(float(r[4]), float(r[5]))
            pct = (err_dist / cmd_dist) * 100.0
            pct_errors.append(pct)
            
    if pct_errors:
        mean_pct = np.mean(pct_errors)
        median_pct = np.median(pct_errors)
        p90_pct = np.percentile(pct_errors, 90)
        
        print(f"{policy_name:<18}: Mean {mean_pct:5.1f}% | Median {median_pct:5.1f}% | P90 {p90_pct:5.1f}% | (Valid episodes: {len(pct_errors)})")
    else:
        print(f"{policy_name:<18}: No valid episodes with cmd_dist >= 0.1m")

analyze_percentage("${VEL_CSV}", "Velocity-Tracking")
analyze_percentage("${P2P_CSV}", "P2P-T Policy")
EOF