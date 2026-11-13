#!/usr/bin/env bash
# Loop over k from 0 to 3 to analyze post-arrival (+1s) steady-state parking performance.
# Usage: bash analyze_post_arrival_1s.sh [NOISE_MODEL]

set -u

NOISE_MODEL=${1:-drift}

echo "=================================================="
echo "Running Post-Arrival Analysis for k=0 to 3 | Noise: ${NOISE_MODEL}"
echo "=================================================="

# Ensure output directory for figures exists
mkdir -p data/fig

for k in 0 1 2 3; do
    echo "--------------------------------------------------"
    echo "Analyzing k=${k}..."
    
    # Check default root paths
    VEL_CSV="data/results_velocity_CL_${NOISE_MODEL}_k${k}.csv"
    P2P_CSV="data/results_p2pt_${NOISE_MODEL}_k${k}.csv"
    OUT_PDF="data/fig/fig_post_arrival_${NOISE_MODEL}_k${k}.pdf"
    
    # Fallback in case your CSVs are stored in the data/ folder
    if [[ ! -f "${VEL_CSV}" ]]; then
        VEL_CSV="data/results_velocity_CL_${NOISE_MODEL}_k${k}.csv"
    fi
    if [[ ! -f "${P2P_CSV}" ]]; then
        P2P_CSV="data/results_p2pt_${NOISE_MODEL}_k${k}.csv"
    fi
    
    # Verify files exist before running the python script
    if [[ ! -f "${VEL_CSV}" || ! -f "${P2P_CSV}" ]]; then
        echo "[WARNING] Missing one or both CSVs for k=${k}. Skipping..."
        echo "  - Checked for: ${VEL_CSV}"
        echo "  - Checked for: ${P2P_CSV}"
        continue
    fi
    
    python scripts/rsl_rl/analyze_post_arrival.py \
        --csv_vel "${VEL_CSV}" \
        --csv_p2p "${P2P_CSV}" \
        --out_pdf "${OUT_PDF}"
        
    echo "[DONE] Generated ${OUT_PDF}"
done

echo "=================================================="
echo "All 1-second post-arrival analyses complete."
echo "=================================================="