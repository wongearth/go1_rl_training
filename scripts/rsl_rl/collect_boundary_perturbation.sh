#!/usr/bin/env bash
# Collect P2P-T boundary perturbation results for 100 episodes.
# Usage: bash collect_boundary_perturbation.sh Isaac-Go1-P2P_Rel_Time-Play-v0

set -u

TASK="${1:?need task id as arg 1}"
NUM="${2:-100}"
RESULTS_CSV="data/results_p2pt_boundary_push.csv"
COMMANDS_CSV="p2p_commands.csv"

SCRIPT="scripts/rsl_rl/play_p2pt_single.py"

mkdir -p data
rm -f "${RESULTS_CSV}"

echo "[collect] task=${TASK} episodes=${NUM} test=Boundary_Perturbation"

for (( i=0; i<NUM; i++ )); do
    echo "=================== EPISODE ${i} / ${NUM} ==================="
    python "${SCRIPT}" \
        --task "${TASK}" \
        --commands_csv "${COMMANDS_CSV}" \
        --results_csv "${RESULTS_CSV}" \
        --play_episode "${i}" \
        --boundary_push \
        --once --append --headless
        
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[collect] WARNING: episode ${i} exited with code ${rc}; continuing."
    fi
done

echo "[collect] DONE. Results in ${RESULTS_CSV}"

# Tally the post-arrival performance to prove recovery
python - "${RESULTS_CSV}" <<'PY'
import csv, sys, math
p = sys.argv[1]
data = list(csv.reader(open(p)))[1:]
n = len(data)

if n:
    # Assuming columns 10, 11 are post-arrival (+1s) X and Y errors based on your previous script setup
    if len(data[0]) >= 16:
        errs_1s = [math.hypot(float(r[10]), float(r[11])) for r in data]
        mean_1s = sum(errs_1s)/n
        errs_sorted = sorted(errs_1s)
        med_1s = errs_sorted[n//2]
        
        vels_1s = [math.hypot(float(r[13]), float(r[14])) for r in data]
        mean_vel_1s = sum(vels_1s)/n
        
        print(f"\n[REVIEWER DATA] 100 Episodes | Boundary Perturbation at T-0.2s")
        print(f"-> Post-Arrival (+1s) Distance Err: {mean_1s*100:.1f} cm (Median: {med_1s*100:.1f} cm)")
        print(f"-> Post-Arrival (+1s) Terminal Vel: {mean_vel_1s*100:.1f} cm/s")
PY