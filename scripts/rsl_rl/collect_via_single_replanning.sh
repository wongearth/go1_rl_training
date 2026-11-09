#!/usr/bin/env bash
# Collect velocity-P2P results with noise injection
# Usage:
#   bash collect_via_single_replanning.sh <TASK> <NOISE_MODEL> <NOISE_K> [NUM] [CHECKPOINT] [RESULTS_CSV] [COMMANDS_CSV]

set -u

TASK="${1:?need task id as arg 1}"
NOISE_MODEL="${2:?need noise_model (none|gaussian|drift) as arg 2}"
NOISE_K="${3:?need noise_k as arg 3}"
NUM="${4:-100}"
CKPT="${5:-}"
RESULTS_CSV="${6:-results_velocity_replanning_${NOISE_MODEL}_k${NOISE_K}.csv}"
COMMANDS_CSV="${7:-p2p_commands.csv}"

SCRIPT="scripts/rsl_rl/play_vel_p2p_single_replanning.py"

# build the optional --checkpoint flag only when a path was given
CKPT_ARG=()
if [[ -n "${CKPT}" ]]; then
    CKPT_ARG=(--checkpoint "${CKPT}")
fi

# build the noise flag
NOISE_ARG=()
if [[ "${NOISE_MODEL}" != "none" && "${NOISE_K}" != "0" ]]; then
    NOISE_ARG=(--noise_model "${NOISE_MODEL}" --noise_k "${NOISE_K}")
fi

# fresh results file (remove so --append starts clean with a header)
rm -f "${RESULTS_CSV}"

echo "[collect] task=${TASK} noise_model=${NOISE_MODEL} k=${NOISE_K} episodes=${NUM}"
echo "[collect] commands=${COMMANDS_CSV}  results=${RESULTS_CSV}"

for (( i=0; i<NUM; i++ )); do
    echo "=================== EPISODE ${i} / ${NUM} ==================="
    python "${SCRIPT}" \
        --task "${TASK}" \
        ${CKPT_ARG[@]+"${CKPT_ARG[@]}"} \
        --commands_csv "${COMMANDS_CSV}" \
        --results_csv "${RESULTS_CSV}" \
        --play_episode "${i}" \
        --once --append --headless \
        ${NOISE_ARG[@]+"${NOISE_ARG[@]}"}
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "[collect] WARNING: episode ${i} exited with code ${rc}; continuing."
    fi
done

echo "[collect] DONE. Results in ${RESULTS_CSV}"

# quick tally
python - "${RESULTS_CSV}" <<'PY'
import csv, sys, math
p = sys.argv[1]
data = list(csv.reader(open(p)))[1:]
n = len(data)

if n:
    errs = [math.hypot(float(r[4]), float(r[5])) for r in data]
    errs_sorted = sorted(errs)
    mean = sum(errs)/n
    med = errs_sorted[n//2]
    big = sum(1 for e in errs if e > 0.4)
    
    # Check if velocity columns exist
    has_vel = len(data[0]) > 7
    vel_str = ""
    if has_vel:
        vels = [math.hypot(float(r[7]), float(r[8])) for r in data]
        mean_vel = sum(vels)/n
        vel_str = f" | mean term vel {mean_vel*100:.1f} cm/s"
        
    print(f"[collect] {n} episodes | mean dist err {mean*100:.1f}cm | median {med*100:.1f}cm | >40cm: {big}{vel_str}")
PY
