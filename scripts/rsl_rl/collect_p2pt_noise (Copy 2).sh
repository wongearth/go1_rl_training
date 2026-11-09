#!/usr/bin/env bash
# Collect P2P noise-test results ONE EPISODE PER FRESH PROCESS.
#
# The multi-episode single-process run has a sequence-dependent failure (a
# command that works in isolation fails deep in the sequence). Running each
# episode in its own process eliminates all cross-episode state -- this is the
# same verified-reliable pattern used for the velocity collection.
#
# Usage:
#   bash collect_p2p_noise_single.sh <TASK> <SENSOR_NOISE> <SENSOR_K> [NUM] [CHECKPOINT] [RESULTS_CSV]
#
#   SENSOR_NOISE : none | lin_vel | ang_vel | gravity | all   (or comma list)
#   SENSOR_K     : noise scale (0 for clean, 1/2/3 ...)
#
# Examples:
#   # clean baseline:
#   bash collect_p2p_noise_single.sh Isaac-Go1-P2P_Rel_Time-Play-v0 none 0 100 "" results_p2p_clean.csv
#   # all-sensors k=3:
#   bash collect_p2p_noise_single.sh Isaac-Go1-P2P_Rel_Time-Play-v0 all 3 100 "" results_p2p_allsensors_k3.csv

set -u
TASK="${1:?need task id as arg 1}"
SENSOR_NOISE="${2:?need sensor_noise (none|lin_vel|ang_vel|gravity|all) as arg 2}"
SENSOR_K="${3:?need sensor_k as arg 3}"
NUM="${4:-100}"
CKPT="${5:-}"
RESULTS_CSV="${6:-results_p2p_${SENSOR_NOISE}_k${SENSOR_K}.csv}"
COMMANDS_CSV="${7:-p2p_commands.csv}"

# SCRIPT="scripts/rsl_rl/collect_p2p_noise.py"
SCRIPT="scripts/rsl_rl/play_p2pt_single.py"

CKPT_ARG=()
if [[ -n "${CKPT}" ]]; then CKPT_ARG=(--checkpoint "${CKPT}"); fi

# # # build the sensor-noise flag only when noise is actually wanted
# NOISE_ARG=()
# if [[ "${SENSOR_NOISE}" != "none" && "${SENSOR_K}" != "0" ]]; then
#     NOISE_ARG=(--sensor_noise "${SENSOR_NOISE}" --sensor_k "${SENSOR_K}")
# fi
# # Temporarily override for drift testing
# NOISE_ARG=(--noise_model drift --noise_k 2.0)
NOISE_ARG=()
if [[ "${SENSOR_NOISE}" == "drift" && "${SENSOR_K}" != "0" ]]; then
    # Route to the pose-command drift model
    NOISE_ARG=(--noise_model drift --noise_k "${SENSOR_K}")
elif [[ "${SENSOR_NOISE}" != "none" && "${SENSOR_K}" != "0" ]]; then
    # Route to the raw physics sensor noise model
    NOISE_ARG=(--sensor_noise "${SENSOR_NOISE}" --sensor_k "${SENSOR_K}")
fi

rm -f "${RESULTS_CSV}"
echo "[single] task=${TASK} sensor_noise=${SENSOR_NOISE} k=${SENSOR_K} episodes=${NUM}"
echo "[single] results -> ${RESULTS_CSV}"

for (( i=0; i<NUM; i++ )); do
    echo "=================== EPISODE ${i}/${NUM} ==================="
    python "${SCRIPT}" --task "${TASK}" ${CKPT_ARG[@]+"${CKPT_ARG[@]}"} \
        --commands_csv "${COMMANDS_CSV}" --results_csv "${RESULTS_CSV}" \
        --play_episode "${i}" --append --headless --once \
        ${NOISE_ARG[@]+"${NOISE_ARG[@]}"}

    rc=$?

    if [[ $rc -ne 0 ]]; then
        echo "[single] WARNING: episode ${i} exited code ${rc}; continuing."
    fi
done

echo "[single] DONE -> ${RESULTS_CSV}"
# quick tally
python - "${RESULTS_CSV}" <<'PY'
import csv, sys, math
p = sys.argv[1]
data = list(csv.reader(open(p)))[1:]
n = len(data)
if n:
    errs = [math.hypot(float(r[4]), float(r[5])) for r in data]
    errs_s = sorted(errs)
    mean = sum(errs)/n
    med = errs_s[n//2]
    big = sum(1 for e in errs if e > 0.4)
    print(f"[single] {n} eps | mean {mean*100:.1f}cm | median {med*100:.1f}cm | max {max(errs)*100:.1f}cm | >40cm: {big}")
PY
