#!/usr/bin/env bash
# Collect P2P noise-test results ONE EPISODE PER FRESH PROCESS.
#
# Usage:
#   bash collect_p2pt_noise.sh <TASK> <NOISE_MODEL> <NOISE_K> [NUM] [CHECKPOINT] [RESULTS_CSV] [COMMANDS_CSV]

set -u

TASK="${1:?need task id as arg 1}"
NOISE_MODEL="${2:?need noise_model (none|gaussian|drift) as arg 2}"
NOISE_K="${3:?need noise_k as arg 3}"
NUM="${4:-100}"
CKPT="${5:-}"
# UPDATE: Prepended 'data/' to the default output path
RESULTS_CSV="${6:-data/results_p2pt_${NOISE_MODEL}_k${NOISE_K}.csv}"
COMMANDS_CSV="${7:-p2p_commands.csv}"

SCRIPT="scripts/rsl_rl/play_p2pt_single.py"

# build the optional --checkpoint flag only when a path was given
CKPT_ARG=()
if [[ -n "${CKPT}" ]]; then 
    CKPT_ARG=(--checkpoint "${CKPT}")
fi

# build the noise flag for the pose-command drift model
NOISE_ARG=()
if [[ "${NOISE_MODEL}" != "none" && "${NOISE_K}" != "0" ]]; then
    NOISE_ARG=(--noise_model "${NOISE_MODEL}" --noise_k "${NOISE_K}")
fi

# UPDATE: Create the data directory if it doesn't exist
mkdir -p data

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
    print(f"[collect] {n} episodes | mean dist err {mean*100:.1f}cm | median {med*100:.1f}cm | >40cm: {big}")
PY