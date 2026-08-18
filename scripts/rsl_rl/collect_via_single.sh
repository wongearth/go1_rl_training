#!/usr/bin/env bash
# Collect velocity-P2P results by running the VERIFIED-CORRECT single-play script
# once per command, each in a FRESH process. This sidesteps the batch-collector's
# cross-episode state accumulation (the ep13 freeze) entirely: every episode gets
# a brand-new interpreter, env, and policy, exactly like the interactive run that
# always works.
#
# Usage:
#   bash collect_via_single.sh <TASK> <CHECKPOINT> [NUM] [COMMANDS_CSV] [RESULTS_CSV]
#
# Example:
#   bash collect_via_single.sh Isaac-Go1-Velocity-Flat-Play-v0 /path/to/policy.pt 100
#
# Notes:
# - Slower than a single batch run (one sim boot per episode), but bulletproof.
# - Results accumulate into RESULTS_CSV via --append.

set -u

TASK="${1:?need task id as arg 1}"
CKPT="${2:-}"   # optional: if empty, the play script auto-finds the latest checkpoint
NUM="${3:-100}"
COMMANDS_CSV="${4:-p2p_commands.csv}"
RESULTS_CSV="${5:-results_velocity.csv}"

SCRIPT="scripts/rsl_rl/play_vel_p2p_single.py"

# build the optional --checkpoint flag only when a path was given
CKPT_ARG=()
if [[ -n "${CKPT}" ]]; then
    CKPT_ARG=(--checkpoint "${CKPT}")
fi

# fresh results file (remove so --append starts clean with a header)
rm -f "${RESULTS_CSV}"

echo "[collect] task=${TASK}"
echo "[collect] commands=${COMMANDS_CSV}  results=${RESULTS_CSV}  episodes=${NUM}"

for (( i=0; i<NUM; i++ )); do
    echo "=================== EPISODE ${i} / ${NUM} ==================="
    python "${SCRIPT}" \
        --task "${TASK}" \
        ${CKPT_ARG[@]+"${CKPT_ARG[@]}"} \
        --commands_csv "${COMMANDS_CSV}" \
        --results_csv "${RESULTS_CSV}" \
        --play_episode "${i}" \
        --once --append --headless
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
rows = list(csv.reader(open(p)))
data = rows[1:]
n = len(data)
errs = [math.hypot(float(r[4]), float(r[5])) for r in data]
big = sum(1 for e in errs if e > 0.4)
if n:
    errs_sorted = sorted(errs)
    mean = sum(errs)/n
    med = errs_sorted[n//2]
    print(f"[collect] {n} episodes | mean dist err {mean*100:.1f}cm | median {med*100:.1f}cm | >40cm: {big}")
PY
