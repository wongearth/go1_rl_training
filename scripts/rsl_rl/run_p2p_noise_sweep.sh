#!/usr/bin/env bash
# Single-process P2P noise sweep (P2P runs clean in one process, so no per-episode
# restart needed -- much faster than the paired shell loop).
#
# Runs the CLEAN baseline once, then each requested (model, k) as its own
# single-process run. All share the same commands, so every noised file is paired
# row-for-row with the one clean file.
#
# Usage:
#   bash run_p2p_noise_sweep.sh <TASK> [NUM] [CHECKPOINT]
#
# Edit MODELS / KS below to choose the sweep.

set -u
TASK="${1:?need task id as arg 1}"
NUM="${2:-100}"
CKPT="${3:-}"
COMMANDS_CSV="${4:-p2p_commands.csv}"

SCRIPT="scripts/rsl_rl/collect_p2p_noise.py"
CLEAN_CSV="results_p2p_clean.csv"

# ---- choose the sweep here ----
MODELS=(gaussian drift)
KS=(1 2 3)
# --------------------------------

CKPT_ARG=()
if [[ -n "${CKPT}" ]]; then CKPT_ARG=(--checkpoint "${CKPT}"); fi

run_one () {  # args: model k out_csv
    local model="$1" k="$2" out="$3"
    echo "=================================================="
    echo "[sweep] model=${model} k=${k} -> ${out}"
    echo "=================================================="
    python "${SCRIPT}" --task "${TASK}" ${CKPT_ARG[@]+"${CKPT_ARG[@]}"} \
        --num_episodes "${NUM}" \
        --commands_csv "${COMMANDS_CSV}" --results_csv "${out}" \
        --inject_commands --headless \
        --noise_model "${model}" --noise_k "${k}"
}

# 1) clean baseline once
run_one none 0 "${CLEAN_CSV}"

# 2) every (model, k)
for model in "${MODELS[@]}"; do
    for k in "${KS[@]}"; do
        run_one "${model}" "${k}" "results_p2p_${model}_k${k}.csv"
    done
done

echo ""
echo "[sweep] DONE. Compare each against ${CLEAN_CSV}, e.g.:"
for model in "${MODELS[@]}"; do
    for k in "${KS[@]}"; do
        echo "  python scripts/rsl_rl/compare_noise.py --clean ${CLEAN_CSV} --noise results_p2p_${model}_k${k}.csv --out_prefix noise_${model}_k${k}"
    done
done
