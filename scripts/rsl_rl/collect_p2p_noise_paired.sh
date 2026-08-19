#!/usr/bin/env bash
# Paired clean-vs-noised P2P collection.
#
# For each command in the shared CSV, runs the P2P policy TWICE in fresh
# processes (the verified-reliable one-episode-per-process pattern):
#   1) clean   (noise_model=none)          -> results_p2p_clean.csv
#   2) noised  (noise_model=$MODEL, k=$K)  -> results_p2p_noise.csv
# Both use the SAME command row, so the two files are paired row-for-row.
#
# Usage:
#   bash collect_p2p_noise_paired.sh <TASK> <MODEL> <K> [NUM] [CHECKPOINT]
#     MODEL = gaussian | drift
#     K     = noise scale (1, 2, 3 ...)
#
# Example:
#   bash collect_p2p_noise_paired.sh Isaac-Go1-P2P_Rel_Time-Play-v0 gaussian 1 100

set -u
TASK="${1:?need task id as arg 1}"
MODEL="${2:?need noise model (gaussian|drift) as arg 2}"
K="${3:?need noise scale k as arg 3}"
NUM="${4:-100}"
CKPT="${5:-}"
COMMANDS_CSV="${6:-p2p_commands.csv}"
CLEAN_CSV="${7:-results_p2p_clean.csv}"
NOISE_CSV="${8:-results_p2p_${MODEL}_k${K}.csv}"

SCRIPT="scripts/rsl_rl/collect_p2p_noise.py"

CKPT_ARG=()
if [[ -n "${CKPT}" ]]; then CKPT_ARG=(--checkpoint "${CKPT}"); fi

rm -f "${CLEAN_CSV}" "${NOISE_CSV}"
echo "[paired] task=${TASK} model=${MODEL} k=${K} episodes=${NUM}"
echo "[paired] clean -> ${CLEAN_CSV}   noised -> ${NOISE_CSV}"

for (( i=0; i<NUM; i++ )); do
    echo "=========== EPISODE ${i}/${NUM}  (clean) ==========="
    python "${SCRIPT}" --task "${TASK}" ${CKPT_ARG[@]+"${CKPT_ARG[@]}"} \
        --commands_csv "${COMMANDS_CSV}" --results_csv "${CLEAN_CSV}" \
        --play_episode "${i}" --inject_commands --append --headless \
        --noise_model none --noise_k 0

    echo "=========== EPISODE ${i}/${NUM}  (noised: ${MODEL} k=${K}) ==========="
    python "${SCRIPT}" --task "${TASK}" ${CKPT_ARG[@]+"${CKPT_ARG[@]}"} \
        --commands_csv "${COMMANDS_CSV}" --results_csv "${NOISE_CSV}" \
        --play_episode "${i}" --inject_commands --append --headless \
        --noise_model "${MODEL}" --noise_k "${K}" --noise_seed "${i}"
done

echo "[paired] DONE."
echo "[paired] compare with:"
echo "  python scripts/rsl_rl/compare_noise.py --clean ${CLEAN_CSV} --noise ${NOISE_CSV} --out_prefix noise_${MODEL}_k${K}"
