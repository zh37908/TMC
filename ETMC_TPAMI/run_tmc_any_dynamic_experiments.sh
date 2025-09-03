#!/usr/bin/env bash
set -euo pipefail

# Absolute paths preferred
PYTHON="/home/hzhaobi/miniconda3/envs/multired/bin/python"
ROOT="/home/hzhaobi/Multired"
TRAINER="$ROOT/TMC/ETMC_TPAMI/train_TMC_any_nyud2.py"

declare -a MODEL_KINDS=("evidential" "base")
declare -a PRETRAINS=("DeCUR" "SimCLR" "BarlowTwins" "No_pretrain")

# SNR 区间（可用环境变量覆盖）
SNR_MIN=${SNR_MIN:-0}
SNR_MAX=${SNR_MAX:-20}
RANGE_TAG="range${SNR_MIN}-${SNR_MAX}"

for model_kind in "${MODEL_KINDS[@]}"; do
  for pre in "${PRETRAINS[@]}"; do
    for run in 1 2 3; do
      NAME="train_TMC_any_${model_kind}_${pre}_${RANGE_TAG}_dynamic_run${run}"
      if [[ "$model_kind" == "evidential" ]]; then
        BASE_SAVEDIR="$ROOT/savepath/TMC_channel_dynamic/nyud"
      else
        BASE_SAVEDIR="$ROOT/savepath/TMCBase_channel_dynamic/nyud"
      fi
      EXP_SAVEDIR="$BASE_SAVEDIR/${pre}/${RANGE_TAG}/dynamic/run${run}"
      echo "=== Running ${model_kind} | pretrain=${pre} | dynamic SNR [${SNR_MIN}, ${SNR_MAX}] dB | run=${run} ==="
      "$PYTHON" "$TRAINER" \
        --model_kind "$model_kind" \
        --channel_mode dynamic \
        --pretrain "$pre" \
        --snr_min "$SNR_MIN" \
        --snr_max "$SNR_MAX" \
        --savedir "$EXP_SAVEDIR" \
        --name "$NAME" \
        --seed "$run"
    done
  done
done


