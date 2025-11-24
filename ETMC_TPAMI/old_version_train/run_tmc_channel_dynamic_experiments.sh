#!/usr/bin/env bash
set -euo pipefail

# Absolute paths preferred
PYTHON="/home/hzhaobi/miniconda3/envs/multired/bin/python"
ROOT="/home/hzhaobi/Multired"

declare -a MODELS=(
  "$ROOT/TMC/ETMC_TPAMI/train_TMC_channel_dynamic_nyud2.py"
  "$ROOT/TMC/ETMC_TPAMI/train_TMC_base_channel_dynamic_nyud2.py"
)

declare -a PRETRAINS=("DeCUR" "SimCLR" "BarlowTwins" "No_pretrain")

# SNR 区间（可按需修改）
SNR_MIN=${SNR_MIN:-0}
SNR_MAX=${SNR_MAX:-20}
RANGE_TAG="range${SNR_MIN}-${SNR_MAX}"

for model in "${MODELS[@]}"; do
  model_base=$(basename "$model" .py)
  for pre in "${PRETRAINS[@]}"; do
    for run in 1 2 3; do
      NAME="${model_base}_${pre}_${RANGE_TAG}_dynamic_run${run}"
      # decide base savedir by model type
      if [[ "$model_base" == "train_TMC_channel_dynamic_nyud2" ]]; then
        BASE_SAVEDIR="$ROOT/savepath/TMC_channel_dynamic/nyud"
      else
        BASE_SAVEDIR="$ROOT/savepath/TMCBase_channel_dynamic/nyud"
      fi
      # enrich savedir path with pretrain/dynamic/run information as nested folders
      EXP_SAVEDIR="$BASE_SAVEDIR/${pre}/${RANGE_TAG}/dynamic/run${run}"
      echo "=== Running ${model_base} | pretrain=${pre} | dynamic SNR [${SNR_MIN}, ${SNR_MAX}] dB | run=${run} ==="
      "$PYTHON" "$model" \
        --pretrain "$pre" \
        --snr_min "$SNR_MIN" \
        --snr_max "$SNR_MAX" \
        --savedir "$EXP_SAVEDIR" \
        --name "$NAME" \
        --seed "$run"
    done
  done
done


