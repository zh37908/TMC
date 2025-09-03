#!/usr/bin/env bash
set -euo pipefail

# Absolute paths preferred
PYTHON="/home/hzhaobi/miniconda3/envs/multired/bin/python"
ROOT="/home/hzhaobi/Multired"
TRAINER="$ROOT/TMC/ETMC_TPAMI/train_TMC_any_nyud2.py"

declare -a MODEL_KINDS=("evidential" "base")
declare -a PRETRAINS=("DeCUR" "SimCLR" "BarlowTwins" "No_pretrain")
declare -a SNRS=(20)

for model_kind in "${MODEL_KINDS[@]}"; do
  for pre in "${PRETRAINS[@]}"; do
    for snr in "${SNRS[@]}"; do
      for run in 1; do
        NAME="train_TMC_any_${model_kind}_${pre}_static_snr${snr}_run${run}"
        if [[ "$model_kind" == "evidential" ]]; then
          BASE_SAVEDIR="$ROOT/savepath/TMC_channel/nyud"
        else
          BASE_SAVEDIR="$ROOT/savepath/TMCBase_channel/nyud"
        fi
        EXP_SAVEDIR="$BASE_SAVEDIR/${pre}/snr${snr}/run${run}"
        echo "=== Running ${model_kind} | pretrain=${pre} | static SNR=${snr} dB | run=${run} ==="
        "$PYTHON" "$TRAINER" \
          --model_kind "$model_kind" \
          --channel_mode static \
          --pretrain "$pre" \
          --channel_snr "$snr" \
          --savedir "$EXP_SAVEDIR" \
          --name "$NAME" \
          --seed "$run"
      done
    done
  done
done


