#!/usr/bin/env bash
set -euo pipefail

# Absolute paths preferred
PYTHON="/home/hzhaobi/miniconda3/envs/multired/bin/python"
ROOT="/home/hzhaobi/Multired"

declare -a MODELS=(
  "$ROOT/TMC/ETMC_TPAMI/train_TMC_channel_nyud2.py"
  "$ROOT/TMC/ETMC_TPAMI/train_TMC_base_channel_nyud2.py"
)

declare -a PRETRAINS=("DeCUR" "SimCLR" "BarlowTwins" "No_pretrain")
declare -a SNRS=(10 20)

for model in "${MODELS[@]}"; do
  model_base=$(basename "$model" .py)
  for pre in "${PRETRAINS[@]}"; do
    for snr in "${SNRS[@]}"; do
      for run in 1 2 3; do
        NAME="${model_base}_${pre}_snr${snr}_run${run}"
        # decide base savedir by model type
        if [[ "$model_base" == "train_TMC_channel_nyud2" ]]; then
          BASE_SAVEDIR="$ROOT/savepath/TMC_channel/nyud"
        else
          BASE_SAVEDIR="$ROOT/savepath/TMCBase_channel/nyud"
        fi
        # enrich savedir path with pretrain/snr/run information as nested folders
        EXP_SAVEDIR="$BASE_SAVEDIR/${pre}/snr${snr}/run${run}"
        echo "=== Running ${model_base} | pretrain=${pre} | snr=${snr} | run=${run} ==="
        "$PYTHON" "$model" \
          --pretrain "$pre" \
          --channel_snr "$snr" \
          --savedir "$EXP_SAVEDIR" \
          --name "$NAME" \
          --seed "$run"
      done
    done
  done
done


